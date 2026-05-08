import pytest
import torch
import numpy as np
from pathlib import Path
from src import train
from src.config import Config


@pytest.fixture
def dummy_cfg(mocker):
    """Provides a small config to test truncations and padding."""
    cfg = Config()
    # Fixed namespace: Patch the property directly on the class where it is defined
    mocker.patch(
        "src.config.Config.max_context",
        new_callable=mocker.PropertyMock,
        return_value=5,
    )
    cfg.pad_token_id = 0
    return cfg


@pytest.fixture
def mock_hf_dataset():
    """Simulates a loaded Hugging Face dataset."""
    return [
        {"input_ids": [1, 2, 3, 4, 5, 6, 7], "labels": [1, 2, 3, 4, 5, 6, 7]},
        {"input_ids": [1, 2], "labels": [1, 2]},
    ]


def test_pretokenized_cipher_dataset(mocker, mock_hf_dataset, dummy_cfg):
    mock_load = mocker.patch("src.train.load_from_disk")
    mock_load.return_value = mock_hf_dataset

    mocker.patch("src.train.cfg", dummy_cfg)
    dataset = train.PretokenizedCipherDataset("dummy/path")

    assert len(dataset) == 2

    item_0 = dataset[0]
    assert isinstance(item_0["input_ids"], torch.Tensor)
    assert isinstance(item_0["labels"], torch.Tensor)

    assert item_0["input_ids"].tolist() == [1, 2, 3, 4, 5]
    assert item_0["labels"].tolist() == [1, 2, 3, 4, 5]

    item_1 = dataset[1]
    assert item_1["input_ids"].tolist() == [1, 2]


def test_safe_pad_collate(mocker, dummy_cfg):
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([4, 5, 6])},
        {"input_ids": torch.tensor([1]), "labels": torch.tensor([4])},
    ]

    mocker.patch("src.train.cfg", dummy_cfg)
    collated = train.safe_pad_collate(batch)

    assert collated["input_ids"].shape == (2, 3)
    assert collated["labels"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)
    assert collated["input_ids"][1].tolist() == [1, 0, 0]
    assert collated["labels"][1].tolist() == [4, -100, -100]
    assert collated["attention_mask"][1].tolist() == [1, 0, 0]


def test_compute_metrics(mocker):
    # Mock token IDs to guarantee the SEP token is found without requiring metadata.json
    mocker.patch(
        "src.config.Config.sep_token_id",
        new_callable=mocker.PropertyMock,
        return_value=20,
    )
    mocker.patch(
        "src.config.Config.eos_token_id",
        new_callable=mocker.PropertyMock,
        return_value=999,
    )

    logits = np.zeros((2, 5, 100))

    # Sample 0
    logits[0, 0, 20] = 1
    logits[0, 1, 30] = 1
    logits[0, 2, 40] = 1
    logits[0, 3, 99] = 1  # Intentionally wrong prediction to create 1 error
    logits[0, 4, 0] = 1

    # Sample 1
    logits[1, 0, 20] = 1
    logits[1, 1, 30] = 1
    logits[1, 2, 40] = 1
    logits[1, 3, 0] = 1
    logits[1, 4, 0] = 1

    labels = np.array([[10, 20, 30, 40, 50], [10, 20, 30, 40, -100]])

    # 5 Valid Output Symbols Expected. Sample 0 will get 1 Error. Total: 1/5 = 0.2
    eval_preds = (logits, labels)
    metrics = train.compute_metrics(eval_preds)

    assert np.isclose(metrics["ser"], 0.2)


def test_compute_metrics_zero_symbols():
    logits = np.zeros((1, 2, 4))
    labels = np.array([[-100, -100]])

    metrics = train.compute_metrics((logits, labels))
    assert metrics["ser"] == 0.0


def test_train_execution(mocker):
    mocker.patch("src.train.cfg.bf16", False)
    mock_get_model = mocker.patch("src.train.get_model")
    mocker.patch("src.train.PretokenizedCipherDataset")
    mock_trainer_class = mocker.patch("src.train.Trainer")
    mock_get_checkpoint = mocker.patch("src.train.get_last_checkpoint")

    mock_model_instance = mocker.Mock()
    mock_get_model.return_value = mock_model_instance
    mock_get_checkpoint.return_value = "dummy/checkpoint/path"

    mock_trainer_instance = mocker.Mock()

    mock_trainer_instance.is_world_process_zero.return_value = True
    mock_trainer_class.return_value = mock_trainer_instance

    train.train()

    mock_model_instance.gradient_checkpointing_enable.assert_called_once()
    mock_trainer_class.assert_called_once()
    mock_trainer_instance.train.assert_called_once_with(
        resume_from_checkpoint="dummy/checkpoint/path"
    )
    mock_trainer_instance.save_model.assert_called_once()


def test_train_bfloat16_configuration(mocker):
    mocker.patch("src.train.cfg.bf16", True)

    mocker.patch("src.train.get_model")
    mocker.patch("src.train.PretokenizedCipherDataset")
    mocker.patch("src.train.Trainer")
    mocker.patch("src.train.get_last_checkpoint")

    mock_training_args = mocker.patch("src.train.TrainingArguments")

    train.train()

    mock_training_args.assert_called_once()
    _, kwargs = mock_training_args.call_args

    assert kwargs.get("bf16") is True


def test_train_use_spaces_enabled(mocker):
    """Test that spaced directories are used when cfg.use_spaces is True."""
    mocker.patch("src.train.cfg.use_spaces", True)

    spaced_train = Path("spaced/train")
    spaced_val = Path("spaced/val")
    mocker.patch(
        "src.config.Config.tokenized_train_dir",
        new_callable=mocker.PropertyMock,
        return_value=spaced_train,
    )
    mocker.patch(
        "src.config.Config.tokenized_val_dir",
        new_callable=mocker.PropertyMock,
        return_value=spaced_val,
    )

    mocker.patch("src.train.get_model")
    mock_ds_class = mocker.patch("src.train.PretokenizedCipherDataset")
    mock_trainer_class = mocker.patch("src.train.Trainer")
    mocker.patch("src.train.get_last_checkpoint")
    mocker.patch("src.train.TrainingArguments")
    mock_trainer_instance = mocker.Mock()
    mock_trainer_instance.is_world_process_zero.return_value = True
    mock_trainer_class.return_value = mock_trainer_instance

    train.train()

    # Check that constructor was called with the spaced paths
    mock_ds_class.assert_any_call(spaced_train)
    mock_ds_class.assert_any_call(spaced_val)

    mock_trainer_instance.save_model.assert_called_once()
    args, _ = mock_trainer_instance.save_model.call_args
    save_path = str(args[0])
    assert save_path.endswith("final_model_with_spaces")


def test_train_use_spaces_disabled(mocker):
    """Test that normal directories are used when cfg.use_spaces is False."""
    mocker.patch("src.train.cfg.use_spaces", False)

    normal_train = Path("normal/train")
    normal_val = Path("normal/val")
    mocker.patch(
        "src.config.Config.tokenized_train_dir",
        new_callable=mocker.PropertyMock,
        return_value=normal_train,
    )
    mocker.patch(
        "src.config.Config.tokenized_val_dir",
        new_callable=mocker.PropertyMock,
        return_value=normal_val,
    )

    mocker.patch("src.train.get_model")
    mock_ds_class = mocker.patch("src.train.PretokenizedCipherDataset")
    mock_trainer_class = mocker.patch("src.train.Trainer")
    mocker.patch("src.train.get_last_checkpoint")
    mocker.patch("src.train.TrainingArguments")
    mock_trainer_instance = mocker.Mock()
    mock_trainer_instance.is_world_process_zero.return_value = True
    mock_trainer_class.return_value = mock_trainer_instance

    train.train()

    # Check that constructor was called with the normal paths
    mock_ds_class.assert_any_call(normal_train)
    mock_ds_class.assert_any_call(normal_val)
    mock_trainer_instance.save_model.assert_called_once()
    args, _ = mock_trainer_instance.save_model.call_args
    save_path = str(args[0])
    assert save_path.endswith("final_model_no_spaces")
