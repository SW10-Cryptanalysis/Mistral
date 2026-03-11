import pytest
import torch
import numpy as np
from src import train
from src.config import Config

@pytest.fixture
def dummy_cfg():
    """Provides a small config to test truncations and padding."""
    cfg = Config()
    cfg.max_context = 5
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
        {"input_ids": torch.tensor([1]), "labels": torch.tensor([4])}
    ]

    mocker.patch("src.train.cfg", dummy_cfg)
    collated = train.safe_pad_collate(batch)

    assert collated["input_ids"].shape == (2, 3)
    assert collated["labels"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)
    assert collated["input_ids"][1].tolist() == [1, 0, 0]
    assert collated["labels"][1].tolist() == [4, -100, -100]
    assert collated["attention_mask"][1].tolist() == [1, 0, 0]

def test_compute_metrics():
    logits = np.zeros((2, 3, 4))

    logits[0, 0, 1] = 1
    logits[0, 1, 2] = 1
    logits[0, 2, 3] = 1

    logits[1, 0, 1] = 1
    logits[1, 1, 2] = 1
    logits[1, 2, 0] = 1

    labels = np.array([
        [1, 2, 0],
        [1, -100, 0]
    ])

    eval_preds = (logits, labels)
    metrics = train.compute_metrics(eval_preds)

    assert np.isclose(metrics["ser"], 0.2)

def test_compute_metrics_zero_symbols():
    logits = np.zeros((1, 2, 4))
    labels = np.array([[-100, -100]])

    metrics = train.compute_metrics((logits, labels))
    assert metrics["ser"] == 0

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
    mock_trainer_instance.train.assert_called_once_with(resume_from_checkpoint="dummy/checkpoint/path")
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
