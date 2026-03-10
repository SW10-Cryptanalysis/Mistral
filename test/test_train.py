import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Import the module to be tested
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
        {"input_ids": [1, 2, 3, 4, 5, 6, 7], "labels": [1, 2, 3, 4, 5, 6, 7]}, # Longer than max_context (5)
        {"input_ids": [1, 2], "labels": [1, 2]},                               # Shorter than max_context
    ]

# =============================================================================
# 1. Dataset Logic Tests
# =============================================================================

@patch("src.train.load_from_disk")
def test_pretokenized_cipher_dataset(mock_load, mock_hf_dataset, dummy_cfg):
    mock_load.return_value = mock_hf_dataset

    with patch("src.train.cfg", dummy_cfg):
        dataset = train.PretokenizedCipherDataset("dummy/path")

        # Test __len__
        assert len(dataset) == 2

        # Test __getitem__ and truncation logic
        item_0 = dataset[0]
        assert isinstance(item_0["input_ids"], torch.Tensor)
        assert isinstance(item_0["labels"], torch.Tensor)

        # Should be truncated to max_context (5)
        assert item_0["input_ids"].tolist() == [1, 2, 3, 4, 5]
        assert item_0["labels"].tolist() == [1, 2, 3, 4, 5]

        # Shorter sequences should remain untouched (padding happens in collator)
        item_1 = dataset[1]
        assert item_1["input_ids"].tolist() == [1, 2]

# =============================================================================
# 2. Collator Logic Tests
# =============================================================================

def test_safe_pad_collate(dummy_cfg):
    # Simulate a batch fetched from the dataset
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([4, 5, 6])},
        {"input_ids": torch.tensor([1]), "labels": torch.tensor([4])}
    ]

    with patch("src.train.cfg", dummy_cfg):
        collated = train.safe_pad_collate(batch)

    # Check shapes (batch_size=2, max_seq_len=3)
    assert collated["input_ids"].shape == (2, 3)
    assert collated["labels"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)

    # Check input padding (pad_token_id = 0)
    assert collated["input_ids"][1].tolist() == [1, 0, 0]

    # Check label padding (-100 for CrossEntropyLoss to ignore)
    assert collated["labels"][1].tolist() == [4, -100, -100]

    # Check attention mask (1 for real tokens, 0 for pad)
    assert collated["attention_mask"][1].tolist() == [1, 0, 0]

# =============================================================================
# 3. Metrics Tests
# =============================================================================

def test_compute_metrics():
    # Shape: (batch_size=2, seq_len=3, vocab_size=4)
    # We will engineer logits so argmax picks specific predictions.
    logits = np.zeros((2, 3, 4))

    # Batch 0 predictions: [1, 2, 3]
    logits[0, 0, 1] = 1
    logits[0, 1, 2] = 1
    logits[0, 2, 3] = 1

    # Batch 1 predictions: [1, 2, 0]
    logits[1, 0, 1] = 1
    logits[1, 1, 2] = 1
    logits[1, 2, 0] = 1

    # Define labels. -100 should be ignored.
    labels = np.array([
        [1, 2, 0],    # 1 mismatch at the end (pred: 3, true: 0)
        [1, -100, 0]  # Exact match for evaluated tokens (pred: [1, _, 0], true: [1, _, 0])
    ])

    eval_preds = (logits, labels)
    metrics = train.compute_metrics(eval_preds)

    # Total evaluated symbols = 3 (from batch 0) + 2 (from batch 1 ignoring -100) = 5
    # Total errors = 1 (the last token of batch 0)
    # Expected SER = 1 / 5 = 0.2
    assert np.isclose(metrics["ser"], 0.2)

def test_compute_metrics_zero_symbols():
    # Edge case: All labels are -100 to prevent ZeroDivisionError
    logits = np.zeros((1, 2, 4))
    labels = np.array([[-100, -100]])

    metrics = train.compute_metrics((logits, labels))
    assert metrics["ser"] == 0

# =============================================================================
# 4. Training Loop Execution Tests
# =============================================================================

@patch("src.train.cfg.bf16", False) # Disables bf16 to prevent ValueError on CPU test env
@patch("src.train.get_model")
@patch("src.train.PretokenizedCipherDataset")
@patch("src.train.Trainer")
@patch("src.train.get_last_checkpoint")
def test_train_execution(mock_get_checkpoint, mock_trainer_class, mock_dataset, mock_get_model):
    # Setup mocks
    mock_model_instance = MagicMock()
    mock_get_model.return_value = mock_model_instance
    mock_get_checkpoint.return_value = "dummy/checkpoint/path"

    mock_trainer_instance = MagicMock()

    # Ensure is_world_process_zero returns True so print statements trigger (for coverage)
    mock_trainer_instance.is_world_process_zero.return_value = True
    mock_trainer_class.return_value = mock_trainer_instance

    # Execute the train function
    train.train()

    # Assert gradient checkpointing was enabled
    mock_model_instance.gradient_checkpointing_enable.assert_called_once()

    # Assert Trainer was instantiated
    mock_trainer_class.assert_called_once()

    # Assert train was called and resumed from the mocked checkpoint
    mock_trainer_instance.train.assert_called_once_with(resume_from_checkpoint="dummy/checkpoint/path")

    # Assert model saving was called
    mock_trainer_instance.save_model.assert_called_once()
