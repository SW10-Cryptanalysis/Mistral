import json
import logging
import pytest
import torch
from pathlib import Path

from src import evaluate
from src.config import cfg as real_cfg


@pytest.fixture
def dummy_cfg():
    original_out = real_cfg.output_dir
    original_val = real_cfg.tokenized_val_dir
    original_use_spaces = real_cfg.use_spaces

    real_cfg.output_dir = Path("dummy/output/dir")
    real_cfg.tokenized_val_dir = Path("dummy/eval/dir")

    yield real_cfg

    real_cfg.output_dir = original_out
    real_cfg.tokenized_val_dir = original_val
    real_cfg.use_spaces = original_use_spaces


def test_evaluate_model_not_found(mocker, caplog, dummy_cfg):
    mock_exists = mocker.patch("src.evaluate.os.path.exists")
    mock_exists.return_value = False

    evaluate.evaluate()

    assert "Model path not found" in caplog.text


def test_evaluate_no_test_files(mocker, caplog, dummy_cfg):
    mock_exists = mocker.patch("src.evaluate.os.path.exists")
    mocker.patch("src.evaluate.MistralForCausalLM.from_pretrained")

    mock_dir = mocker.Mock(spec=Path)
    mock_dir.glob.return_value = []

    mocker.patch("src.evaluate.cfg.use_spaces", True)
    mocker.patch("src.evaluate.cfg.tokenized_spaced_test_dir", mock_dir)

    mock_exists.return_value = True

    evaluate.evaluate()

    assert "No test files found" in caplog.text


def test_evaluate_full_execution(mocker, caplog, dummy_cfg):
    caplog.set_level(logging.INFO, logger="evaluate.py")
    mock_exists = mocker.patch("src.evaluate.os.path.exists")
    mock_from_pretrained = mocker.patch(
        "src.evaluate.MistralForCausalLM.from_pretrained"
    )

    mock_dir = mocker.Mock(spec=Path)
    mock_dir.glob.return_value = [Path("dummy_test_1.json")]

    mocker.patch("src.evaluate.cfg.use_spaces", True)
    mocker.patch("src.evaluate.cfg.tokenized_spaced_test_dir", mock_dir)

    mock_exists.return_value = True

    mock_model = mocker.Mock()
    mock_from_pretrained.return_value = mock_model

    sep_token = dummy_cfg.sep_token_id
    char_offset = dummy_cfg.char_offset

    generated_tokens = [char_offset + i for i in range(3)]  # maps strictly to 'abc'
    input_cipher_ids = [100, 101, 102]
    mock_generated_sequence = input_cipher_ids + [sep_token] + generated_tokens
    mock_model.generate.return_value = torch.tensor([mock_generated_sequence])

    mock_file_content = json.dumps({"ciphertext": "100 101 102", "plaintext": "abc"})

    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_file_content))
    evaluate.evaluate()

    assert "Pred Plaintext: abc" in caplog.text
    assert "Symbol Error Rate (SER): 0.0000" in caplog.text


def test_evaluate_use_spaces_enabled(mocker, caplog, dummy_cfg):
    """Test that spaced directory is used when cfg.use_spaces is True."""
    mocker.patch("src.evaluate.os.path.exists", return_value=True)
    mocker.patch("src.evaluate.MistralForCausalLM.from_pretrained")

    mocker.patch("src.evaluate.cfg.use_spaces", True)

    mock_spaced_dir = mocker.Mock(spec=Path)
    mock_spaced_dir.glob.return_value = []
    mocker.patch("src.evaluate.cfg.tokenized_spaced_test_dir", mock_spaced_dir)

    evaluate.evaluate()

    mock_spaced_dir.glob.assert_called_once_with("*.json")
    assert "No test files found" in caplog.text


def test_evaluate_use_spaces_disabled(mocker, caplog, dummy_cfg):
    """Test that normal directory is used when cfg.use_spaces is False."""
    mocker.patch("src.evaluate.os.path.exists", return_value=True)
    mocker.patch("src.evaluate.MistralForCausalLM.from_pretrained")

    mocker.patch("src.evaluate.cfg.use_spaces", False)

    mock_normal_dir = mocker.Mock(spec=Path)
    mock_normal_dir.glob.return_value = []
    mocker.patch("src.evaluate.cfg.tokenized_test_dir", mock_normal_dir)

    evaluate.evaluate()

    mock_normal_dir.glob.assert_called_once_with("*.json")
    assert "No test files found" in caplog.text
