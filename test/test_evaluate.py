import json
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src import evaluate
from src.config import cfg as real_cfg

@pytest.fixture
def dummy_cfg():
    original_out = real_cfg.output_dir
    original_val = real_cfg.val_dir

    real_cfg.output_dir = Path("dummy/output/dir")
    real_cfg.val_dir = Path("dummy/eval/dir")

    yield real_cfg

    real_cfg.output_dir = original_out
    real_cfg.val_dir = original_val

@patch("src.evaluate.os.path.exists")
def test_evaluate_model_not_found(mock_exists, caplog, dummy_cfg):
    mock_exists.return_value = False

    evaluate.evaluate()

    # Check the log records instead of stdout
    assert "Model path not found" in caplog.text

@patch("src.evaluate.os.path.exists")
@patch("src.evaluate.glob.glob")
@patch("src.evaluate.MistralForCausalLM.from_pretrained")
def test_evaluate_no_test_files(mock_from_pretrained, mock_glob, mock_exists, caplog, dummy_cfg):
    mock_exists.return_value = True
    mock_glob.return_value = []

    evaluate.evaluate()

    assert "No test files found" in caplog.text

@patch("src.evaluate.os.path.exists")
@patch("src.evaluate.glob.glob")
@patch("src.evaluate.MistralForCausalLM.from_pretrained")
def test_evaluate_full_execution(mock_from_pretrained, mock_glob, mock_exists, caplog, dummy_cfg):
    mock_exists.return_value = True
    mock_glob.return_value = ["dummy_test_1.json"]

    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model

    # Setup dummy tokens
    sep_token = dummy_cfg.unique_homophones + 1
    char_offset = sep_token + 1

    # Mocking the model response: [input] + [sep] + [generated]
    generated_tokens = [char_offset + i for i in range(3)] # maps to 'abc'
    input_cipher_ids = [100, 101, 102]
    mock_generated_sequence = input_cipher_ids + [sep_token] + generated_tokens
    mock_model.generate.return_value = torch.tensor([mock_generated_sequence])

    mock_file_content = json.dumps({
        "ciphertext": "100 101 102",
        "plaintext": "abc"
    })

    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        evaluate.evaluate()

    # Now we verify actual content!
    assert "Pred Plaintext: abc" in caplog.text
    assert "Symbol Error Rate (SER): 0.0000" in caplog.text
