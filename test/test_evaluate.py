import json
import pytest
import torch
from pathlib import Path

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

def test_evaluate_model_not_found(mocker, caplog, dummy_cfg):
    mock_exists = mocker.patch("src.evaluate.os.path.exists")
    mock_exists.return_value = False

    evaluate.evaluate()

    assert "Model path not found" in caplog.text

def test_evaluate_no_test_files(mocker, caplog, dummy_cfg):
    mock_exists = mocker.patch("src.evaluate.os.path.exists")
    mocker.patch("src.evaluate.MistralForCausalLM.from_pretrained")
    mock_glob = mocker.patch("src.evaluate.glob.glob")

    mock_exists.return_value = True
    mock_glob.return_value = []

    evaluate.evaluate()

    assert "No test files found" in caplog.text

def test_evaluate_full_execution(mocker, caplog, dummy_cfg):
    mock_exists = mocker.patch("src.evaluate.os.path.exists")
    mock_glob = mocker.patch("src.evaluate.glob.glob")
    mock_from_pretrained = mocker.patch("src.evaluate.MistralForCausalLM.from_pretrained")

    mock_exists.return_value = True
    mock_glob.return_value = ["dummy_test_1.json"]

    mock_model = mocker.Mock()
    mock_from_pretrained.return_value = mock_model

    sep_token = dummy_cfg.unique_homophones + 1
    char_offset = sep_token + 1

    generated_tokens = [char_offset + i for i in range(3)] # maps to 'abc'
    input_cipher_ids = [100, 101, 102]
    mock_generated_sequence = input_cipher_ids + [sep_token] + generated_tokens
    mock_model.generate.return_value = torch.tensor([mock_generated_sequence])

    mock_file_content = json.dumps({
        "ciphertext": "100 101 102",
        "plaintext": "abc"
    })

    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_file_content))
    evaluate.evaluate()

    assert "Pred Plaintext: abc" in caplog.text
    assert "Symbol Error Rate (SER): 0.0000" in caplog.text
