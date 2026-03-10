import json
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from src import evaluate
from src.config import cfg as real_cfg  # Import the actual cfg object

@pytest.fixture
def dummy_cfg():
	"""Provides a configuration with custom paths for testing."""
	# We modify the existing real_cfg temporarily to ensure the reference is the same
	original_val = getattr(real_cfg, "val_dir", None)
	original_out = real_cfg.output_dir
	
	real_cfg.output_dir = Path("dummy/output/dir")
	# Your evaluate.py uses cfg.eval_dir or cfg.val_dir? 
	# Looking at your previous errors, it seems you need val_dir.
	setattr(real_cfg, "val_dir", Path("dummy/eval/dir"))
	
	yield real_cfg
	
	# Cleanup
	real_cfg.output_dir = original_out
	if original_val:
		setattr(real_cfg, "val_dir", original_val)

@patch("src.evaluate.os.path.exists")
def test_evaluate_model_not_found(mock_exists, capsys, dummy_cfg):
	mock_exists.return_value = False
	# We don't need to patch cfg here because dummy_cfg fixture already modified the global singleton
	evaluate.evaluate()
	captured = capsys.readouterr()
	assert "Model path" in captured.out

@patch("src.evaluate.os.path.exists")
@patch("src.evaluate.glob.glob")
@patch("src.evaluate.MistralForCausalLM.from_pretrained")
def test_evaluate_no_test_files(mock_from_pretrained, mock_glob, mock_exists, capsys, dummy_cfg):
	mock_exists.return_value = True
	mock_glob.return_value = [] 
	mock_from_pretrained.return_value = MagicMock()

	evaluate.evaluate()
	captured = capsys.readouterr()
	assert "No test files found" in captured.out

@patch("src.evaluate.os.path.exists")
@patch("src.evaluate.glob.glob")
@patch("src.evaluate.MistralForCausalLM.from_pretrained")
def test_evaluate_full_execution(mock_from_pretrained, mock_glob, mock_exists, capsys, dummy_cfg):
	mock_exists.return_value = True
	mock_glob.return_value = ["dummy_test_1.json"]

	mock_model = MagicMock()
	mock_from_pretrained.return_value = mock_model

	# Match your specific config logic
	sep_token = dummy_cfg.unique_homophones + 1  
	char_offset = sep_token + 1                  

	generated_tokens = [char_offset + i for i in range(3)] 
	input_cipher_ids = [100, 101, 102]
	mock_generated_sequence = input_cipher_ids + [sep_token] + generated_tokens
	mock_model.generate.return_value = torch.tensor([mock_generated_sequence])

	mock_file_content = json.dumps({
		"ciphertext": "100 101 102",
		"plaintext": "abc"
	})

	with patch("builtins.open", mock_open(read_data=mock_file_content)):
		evaluate.evaluate()

	captured = capsys.readouterr()
	assert "Pred Plaintext: abc" in captured.out
	assert "Symbol Error Rate (SER): 0.0000" in captured.out