import math
import pytest
import torch
from unittest.mock import patch, MagicMock
from transformers import MistralConfig, MistralForCausalLM

from src import model
from src.config import Config

@pytest.fixture
def tiny_cfg():
    """Provides a microscopic model config to keep tests fast and CPU-friendly."""
    cfg = Config()
    cfg.vocab_size = 128
    cfg.hidden_size = 16
    cfg.intermediate_size = 32
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 1
    return cfg

def test_apply_custom_initialization(tiny_cfg):
    hf_config = MistralConfig(
        vocab_size=tiny_cfg.vocab_size,
        hidden_size=tiny_cfg.hidden_size,
        intermediate_size=tiny_cfg.intermediate_size,
        num_hidden_layers=tiny_cfg.num_hidden_layers,
        num_attention_heads=tiny_cfg.num_attention_heads,
        num_key_value_heads=tiny_cfg.num_key_value_heads,
    )
    dummy_model = MistralForCausalLM(hf_config)

    model.apply_custom_initialization(dummy_model, hf_config)

    params = dict(dummy_model.named_parameters())
    embed_std = params["model.embed_tokens.weight"].std().item()
    assert 0.015 < embed_std < 0.025, f"Embed std {embed_std} is out of bounds."

    expected_scaled_std = 0.02 / math.sqrt(2 * tiny_cfg.num_hidden_layers)
    down_proj_std = params["model.layers.0.mlp.down_proj.weight"].std().item()

    assert (expected_scaled_std * 0.5) < down_proj_std < (expected_scaled_std * 1.5)

@patch("src.model.torch.cuda.is_available")
def test_get_model_sdpa_fallback(mock_cuda, tiny_cfg):
    mock_cuda.return_value = False

    with (
        patch("src.model.cfg", tiny_cfg),
        patch("os.environ.get", return_value="0")
    ):
        test_model = model.get_model()

    attn_impl = getattr(test_model.config, "_attn_implementation", getattr(test_model.config, "attn_implementation", None))
    assert attn_impl == "sdpa"
    assert test_model.dtype == torch.bfloat16

@patch("src.model.torch.cuda.is_available")
@patch("src.model.MistralForCausalLM")
def test_get_model_flash_attention(mock_mistral_cls, mock_cuda, tiny_cfg):
    mock_cuda.return_value = True
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.num_parameters.return_value = 100_000_000
    mock_mistral_cls.return_value = mock_model_instance

    with (
        patch("src.model.cfg", tiny_cfg),
        patch("os.environ.get", return_value="0"),
    ):
        model.get_model()

    called_config = mock_mistral_cls.call_args[0][0]
    attn_impl = getattr(called_config, "_attn_implementation", getattr(called_config, "attn_implementation", None))
    assert attn_impl == "flash_attention_2"
