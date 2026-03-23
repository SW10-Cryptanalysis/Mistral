from pathlib import Path
from src.config import Config, cfg


def test_config_defaults():
    assert cfg.vocab_size == 2560
    assert cfg.max_context == 20100
    assert cfg.pad_token_id == 0
    assert cfg.sep_token_id == 2504
    assert cfg.space_token_id == 2505
    assert cfg.bos_token_id == 2506
    assert cfg.eos_token_id == 2507

    assert isinstance(cfg.unique_homophones, int)
    assert isinstance(cfg.unique_letters, int)
    assert isinstance(cfg.vocab_size, int)
    assert isinstance(cfg.max_context, int)
    assert isinstance(cfg.pad_token_id, int)
    assert isinstance(cfg.sep_token_id, int)
    assert isinstance(cfg.space_token_id, int)
    assert isinstance(cfg.bos_token_id, int)
    assert isinstance(cfg.eos_token_id, int)
    assert isinstance(cfg.hidden_size, int)
    assert isinstance(cfg.intermediate_size, int)
    assert isinstance(cfg.num_hidden_layers, int)
    assert isinstance(cfg.num_attention_heads, int)
    assert isinstance(cfg.num_key_value_heads, int)
    assert isinstance(cfg.sliding_window, int)
    assert isinstance(cfg.rope_theta, float)
    assert isinstance(cfg.batch_size, int)
    assert isinstance(cfg.grad_accum, int)
    assert isinstance(cfg.learning_rate, float)
    assert isinstance(cfg.epochs, int)
    assert isinstance(cfg.grad_checkpoint, bool)
    assert isinstance(cfg.torch_compile, bool)
    assert isinstance(cfg.bf16, bool)
    assert isinstance(cfg.use_spaces, bool)
    assert isinstance(cfg.logging_steps, int)
    assert isinstance(cfg.save_steps, int)
    assert isinstance(cfg.eval_steps, int)
    assert isinstance(cfg.save_total_limit, int)
    assert isinstance(cfg.output_dir, Path)
    assert isinstance(cfg.tokenized_training_dir, Path)
    assert isinstance(cfg.tokenized_test_dir, Path)
    assert isinstance(cfg.tokenized_val_dir, Path)
    assert isinstance(cfg.tokenized_spaced_train_dir, Path)
    assert isinstance(cfg.tokenized_spaced_val_dir, Path)
    assert isinstance(cfg.tokenized_spaced_test_dir, Path)


def test_config_paths():
    assert isinstance(cfg.output_dir, Path)
    assert isinstance(cfg.tokenized_training_dir, Path)

    assert cfg.output_dir.name == "outputs"
    assert "Ciphers" in str(cfg.tokenized_training_dir)


def test_custom_config_instantiation():
    custom_cfg = Config(vocab_size=3000, batch_size=8)
    assert custom_cfg.vocab_size == 3000
    assert custom_cfg.batch_size == 8
    assert custom_cfg.learning_rate == 3e-4
