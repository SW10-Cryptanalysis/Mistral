from pathlib import Path
from src.config import Config, cfg

def test_config_defaults():
    # Verify core calculations
    assert cfg.vocab_size == 2560
    assert cfg.max_context == 20000

    # Verify token IDs
    assert cfg.pad_token_id == 0
    assert cfg.sep_token_id == 2495
    assert cfg.eos_token_id == 2498

def test_config_paths():
    # Verify paths are created as pathlib.Path objects
    assert isinstance(cfg.output_dir, Path)
    assert isinstance(cfg.tokenized_training_dir, Path)

    # Verify directory names resolve properly
    assert cfg.output_dir.name == "outputs"
    assert "Ciphers" in str(cfg.tokenized_training_dir)

def test_custom_config_instantiation():
    # Ensure dataclass allows overriding for experiments
    custom_cfg = Config(vocab_size=3000, batch_size=8)
    assert custom_cfg.vocab_size == 3000
    assert custom_cfg.batch_size == 8
    # Unchanged values should remain default
    assert custom_cfg.learning_rate == 3e-4
