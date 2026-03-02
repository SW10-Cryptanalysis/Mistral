from dataclasses import dataclass
from pathlib import Path

# Context sizing for Ciphers
TEXT_LEN = 8_192
TOTAL_SEQ = 16_384  # Sufficient for max train length of 9950 tokens

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
TRAINING_DIR = DATA_DIR / "Training_Arrow"
TEST_DIR = DATA_DIR / "Test_Arrow"
EVAL_DIR = DATA_DIR / "Test"
VALIDATION_DIR = DATA_DIR / "Validation"

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

@dataclass
class Config:
    # ARCHITECTURE (Scaled for from-scratch training on 4x L4 GPUs)
    unique_homophones: int = 2500
    max_context: int = TOTAL_SEQ
    vocab_size: int = 2560  # Padded to multiple of 64 for Tensor Core efficiency
    
    # Mistral Specific Hyperparameters
    hidden_size: int = 512
    intermediate_size: int = 2048 # Usually 4x hidden_size
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # Grouped Query Attention (GQA)
    sliding_window: int = 4096
    rope_theta: float = 10000.0
    
    # TRAINING
    batch_size: int = 2 # Per device. Will act as 8 with 4 GPUs
    grad_accum: int = 8
    learning_rate: float = 3e-4
    epochs: int = 3
    grad_checkpoint: bool = True
    bf16: bool = True

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    data_dir: Path = TRAINING_DIR
    test_dir: Path = TEST_DIR
    eval_dir: Path = EVAL_DIR

cfg = Config()