from dataclasses import dataclass
from pathlib import Path

# Context sizing for Ciphers
TEXT_LEN = 10000
TOTAL_SEQ = TEXT_LEN * 2

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
TRAINING_DIR = DATA_DIR / "Training_Arrow"
TEST_DIR = DATA_DIR / "Test_Arrow"
VALIDATION_DIR = DATA_DIR / "Validation_Arrow"

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
TOKENIZED_TRAINING_DIR = OUTPUT_DIR / "Tokenized_Training"
TOKENIZED_TEST_DIR = OUTPUT_DIR / "Tokenized_Test"
TOKENIZED_VALIDATION_DIR = OUTPUT_DIR / "Tokenized_Validation"

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
    torch_compile: bool = True
    
    # STEPS
    logging_steps: int = 50
    save_steps: int = 250
    eval_steps: int = 1000
    save_total_limit: int = 2

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    data_dir: Path = TRAINING_DIR
    test_dir: Path = TEST_DIR
    val_dir: Path = VALIDATION_DIR
    tokenized_training_dir: Path = TOKENIZED_TRAINING_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR
    tokenized_val_dir: Path = TOKENIZED_VALIDATION_DIR

cfg = Config()