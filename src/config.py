from dataclasses import dataclass
from pathlib import Path

# Context sizing for Ciphers based on provided metadata
TEXT_LEN = 10000 # Max length is 9950
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 10 

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Expecting pre-tokenized Arrow directories
TOKENIZED_TRAINING_DIR = DATA_DIR / "tokenized_normal" / "Training"
TOKENIZED_TEST_DIR = DATA_DIR / "tokenized_normal" / "Test"
TOKENIZED_VALIDATION_DIR = DATA_DIR / "tokenized_normal" / "Validation"

@dataclass
class Config:
    # ARCHITECTURE
    unique_homophones: int = 2067 
    unique_letters: int = 26
    vocab_size: int = 2176  # 2067 + 26 + buffer, padded to multiple of 64 for Tensor Cores
    max_context: int = TOTAL_SEQ + BUFFER
    
    # Mistral Specific Hyperparameters
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA
    sliding_window: int = 4096
    rope_theta: float = 1000000.0 # Increased for longer contexts
    
    # TRAINING (Optimized for 4-8x L4 GPUs)
    batch_size: int = 1 
    grad_accum: int = 16
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
    tokenized_training_dir: Path = TOKENIZED_TRAINING_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR
    tokenized_val_dir: Path = TOKENIZED_VALIDATION_DIR

cfg = Config()