from dataclasses import dataclass
from pathlib import Path

# Context sizing for Ciphers based on provided metadata
TEXT_LEN = 9961 # Updated to match the max_length from metadata
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 78 # Buffer to reach clean tensor shapes

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

TOKENIZED_TRAINING_DIR = DATA_DIR / "tokenized_normal" / "Training"
TOKENIZED_TEST_DIR = DATA_DIR / "tokenized_normal" / "Test"
TOKENIZED_VALIDATION_DIR = DATA_DIR / "tokenized_normal" / "Validation"

@dataclass
class Config:
    # ARCHITECTURE
    unique_homophones: int = 2494 # Updated from metadata.json max_symbol_id
    unique_letters: int = 26
    # 2494 + 26 = 2520. Padded to 2560 for L4 Ada Lovelace Tensor Cores
    vocab_size: int = 2560  
    max_context: int = TOTAL_SEQ + BUFFER # 20000 exactly
    
    # Token IDs (Strictly mapped to prevent overlap)
    pad_token_id: int = 0
    sep_token_id: int = 2495
    space_token_id: int = 2496
    bos_token_id: int = 2497
    eos_token_id: int = 2498
    
    # Mistral Specific Hyperparameters
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA
    sliding_window: int = 20000 # Disabled effectively to allow global cross-sequence attention
    rope_theta: float = 1_000_000.0 # Excellent for 20k context
    
    # TRAINING (AAU AI-Lab Optimization: 4-8x L4 GPUs)
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