from dataclasses import dataclass
from pathlib import Path

# Data sizing
CIPHER_LEN = 10_000
PLAIN_LEN = 10_000
TOTAL_SEQ = CIPHER_LEN + PLAIN_LEN + 10 

UNIQUE_HOMOPHONE_COUNT = 8192
UNIQUE_LETTER_COUNT = 30

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

@dataclass
class Config:
    # OPTIMIZED ARCHITECTURE
    vocab_size: int = UNIQUE_HOMOPHONE_COUNT + UNIQUE_LETTER_COUNT + 10
    max_context: int = TOTAL_SEQ
    
    dims: int = 256
    layers: int = 8
    att_heads: int = 8        
    intermediate_size: int = 1024
    num_key_value_heads: int = 2 
    
    # TRAINING
    batch_size: int = 2       
    grad_accum: int = 4
    learning_rate: float = 4e-4
    epochs: int = 5
    grad_checkpoint: bool = True 
    
    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    data_dir: Path = DATA_DIR / "Training_Arrow"
    test_dir: Path = DATA_DIR / "Test_Arrow"
    eval_dir: Path = DATA_DIR / "Test"

cfg = Config()