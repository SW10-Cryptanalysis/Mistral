from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # PATHS
    DATA_DIR = Path("./data/Ciphers")
    TRAINING_DIR = DATA_DIR / "Training_Arrow"
    TEST_DIR = DATA_DIR / "Test_Arrow"
    OUTPUT_DIR = Path("./outputs/mistral_cipher")

    # ARCHITECTURE
    # max_symbol_id is 2067. Vocab must cover cipher symbols + SEP + plaintext chars.
    vocab_size: int = 2560  
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 2  # Grouped Query Attention (GQA)
    max_position_embeddings: int = 16384
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 4096 
    
    # TRAINING
    batch_size: int = 2  # Small batch for 16k context
    grad_accum: int = 16
    epochs: int = 5
    log_steps: int = 10
    save_steps: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True

cfg = Config()