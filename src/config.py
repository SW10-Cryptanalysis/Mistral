import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from easy_logging import EasyFormatter

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

TEXT_LEN = 9961
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 78
UNIQUE_HOMOPHONE_COUNT = 2494

DATA_DIR = Path(__file__).parent.parent.parent / "Ciphers"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
HOMOPHONE_FILE = "metadata.json"

TOKENIZED_TRAINING_DIR = DATA_DIR / "tokenized_normal" / "Training"
TOKENIZED_VALIDATION_DIR = DATA_DIR / "tokenized_normal" / "Validation"
TOKENIZED_TEST_DIR = DATA_DIR / "tokenized_normal" / "Test"

TOKENIZED_SPACED_TRAINING_DIR = DATA_DIR / "tokenized_spaced" / "Training"
TOKENIZED_SPACED_VALIDATION_DIR = DATA_DIR / "tokenized_spaced" / "Validation"
TOKENIZED_SPACED_TEST_DIR = DATA_DIR / "tokenized_spaced" / "Test"


@dataclass
class Config:
    """Centralized model, data, and training configuration values."""

    # ARCHITECTURE
    unique_homophones: int = UNIQUE_HOMOPHONE_COUNT
    unique_letters: int = 26
    vocab_size: int = (
        2560  # Padded to nearest multiple of 64 for L4 Ada Lovelace Tensor Cores
    )
    max_context: int = TOTAL_SEQ + BUFFER  # 20000 exactly

    # Mistral Specific Hyperparameters
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    sliding_window: int = 20000
    rope_theta: float = 1_000_000.0

    # TRAINING (AAU AI-Lab Optimization: 4-8x L4 GPUs)
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 3e-4
    epochs: int = 3
    grad_checkpoint: bool = True
    torch_compile: bool = False
    bf16: bool = True
    use_spaces: bool = True

    # STEPS
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 1000
    save_total_limit: int = 2

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    tokenized_training_dir: Path = TOKENIZED_TRAINING_DIR
    tokenized_val_dir: Path = TOKENIZED_VALIDATION_DIR
    tokenized_test_dir: Path = TOKENIZED_TEST_DIR

    tokenized_spaced_train_dir: Path = TOKENIZED_SPACED_TRAINING_DIR
    tokenized_spaced_val_dir: Path = TOKENIZED_SPACED_VALIDATION_DIR
    tokenized_spaced_test_dir: Path = TOKENIZED_SPACED_TEST_DIR

    # Token IDs
    pad_token_id: int = 0

    @property
    def sep_token_id(self) -> int:
        """Seperator token."""
        return self.unique_homophones + 1

    @property
    def space_token_id(self) -> int:
        """Space token."""
        return self.sep_token_id + 1

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token."""
        return self.space_token_id + 1

    @property
    def eos_token_id(self) -> int:
        """End of sequence token."""
        return self.bos_token_id + 1

    @property
    def char_offset(self) -> int:
        """Offset for character token IDs."""
        return self.eos_token_id + 1

    def load_homophones(self) -> None:
        """Load homophone mappings from the metadata file."""
        homophone_path = os.path.join(DATA_DIR, HOMOPHONE_FILE)
        if os.path.exists(homophone_path):
            try:
                with open(homophone_path) as f:
                    meta = json.load(f)
                    self.unique_homophones = int(meta["max_symbol_id"])
            except OSError as e:
                logger.warning("Could not read file: %s", HOMOPHONE_FILE)
                logger.warning("Using default value: %d", self.unique_homophones)
                logger.warning("Error details: %s", str(e))
            except (ValueError, KeyError) as e:
                logger.warning("Invalid or missing data in: %s", HOMOPHONE_FILE)
                logger.warning("Using default value: %d", self.unique_homophones)
                logger.warning("Error details: %s", str(e))

        raw = self.unique_homophones + self.unique_letters + BUFFER
        self.vocab_size = (
            (raw + 63) // 64 * 64
        )  # Padded to nearest multiple of 64 for L4 Ada Lovelace Tensor Cores


cfg = Config()
