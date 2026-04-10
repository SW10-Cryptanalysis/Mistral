import os
import json
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from easy_logging import EasyFormatter

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--without-spaces",
    action="store_true",
    default=False,
    help="If enabled the model trains without space tokens in the training dataset",
)
cli_args, _ = parser.parse_known_args()

MAX_PLAIN_SPACES = 13077
MAX_PLAIN_NORMAL = 10063

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
    buffer: int = 10
    unique_letters: int = 26
    unique_homophones: int = 0
    vocab_size: int = 0
    max_context: int = 0

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
    use_spaces: bool = not cli_args.without_spaces

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

    def __post_init__(self) -> None:
        """Calculate dynamic variables after the dataclass is initialized."""
        if self.use_spaces:
            self.max_context = (MAX_PLAIN_SPACES * 2) + self.buffer
        else:
            self.max_context = (MAX_PLAIN_NORMAL * 2) + self.buffer

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
        if not os.path.exists(homophone_path):
            raise FileNotFoundError(
                f"Metadata file not found at: {homophone_path}. "
                "Cannot determine unique_homophones — aborting.",
                1,
            )
        try:
            with open(homophone_path) as f:
                meta = json.load(f)
                self.unique_homophones = int(meta["max_symbol_id"])
        except OSError as e:
            raise OSError(f"Could not read file: {homophone_path}") from e
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Invalid or missing 'max_symbol_id' in {homophone_path}",
            ) from e

        raw = self.unique_homophones + self.unique_letters + self.buffer
        self.vocab_size = (
            (raw + 63) // 64 * 64
        )  # Padded to nearest multiple of 64 for L4 Ada Lovelace Tensor Cores


cfg = Config()
cfg.load_homophones()
