import os
import sys
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
    "--with-spaces",
    action="store_true",
    default=True,
    help="If enabled the model trains with space tokens in the training dataset",
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default=None,
    help="Path to the dataset (e.g., 'tokenized_spaced', 'tokenized_normal_truncated_4000').",
)
cli_args, _ = parser.parse_known_args()

_errors = []
if cli_args.dataset_path is None:
    _errors.append(
        "  --dataset-path is required. Specify the dataset subdirectory (e.g., 'tokenized_normal', 'tokenized_spaced_truncated_4000').",
    )

if _errors:
    logger.error("Missing required arguments:\n" + "\n".join(_errors))
    logger.error(
        "Example: sbatch train.slurm --dataset-path 'tokenized_normal_truncated_4000' --with-spaces",
    )
    sys.exit(1)

MAX_PLAIN_SPACES = 13077
MAX_PLAIN_NORMAL = 10063

DATA_DIR = Path("/work/Ciphers")
OUTPUT_DIR = Path("/work/Mistral/outputs")
HOMOPHONE_FILE = "metadata.json"


@dataclass
class Config:
    """Centralized model, data, and training configuration values."""

    # ARCHITECTURE
    buffer: int = 10
    unique_letters: int = 26
    unique_homophones: int = 0
    vocab_size: int = 0

    # Mistral Specific Hyperparameters
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    sliding_window: int = 20000
    rope_theta: float = 1_000_000.0

    # TRAINING
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 3e-4
    epochs: int = 3
    grad_checkpoint: bool = True
    torch_compile: bool = True
    bf16: bool = True
    use_spaces: bool = not cli_args.with_spaces

    # STEPS
    logging_steps: int = 10
    save_steps: int = 30000
    eval_steps: int = 20000
    save_total_limit: int = 2

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    data_dir: Path = DATA_DIR
    dataset_path: str = cli_args.dataset_path

    # Token IDs
    pad_token_id: int = 0

    @property
    def max_context(self) -> int:
        """Calculate dynamic variables after the dataclass is initialized."""
        if self.use_spaces:
            return (MAX_PLAIN_SPACES * 2) + self.buffer
        return (MAX_PLAIN_NORMAL * 2) + self.buffer

    @property
    def final_output_dir(self) -> Path:
        """Return the output directory path for saving fine-tuned models, differentiated by space token usage."""
        return self.output_dir / f"{self.dataset_path}_model"

    @property
    def tokenized_train_dir(self) -> Path:
        """Path for tokenized training data."""
        return self.data_dir / self.dataset_path / "Training"

    @property
    def tokenized_val_dir(self) -> Path:
        """Path for tokenized validation data."""
        return self.data_dir / self.dataset_path / "Validation"

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
        logger.info(
            f"Config initialized: unique_homophones={self.unique_homophones}, sep_token_id={self.sep_token_id}, space_token_id={self.space_token_id}, bos_token_id={self.bos_token_id}, eos_token_id={self.eos_token_id}, char_offset={self.char_offset}, vocab_size={self.vocab_size}",
        )
        logger.info(
            f"Max len set to {self.max_context} based on use_spaces={self.use_spaces}",
        )
        logger.info(f"Training directory: {self.tokenized_train_dir}")
        logger.info(f"Validation directory: {self.tokenized_val_dir}")


cfg = Config()
cfg.load_homophones()
