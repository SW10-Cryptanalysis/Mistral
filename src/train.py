import os
import time
import torch
import logging
import numpy as np
import transformers
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from src.config import cfg
from src.model import get_model
from easy_logging import EasyFormatter

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"  # type:ignore
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class HardwareOptimizationCallback(TrainerCallback):
    """Custom callback for AAU AI-Lab 4-8x L4 GPU performance telemetry."""

    def __init__(self) -> None:
        """Initialize the callback."""
        self.epoch_start_time = 0

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: transformers.TrainerControl,
    ) -> None:
        """Log the start time of the epoch."""
        self.epoch_start_time = time.time()
        self.epoch_start_step = state.global_step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: transformers.TrainerControl,
    ) -> None:
        """Log performance metrics at the end of the epoch."""
        epoch_time = time.time() - self.epoch_start_time
        steps_this_epoch = state.global_step - self.epoch_start_step
        num_devices = max(torch.cuda.device_count(), 1)

        # Tokens per second (Approximate based on max context)
        total_tokens_per_epoch = (
            steps_this_epoch
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * cfg.max_context
            * num_devices
        )
        tokens_per_sec = total_tokens_per_epoch / epoch_time if epoch_time > 0 else 0

        # VRAM Tracking (Memory Management)
        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        logger.info(f"--- Epoch {state.epoch} Performance Metrics ---")
        logger.info(f"Wall-clock time: {epoch_time:.2f} seconds")
        logger.info(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        logger.info(f"Peak VRAM: {peak_vram_gb:.2f} GB")
        logger.info(
            (
                f"Wall-clock time per 10k-token sample: {(epoch_time / (total_tokens_per_epoch / 10000)):.4f} seconds"
                if total_tokens_per_epoch > 0
                else "N/A"
            ),
        )
        logger.info("-" * 40)


def log_environment_details(seed: int) -> None:
    """Logs SOTA Implementation library versions and hardware details."""
    logger.info("=== AAU AI-Lab Execution Environment ===")
    logger.info(f"Seed: {seed}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Transformers Version: {transformers.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        logger.info(f"GPU Count: {device_count}")
        for i in range(device_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        import flash_attn  # type: ignore

        logger.info(f"FlashAttention-2 Version: {flash_attn.__version__}")
    except ImportError:
        logger.warning("FlashAttention-2 not found. Sequence bottlenecks may occur.")
    logger.info("========================================")


class PretokenizedCipherDataset(Dataset):
    """Dataset wrapper for loading pre-tokenized cipher samples from disk."""

    def __init__(self, directory_path: str | Path) -> None:
        """Load a serialized Hugging Face dataset from `directory_path`."""
        self.hf_dataset = load_from_disk(str(directory_path))
        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.warning("Dataset at %s is empty!", directory_path)

    def __len__(self) -> int:
        """Return the number of examples available in the dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Fetch one sample and convert token arrays into `torch.long` tensors."""
        item = self.hf_dataset[idx]

        if (
            len(item["input_ids"]) > cfg.max_context
            or len(item["labels"]) > cfg.max_context
        ):
            logger.info(
                f"Sample {idx} truncated: input_ids {len(item['input_ids'])} -> {cfg.max_context}, labels {len(item['labels'])} -> {cfg.max_context}",
            )

        # Mandatory Training Objective (Equal Loss Weighting allows mapping logic to thrive)
        input_ids = item["input_ids"][: cfg.max_context]
        labels = item["labels"][: cfg.max_context]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def safe_pad_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length token tensors and build the attention mask."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=cfg.pad_token_id,
    )

    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    )

    attention_mask = (input_ids_padded != cfg.pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
    }


def preprocess_logits_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Applies argmax dynamically to each batch's logits during evaluation.

    This vital Memory Management strategy prevents OOM errors by avoiding
    the accumulation of [Batch, Seq, Vocab] float tensors in GPU memory.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(
    eval_preds: EvalPrediction | tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Compute symbol error rate (SER) while ignoring padded labels."""
    if isinstance(eval_preds, tuple):
        predictions, labels = eval_preds
    else:
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    # We drop the last prediction (it predicts what comes after the sequence ends)
    # We drop the first label (the model doesn't predict the BOS token)
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]

    total_errors = 0
    total_symbols = 0

    for i in range(labels.shape[0]):
        # Mask out padding (-100)
        mask = labels[i] != -100
        val_labels = labels[i][mask]
        val_preds = predictions[i][mask]

        # Calculate mismatches
        total_errors += np.sum(val_labels != val_preds)
        total_symbols += len(val_labels)

    ser = total_errors / total_symbols if total_symbols > 0 else 0.0
    return {"ser": ser}


def train() -> None:
    """Main training loop using Hugging Face's Trainer API."""
    # Seed Tracking
    run_seed = 42
    set_seed(run_seed)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        log_environment_details(run_seed)

    model = get_model()

    if cfg.grad_checkpoint:
        model.gradient_checkpointing_enable()

    if cfg.use_spaces:
        logger.info("Using space tokens in training.")
        train_ds = PretokenizedCipherDataset(cfg.tokenized_spaced_train_dir)
        val_ds = PretokenizedCipherDataset(cfg.tokenized_spaced_val_dir)
    else:
        logger.info("Not using space tokens in training.")
        train_ds = PretokenizedCipherDataset(cfg.tokenized_training_dir)
        val_ds = PretokenizedCipherDataset(cfg.tokenized_val_dir)

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        eval_accumulation_steps=4,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        seed=run_seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=safe_pad_collate,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[HardwareOptimizationCallback()],
    )

    last_checkpoint = get_last_checkpoint(str(cfg.output_dir))

    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_model_name = "final_model"
    if cfg.use_spaces:
        final_model_name += "_with_spaces"
    else:
        final_model_name += "_no_spaces"

    if trainer.is_world_process_zero():
        trainer.save_model(os.path.join(str(cfg.output_dir), final_model_name))


if __name__ == "__main__":
    train()
