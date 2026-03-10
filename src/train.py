import os
import torch
import numpy as np
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from config import cfg
from model import get_model

torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class PretokenizedCipherDataset(Dataset):
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Warning: Dataset at {directory_path} is empty.")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Mandatory Training Objective (Equal Loss Weighting)
        input_ids = item["input_ids"][:cfg.max_context]
        labels = item["labels"][:cfg.max_context]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def safe_pad_collate(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=cfg.pad_token_id
    )
    
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    
    attention_mask = (input_ids_padded != cfg.pad_token_id).long()
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }
    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

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

    ser = total_errors / total_symbols if total_symbols > 0 else 0
    return {"ser": ser}


def train():
    model = get_model()
    
    # Apply gradient checkpointing for VRAM management
    if cfg.grad_checkpoint:
        model.gradient_checkpointing_enable()
        
    train_ds = PretokenizedCipherDataset(cfg.tokenized_training_dir)
    test_ds = PretokenizedCipherDataset(cfg.tokenized_test_dir)
    
    fsdp_config = {
        "transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
        "backward_prefetch": "backward_pre",
        "forward_prefetch": "True",
        "use_orig_params": "True",
        "sync_module_states": "True",
    }

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=8,
        fsdp="full_shard auto_wrap",
        fsdp_config=fsdp_config
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=safe_pad_collate,
        compute_metrics=compute_metrics
    )

    if trainer.is_world_process_zero():
        print(f"Starting training on {torch.cuda.device_count()} GPUs...")
        
    last_checkpoint = get_last_checkpoint(str(cfg.output_dir))
    if last_checkpoint is not None and trainer.is_world_process_zero():
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        
    trainer.train(resume_from_checkpoint=last_checkpoint)
        
    if trainer.is_world_process_zero():
        print("Saving final model...")
        trainer.save_model(os.path.join(str(cfg.output_dir), "final_model"))

if __name__ == "__main__":
    train()