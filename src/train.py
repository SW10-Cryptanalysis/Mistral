import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from config import cfg
from model import get_model

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class PretokenizedCipherDataset(Dataset):
    """
    Directly loads Arrow files mapped by the preprocessing pipeline.
    All data is pre-tokenized and labels are pre-set for Equal Loss Weighting.
    """
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
        
        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Warning: Dataset at {directory_path} is empty.")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Data is already tokenized. We enforce the Mandatory Training Objective 
        # by ensuring labels equal input_ids for full sequence modeling.
        input_ids = item["input_ids"][:cfg.max_context]
        labels = item["labels"][:cfg.max_context]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def custom_collate(batch):
    """
    Pads the pre-tokenized batch dynamically to the longest sequence in the batch.
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    # Since we are predicting the whole sequence, padding tokens in the labels are set to -100
    # to avoid penalizing the model for padding.
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    attention_mask = (input_ids_padded != 0).long()
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }

def train():
    model = get_model()
    
    train_ds = PretokenizedCipherDataset(cfg.tokenized_training_dir)
    test_ds = PretokenizedCipherDataset(cfg.tokenized_test_dir)

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        
        # Checkpointing & Memory Management (L4 Optimization)
        bf16=cfg.bf16,
        
        # Logging
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        
        # Performance optimizations
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=8,
        
        # DDP/FSDP configurations for 4x L4 setup
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
            "activation_checkpointing": cfg.grad_checkpoint
        },
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=custom_collate
    )

    if trainer.is_world_process_zero():
        print(f"Starting distributed training on {torch.cuda.device_count()} GPUs...")
        
    trainer.train()
    
    if trainer.is_world_process_zero():
        print("Saving final model...")
        trainer.save_model(os.path.join(str(cfg.output_dir), "final_model"))
        print("Model saved successfully.")

if __name__ == "__main__":
    train()