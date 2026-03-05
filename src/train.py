import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from config import cfg
from model import get_model

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class ArrowDatasetWrapper(Dataset):
    def __init__(self, directory_path):
        self.hf_dataset = load_from_disk(str(directory_path))
        
        if len(self.hf_dataset) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Warning: Dataset at {directory_path} is empty.")
            
        self.max_homophone = cfg.unique_homophones 
        self.sep_token = self.max_homophone + 1
        char_offset = self.sep_token + 1
        chars = "abcdefghijklmnopqrstuvwxyz "
        
        self.char_to_id = {char: i + char_offset for i, char in enumerate(chars)}

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        data = self.hf_dataset[idx]
        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        plain_text = data.get("plaintext", "")
        plain_ids = [self.char_to_id.get(char, 0) for char in plain_text] 
        
        # Build full sequence
        input_ids = cipher_ids + [self.sep_token] + plain_ids
        
        if len(input_ids) > cfg.max_context:
            input_ids = input_ids[:cfg.max_context]
            
        # CIPHERTEXT MASKING STRATEGY:
        # We set labels for the ciphertext to -100 so the model does NOT calculate 
        # cross-entropy loss on predicting the ciphertext from itself. 
        # It only calculates loss on predicting plaintext from the ciphertext context.
        labels = ([-100] * len(cipher_ids)) + [-100] + plain_ids
        
        if len(labels) > cfg.max_context:
            labels = labels[:cfg.max_context]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def custom_collate(batch):
    """
    Pads the batch dynamically to the longest sequence in the batch.
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad input_ids with 0 and labels with -100
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # Attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids_padded != 0).long()
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }

def train():
    model = get_model()
    
    train_ds = ArrowDatasetWrapper(cfg.data_dir)
    test_ds = ArrowDatasetWrapper(cfg.test_dir)

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        
        # Checkpointing & Memory Management
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