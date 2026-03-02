import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import cfg
from model import get_mistral_model

class CipherDataset(Dataset):
    def __init__(self, directory_path):
        self.dataset = load_from_disk(str(directory_path))
        self.sep_token = 2068 # max_symbol_id + 1
        self.char_offset = 2069
        self.chars = "abcdefghijklmnopqrstuvwxyz "
        self.char_to_id = {c: i + self.char_offset for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        cipher_ids = [int(x) for x in item["ciphertext"].split()]
        plain_ids = [self.char_to_id.get(c, self.char_to_id[' ']) for c in item["plaintext"]]
        
        # Format: [Cipher] [SEP] [Plaintext]
        input_ids = cipher_ids + [self.sep_token] + plain_ids
        
        # Truncate to max context
        input_ids = input_ids[:cfg.max_position_embeddings]
        
        # We only want to compute loss on the Plaintext part (the labels)
        # However, for causal LM, standard practice is to shift internally.
        # To ignore cipher loss, we could set labels for cipher tokens to -100.
        labels = [-100] * (len(cipher_ids) + 1) + plain_ids
        labels = labels[:cfg.max_position_embeddings]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def train():
    model = get_mistral_model()
    train_ds = CipherDataset(cfg.TRAINING_DIR)
    
    training_args = TrainingArguments(
        output_dir=str(cfg.OUTPUT_DIR),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        bf16=cfg.bf16,
        logging_steps=cfg.log_steps,
        num_train_epochs=cfg.epochs,
        save_steps=cfg.save_steps,
        # L4 GPUs support FlashAttention2 via the attn_implementation flag
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        # Standard DC handles padding if sequences varied, but we truncate/pack
        data_collator=DataCollatorForLanguageModeling(tokenizer=None, mlm=False)
    )

    trainer.train()
    trainer.save_model(str(cfg.OUTPUT_DIR / "final_mistral"))

if __name__ == "__main__":
    train()