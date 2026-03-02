import os
import torch
from datasets import load_from_disk
from model import get_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import cfg

# Performance: Expandable segments help prevent fragmentation with long sequences
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def tokenize_function(example):
    # Mapping logic (Simplified to focus on efficiency)
    max_homophone = 8192
    sep_token = max_homophone + 1
    char_offset = sep_token + 1
    chars = "abcdefghijklmnopqrstuvwxyz " 
    char_to_id = {char: i for i, char in enumerate(chars)}

    # Truncate to ensure they fit in the 20k window (10k each)
    cipher_ids = [int(x) for x in example["ciphertext"].split()][:10000]
    plain_ids = [char_to_id.get(c, 0) + char_offset for c in example["plaintext"]][:10000]

    full_seq = cipher_ids + [sep_token] + plain_ids
    
    return {
        "input_ids": full_seq,
        "labels": full_seq.copy()
    }

def train():
    # This will now exit if Flash Attention is missing
    model = get_model()
    
    train_ds = load_from_disk(str(cfg.data_dir))
    test_ds = load_from_disk(str(cfg.test_dir))

    train_ds = train_ds.map(tokenize_function, num_proc=16) # Higher proc for speed
    test_ds = test_ds.map(tokenize_function, num_proc=16)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        gradient_checkpointing=cfg.grad_checkpoint,
        bf16=True, 
        tf32=True, # L4 GPUs support TF32 for faster float32 math
        logging_steps=5,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        torch_compile=True, 
        # Mistral uses Flash Attention 2 automatically in Transformers 4.34+ 
        # if the package is installed.
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=None, mlm=False, pad_to_multiple_of=8)
    )

    print("--- Starting Training ---")
    trainer.train()
    trainer.save_model(str(cfg.output_dir / "final_cipher_model"))

if __name__ == "__main__":
    train()