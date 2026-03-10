import json
import glob
import os
import logging
import importlib
import torch
import Levenshtein
from transformers import MistralForCausalLM
from src.config import cfg

handler = logging.StreamHandler()
try:
    easy_logging_module = importlib.import_module("easy_logging")
    easy_formatter_cls = easy_logging_module.EasyFormatter
    handler.setFormatter(easy_formatter_cls())
except (ImportError, AttributeError):
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

logger = logging.getLogger("evaluate.py")
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Locate and Load Model
    model_path = os.path.join(cfg.output_dir, "final_model")
    if not os.path.exists(model_path):
        logger.warning("Model path not found: %s", model_path)
        return


    # We load via HF's from_pretrained natively now
    model = MistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.eval()

    # 2. Tokenization Setup
    sep_token = cfg.unique_homophones + 1
    char_offset = sep_token + 1
    chars = "abcdefghijklmnopqrstuvwxyz "
    id_to_char = {i + char_offset: char for i, char in enumerate(chars)}

    # 3. Load Test Files
    test_files = glob.glob(os.path.join(cfg.val_dir, "*.json"))[:10]
    if not test_files:
        logger.warning("No test files found in: %s", cfg.val_dir)
        return


    for file_path in test_files:
        with open(file_path) as f:
            data = json.load(f)

        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        true_plain = data["plaintext"]

        max_cipher_len = cfg.max_context - 200
        if len(cipher_ids) > max_cipher_len:
            cipher_ids = cipher_ids[:max_cipher_len]

        input_ids = cipher_ids + [sep_token]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = torch.ones_like(input_tensor).to(device)


        # 4. Use HF's optimized Generation API
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=100,
                pad_token_id=0,
                eos_token_id=0,
                do_sample=False, # Greedy decoding
                use_cache=True,   # KV Caching for speed
            )

        # Extract only the newly generated tokens
        generated_ids = outputs[0][input_tensor.shape[1]:].tolist()

        # 5. Decode
        pred_plain = "".join([id_to_char.get(idx, "?") for idx in generated_ids])

        # Calculate SER
        true_plain_subset = true_plain[:len(pred_plain)]
        ser = Levenshtein.distance(true_plain_subset, pred_plain) / max(len(true_plain_subset), 1)

        logger.info("Pred Plaintext: %s", pred_plain)
        logger.info("Symbol Error Rate (SER): %.4f", ser)



if __name__ == "__main__":
    evaluate()
