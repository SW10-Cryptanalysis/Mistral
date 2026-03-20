import json
import os
import logging
import torch
import Levenshtein
from transformers import MistralForCausalLM
from easy_logging import EasyFormatter
from src.config import cfg

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("evaluate.py")
logger.addHandler(handler)


def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Locate and Load Model
    model_path = os.path.join(cfg.output_dir, "final_model")
    if not os.path.exists(model_path):
        logger.warning("Model path not found: %s", model_path)
        return

    model = MistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.eval()

    # 2. Tokenization Setup
    sep_token = cfg.sep_token_id
    char_offset = cfg.char_offset
    chars = "abcdefghijklmnopqrstuvwxyz "
    id_to_char = {i + char_offset: char for i, char in enumerate(chars)}

    # 3. Load Test Files
    test_dir = (
        cfg.tokenized_spaced_test_dir if cfg.use_spaces else cfg.tokenized_test_dir
    )
    test_files = list(test_dir.glob("*.json"))[:10]

    if not test_files:
        logger.warning("No test files found in: %s", test_dir)
        return

    for file_path in test_files:
        with open(file_path) as f:
            data = json.load(f)

        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        true_plain = data["plaintext"]

        # Prevent out-of-memory sequence bottlenecks by enforcing max_context limit
        max_cipher_len = (cfg.max_context // 2) - 100
        if len(cipher_ids) > max_cipher_len:
            cipher_ids = cipher_ids[:max_cipher_len]
            true_plain = true_plain[:max_cipher_len]

        input_ids = cipher_ids + [sep_token]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = torch.ones_like(input_tensor).to(device)

        # 4. Use HF's optimized Generation API with Strict Length Bound
        generation_length_limit = len(cipher_ids) + 1

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=generation_length_limit,
                pad_token_id=cfg.pad_token_id,
                eos_token_id=cfg.eos_token_id,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = outputs[0][input_tensor.shape[1] :].tolist()

        # Safely remove the EOS token to prevent dictionary miss '?' characters
        if cfg.eos_token_id in generated_ids:
            eos_index = generated_ids.index(cfg.eos_token_id)
            generated_ids = generated_ids[:eos_index]

        # 5. Decode
        pred_plain = "".join([id_to_char.get(idx, "?") for idx in generated_ids])

        # Calculate SER
        ser = Levenshtein.distance(true_plain, pred_plain) / max(len(true_plain), 1)

        logger.info("Pred Plaintext: %s", pred_plain)
        logger.info("Symbol Error Rate (SER): %.4f", ser)


if __name__ == "__main__":
    evaluate()
