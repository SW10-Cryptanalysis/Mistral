import torch
import json
import glob
from transformers import MistralForCausalLM
from config import cfg

@torch.no_grad()
def evaluate_performance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MistralForCausalLM.from_pretrained(cfg.OUTPUT_DIR / "final_mistral").to(device)
    model.eval()

    # Constants
    sep_token = 2068
    char_offset = 2069
    id_to_char = {i + char_offset: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}

    test_files = glob.glob(str(cfg.DATA_DIR / "Test/*.json"))[:5]

    for file in test_files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        input_ids = torch.tensor([cipher_ids + [sep_token]]).to(device)

        # Generate using KV-Caching for efficiency
        output = model.generate(
            input_ids,
            max_new_tokens=200, # Testing window
            do_sample=False,
            use_cache=True
        )

        # Decode only the generated part
        pred_ids = output[0][len(cipher_ids)+1:].cpu().tolist()
        prediction = "".join([id_to_char.get(i, '?') for i in pred_ids])
        
        print(f"File: {file}")
        print(f"Target: {data['plaintext'][:50]}...")
        print(f"Pred:   {prediction[:50]}...")
        print("-" * 30)

if __name__ == "__main__":
    evaluate_performance()