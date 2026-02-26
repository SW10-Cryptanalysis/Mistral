import json
import glob
import os
import torch
from model import MistralForCausalLM
from config import cfg

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = cfg.output_dir / "final_model"
    
    model = MistralForCausalLM.from_pretrained(model_path).to(device).half() # type: ignore
    model.eval()

    # Mapping logic (Inverse of train)
    max_homophone = 8192
    sep_token = max_homophone + 1
    char_offset = sep_token + 1
    chars = "abcdefghijklmnopqrstuvwxyz "
    id_to_char = {i + char_offset: char for i, char in enumerate(chars)}

    test_files = glob.glob(os.path.join(cfg.eval_dir, "*.json"))[:5]

    for file_path in test_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        cipher_ids = [int(x) for x in data["ciphertext"].split()][:10000]
        input_ids = torch.tensor([cipher_ids + [sep_token]]).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=100, # Adjust based on expected plaintext length
                pad_token_id=0
            )
        
        # Remove the cipher prompt
        generated_part = output[0][len(cipher_ids)+1:]
        decoded = "".join([id_to_char.get(int(i), "?") for i in generated_part])
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"Decoded: {decoded}\n")

if __name__ == "__main__":
    evaluate()