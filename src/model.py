import os
import torch
from transformers import MistralConfig, MistralForCausalLM
from config import cfg

def get_model():
    """
    Instantiates a fresh Mistral architecture configured for cipher decryption.
    Utilizes Flash Attention 2 natively if available.
    """
    # Configure the Mistral architecture
    config = MistralConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        hidden_act="silu",
        max_position_embeddings=cfg.max_context,
        sliding_window=cfg.sliding_window,
        rope_theta=cfg.rope_theta,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        # Native HuggingFace FA2 integration
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa"
    )

    model = MistralForCausalLM(config)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Mistral Architecture Initialized. Params: {model.num_parameters() / 1e6:.1f}M")
        
    return model