import os
import math
import torch
import torch.nn as nn
from transformers import MistralConfig, MistralForCausalLM
from config import cfg

def apply_custom_initialization(model, config):
    """
    Applies variance-preserving weight initialization based on Han (2025).
    Enforces a stable standard deviation band (0.02) and depth-dependent residual scaling.
    """
    std = 0.02
    # Residual scaling factor: 1 / sqrt(2 * Layers)
    scaled_std = std / math.sqrt(2 * config.num_hidden_layers)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1. Base initialization for embeddings and Q/K/V/Gate/Up projections
        if any(nd in name for nd in ["embed_tokens.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight"]):
            nn.init.normal_(param, mean=0.0, std=std)
            
        # 2. Depth-dependent scaling for layers writing directly to the residual stream
        elif any(nd in name for nd in ["o_proj.weight", "down_proj.weight"]):
            nn.init.normal_(param, mean=0.0, std=scaled_std)
            
        # 3. LM Head (Output mapping)
        elif "lm_head.weight" in name:
            nn.init.normal_(param, mean=0.0, std=std)

def get_model():
    """
    Instantiates a Mistral architecture configured for sequence-to-sequence cipher decryption.
    Forces Flash Attention 2 for large context windows.
    """
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
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa"
    )

    model = MistralForCausalLM(config)
    
    # Apply the academic variance-preserving initialization
    apply_custom_initialization(model, config)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Mistral Architecture Initialized. Params: {model.num_parameters() / 1e6:.1f}M")
        print("Applied Theory-Grounded Variance Initialization (std=0.02, with depth scaling).")
        
    return model