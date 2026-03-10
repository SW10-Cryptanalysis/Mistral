import os
import math
import torch
import torch.nn as nn
from transformers import MistralConfig, MistralForCausalLM
from src.config import cfg

def apply_custom_initialization(model: nn.Module, config: MistralConfig) -> None:
    """Applies variance-preserving weight initialization based on Han (2025).

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

def get_model() -> MistralForCausalLM:
    """Instantiates a Mistral architecture configured for sequence-to-sequence cipher decryption.

    Forces Flash Attention 2 for large context windows and Bfloat16 for Ada Lovelace Tensor Cores.
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
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
        torch_dtype=torch.bfloat16, # Added: Resolves FA2 dtype warning and natively targets L4 architecture
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
    )

    # Initialize directly in bfloat16 to avoid VRAM spikes during weight casting under FSDP
    model = MistralForCausalLM(config).to(torch.bfloat16)

    # Apply the academic variance-preserving initialization
    apply_custom_initialization(model, config)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        pass

    return model
