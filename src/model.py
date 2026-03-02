import torch
from transformers import MistralConfig, MistralForCausalLM
from config import cfg

def get_mistral_model():
    """
    Initializes a MistralForCausalLM with a configuration optimized for 
    long-sequence homophonic deciphering.
    """
    configuration = MistralConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        sliding_window=cfg.sliding_window,
        tie_word_embeddings=False,
        use_cache=True
    )
    
    model = MistralForCausalLM(configuration)
    
    # Enable Gradient Checkpointing for 16k context on L4 GPUs
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    return model