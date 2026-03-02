## model.py
import torch
import sys
from transformers import MistralConfig, MistralForCausalLM
from config import cfg

def check_flash_attention():
    """Stops execution if Flash Attention 2 is not available"""
    try:
        from flash_attn import flash_attn_func
        # Test a dummy attention operation to ensure kernels are compatible with GPU
        q = torch.randn(1, 128, 8, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 128, 8, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 128, 8, 64, dtype=torch.bfloat16, device="cuda")
        _ = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        print("✅ Flash Attention 2 Hardware Test Passed.")
    except Exception as e:
        print("\n" + "!"*60)
        print(f"❌ FLASH ATTENTION TEST FAILED: {e}")
        print("Training cannot proceed on 20k tokens without Flash Attention.")
        print("!"*60 + "\n")
        sys.exit(1) # CRITICAL: Actually stop the process

def get_model():
    """Initializes optimized Mistral model with mandatory Flash Attention 2"""
    
    # 1. Mandatory check
    check_flash_attention()

    configuration = MistralConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.dims,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.layers,
        num_attention_heads=cfg.att_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        hidden_act="silu",
        max_position_embeddings=cfg.max_context,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,    # Set to False during training to save memory
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        sliding_window=None, 
    )

    # 2. Force Flash Attention 2 implementation via constructor
    # In Transformers v5, this is the standard way to trigger the kernel
    model = MistralForCausalLM(
        configuration, 
        attn_implementation="flash_attention_2"
    )
    
    # 3. Use bfloat16 (Mandatory for Flash Attention 2 efficiency)
    model = model.to(dtype=torch.bfloat16)
    
    print(f"🚀 Mistral Model initialized with native Flash Attention 2.")
    print(f"Total Params: {model.num_parameters():,}")
    
    return model