import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from .format import (
    write_tcyz, TYPE_INT32, TYPE_FLOAT32, TYPE_STRING
)

def get_huggingface_mapping(num_layers):
    mapping = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight"
    }
    for i in range(num_layers):
        mapping[f"model.layers.{i}.input_layernorm.weight"] = f"layers.{i}.attention_norm.weight"
        mapping[f"model.layers.{i}.post_attention_layernorm.weight"] = f"layers.{i}.ffn_norm.weight"
        
        mapping[f"model.layers.{i}.self_attn.q_proj.weight"] = f"layers.{i}.attention.wq.weight"
        mapping[f"model.layers.{i}.self_attn.k_proj.weight"] = f"layers.{i}.attention.wk.weight"
        mapping[f"model.layers.{i}.self_attn.v_proj.weight"] = f"layers.{i}.attention.wv.weight"
        mapping[f"model.layers.{i}.self_attn.o_proj.weight"] = f"layers.{i}.attention.wo.weight"
        
        mapping[f"model.layers.{i}.self_attn.q_proj.bias"] = f"layers.{i}.attention.wq.bias"
        mapping[f"model.layers.{i}.self_attn.k_proj.bias"] = f"layers.{i}.attention.wk.bias"
        mapping[f"model.layers.{i}.self_attn.v_proj.bias"] = f"layers.{i}.attention.wv.bias"
        mapping[f"model.layers.{i}.self_attn.o_proj.bias"] = f"layers.{i}.attention.wo.bias"
        
        mapping[f"model.layers.{i}.mlp.gate_proj.weight"] = f"layers.{i}.feed_forward.w1.weight"
        mapping[f"model.layers.{i}.mlp.down_proj.weight"] = f"layers.{i}.feed_forward.w2.weight"
        mapping[f"model.layers.{i}.mlp.up_proj.weight"] = f"layers.{i}.feed_forward.w3.weight"
        
        mapping[f"model.layers.{i}.mlp.gate_proj.bias"] = f"layers.{i}.feed_forward.w1.bias"
        mapping[f"model.layers.{i}.mlp.down_proj.bias"] = f"layers.{i}.feed_forward.w2.bias"
        mapping[f"model.layers.{i}.mlp.up_proj.bias"] = f"layers.{i}.feed_forward.w3.bias"
        
    return mapping

def convert_hf_to_tcyz(model_name_or_path, output_path, target_dtype="float16"):
    print(f"Loading configuration from {model_name_or_path}...")
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    dim = getattr(config, "hidden_size", None)
    n_layers = getattr(config, "num_hidden_layers", None)
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    vocab_size = getattr(config, "vocab_size", None)
    hidden_dim = getattr(config, "intermediate_size", None)
    norm_eps = getattr(config, "rms_norm_eps", 1e-6)
    max_seq_len = getattr(config, "max_position_embeddings", 2048)
    
    rope_theta = 10000.0
    if hasattr(config, "rope_theta"):
        rope_theta = config.rope_theta
    elif hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        if "rope_theta" in config.rope_scaling:
            rope_theta = config.rope_scaling["rope_theta"]
            
    if None in (dim, n_layers, n_heads, vocab_size, hidden_dim):
        raise ValueError("Could not extract all necessary architecture parameters from configuration.")
        
    print(f"Loading weights from {model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if target_dtype == "float16" else (torch.bfloat16 if target_dtype == "bfloat16" else torch.float32),
        device_map="cpu",
        trust_remote_code=True
    )
    
    state_dict = model.state_dict()
    
    metadata = {
        "dim": (TYPE_INT32, int(dim)),
        "n_layers": (TYPE_INT32, int(n_layers)),
        "n_heads": (TYPE_INT32, int(n_heads)),
        "n_kv_heads": (TYPE_INT32, int(n_kv_heads)),
        "vocab_size": (TYPE_INT32, int(vocab_size)),
        "hidden_dim": (TYPE_INT32, int(hidden_dim)),
        "norm_eps": (TYPE_FLOAT32, float(norm_eps)),
        "max_seq_len": (TYPE_INT32, int(max_seq_len)),
        "rope_theta": (TYPE_FLOAT32, float(rope_theta)),
        "tokenizer_name": (TYPE_STRING, model_name_or_path)
    }
    
    mapping = get_huggingface_mapping(n_layers)
    tensors = {}
    
    for hf_name, tcyz_name in mapping.items():
        if hf_name in state_dict:
            t = state_dict[hf_name]
            if target_dtype == "float16":
                t = t.to(torch.float16)
            elif target_dtype == "bfloat16":
                t = t.to(torch.bfloat16)
            elif target_dtype == "float32":
                t = t.to(torch.float32)
            tensors[tcyz_name] = t
            
    print(f"Writing {len(tensors)} tensors to {output_path}...")
    write_tcyz(output_path, metadata, tensors)
    print("Conversion complete!")
