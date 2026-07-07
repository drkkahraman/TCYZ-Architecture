import torch
import torch.nn as nn
import math

class ModelArgs:
    def __init__(self, **kwargs):
        self.dim = kwargs.get('dim', 4096)
        self.n_layers = kwargs.get('n_layers', 32)
        self.n_heads = kwargs.get('n_heads', 32)
        self.n_kv_heads = kwargs.get('n_kv_heads', 32)
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.hidden_dim = kwargs.get('hidden_dim', 11008)
        self.norm_eps = kwargs.get('norm_eps', 1e-6)
        self.max_seq_len = kwargs.get('max_seq_len', 2048)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin, start_pos):
    seq_len = xq.shape[1]
    cos = freqs_cos[start_pos : start_pos + seq_len].unsqueeze(0).unsqueeze(2)
    sin = freqs_sin[start_pos : start_pos + seq_len].unsqueeze(0).unsqueeze(2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.to(dtype=xq.dtype), xk_out.to(dtype=xk.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = None

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None
        
        self.wq_bias = None
        self.wk_bias = None
        self.wv_bias = None
        self.wo_bias = None
        
        self.cache_k = None
        self.cache_v = None

    def init_cache(self, max_batch_size, max_seq_len, device, dtype):
        self.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
            device=device, dtype=dtype
        )
        self.cache_v = torch.zeros(
            (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
            device=device, dtype=dtype
        )

    def forward(self, x, start_pos, freqs_cos, freqs_sin, mask=None):
        bsz, seqlen, _ = x.shape
        
        xq = nn.functional.linear(x, self.wq, self.wq_bias)
        xk = nn.functional.linear(x, self.wk, self.wk_bias)
        xv = nn.functional.linear(x, self.wv, self.wv_bias)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin, start_pos)
        
        self.cache_k = self.cache_k.to(device=xq.device, dtype=xq.dtype)
        self.cache_v = self.cache_v.to(device=xq.device, dtype=xq.dtype)
        
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        
        if self.n_rep > 1:
            keys = torch.repeat_interleave(keys, self.n_rep, dim=2)
            values = torch.repeat_interleave(values, self.n_rep, dim=2)
            
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = nn.functional.softmax(scores.float(), dim=-1).to(xq.dtype)
        
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return nn.functional.linear(output, self.wo, self.wo_bias)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = None
        self.w2 = None
        self.w3 = None
        
        self.w1_bias = None
        self.w2_bias = None
        self.w3_bias = None

    def forward(self, x):
        return nn.functional.linear(
            nn.functional.silu(nn.functional.linear(x, self.w1, self.w1_bias)) * 
            nn.functional.linear(x, self.w3, self.w3_bias), 
            self.w2, 
            self.w2_bias
        )

class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, args: ModelArgs):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cos, freqs_sin, mask=None):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cos, freqs_sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class TCYZModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = None
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = None
        
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta
        )

    def load_tensors(self, tensors, device='cpu'):
        def load_param(name, required=True):
            if name not in tensors:
                if required:
                    raise KeyError(f"Weight {name} not found in model file")
                return None
            return nn.Parameter(tensors[name].to(device))

        self.tok_embeddings = load_param("tok_embeddings.weight")
        self.norm.weight = load_param("norm.weight")
        
        if "output.weight" in tensors:
            self.output = load_param("output.weight")
        else:
            self.output = self.tok_embeddings

        for i in range(self.args.n_layers):
            layer = self.layers[i]
            layer.attention_norm.weight = load_param(f"layers.{i}.attention_norm.weight")
            layer.ffn_norm.weight = load_param(f"layers.{i}.ffn_norm.weight")
            
            layer.attention.wq = load_param(f"layers.{i}.attention.wq.weight")
            layer.attention.wk = load_param(f"layers.{i}.attention.wk.weight")
            layer.attention.wv = load_param(f"layers.{i}.attention.wv.weight")
            layer.attention.wo = load_param(f"layers.{i}.attention.wo.weight")
            
            layer.attention.wq_bias = load_param(f"layers.{i}.attention.wq.bias", required=False)
            layer.attention.wk_bias = load_param(f"layers.{i}.attention.wk.bias", required=False)
            layer.attention.wv_bias = load_param(f"layers.{i}.attention.wv.bias", required=False)
            layer.attention.wo_bias = load_param(f"layers.{i}.attention.wo.bias", required=False)
            
            layer.feed_forward.w1 = load_param(f"layers.{i}.feed_forward.w1.weight")
            layer.feed_forward.w2 = load_param(f"layers.{i}.feed_forward.w2.weight")
            layer.feed_forward.w3 = load_param(f"layers.{i}.feed_forward.w3.weight")
            
            layer.feed_forward.w1_bias = load_param(f"layers.{i}.feed_forward.w1.bias", required=False)
            layer.feed_forward.w2_bias = load_param(f"layers.{i}.feed_forward.w2.bias", required=False)
            layer.feed_forward.w3_bias = load_param(f"layers.{i}.feed_forward.w3.bias", required=False)

        self.freqs_cos = self.freqs_cos.to(device)
        self.freqs_sin = self.freqs_sin.to(device)

    def init_weights(self, device='cpu'):
        head_dim = self.args.dim // self.args.n_heads
        
        self.tok_embeddings = nn.Parameter(torch.randn(self.args.vocab_size, self.args.dim, device=device) * 0.02)
        self.norm.weight = nn.Parameter(torch.ones(self.args.dim, device=device))
        self.output = nn.Parameter(torch.randn(self.args.vocab_size, self.args.dim, device=device) * 0.02)
        
        for i in range(self.args.n_layers):
            layer = self.layers[i]
            layer.attention_norm.weight = nn.Parameter(torch.ones(self.args.dim, device=device))
            layer.ffn_norm.weight = nn.Parameter(torch.ones(self.args.dim, device=device))
            
            layer.attention.wq = nn.Parameter(torch.randn(self.args.dim, self.args.dim, device=device) * 0.02)
            layer.attention.wk = nn.Parameter(torch.randn(self.args.n_kv_heads * head_dim, self.args.dim, device=device) * 0.02)
            layer.attention.wv = nn.Parameter(torch.randn(self.args.n_kv_heads * head_dim, self.args.dim, device=device) * 0.02)
            layer.attention.wo = nn.Parameter(torch.randn(self.args.dim, self.args.dim, device=device) * 0.02)
            
            layer.feed_forward.w1 = nn.Parameter(torch.randn(self.args.hidden_dim, self.args.dim, device=device) * 0.02)
            layer.feed_forward.w2 = nn.Parameter(torch.randn(self.args.dim, self.args.hidden_dim, device=device) * 0.02)
            layer.feed_forward.w3 = nn.Parameter(torch.randn(self.args.hidden_dim, self.args.dim, device=device) * 0.02)

        self.freqs_cos = self.freqs_cos.to(device)
        self.freqs_sin = self.freqs_sin.to(device)

    def to_state_dict_tcyz(self):
        tensors = {
            "tok_embeddings.weight": self.tok_embeddings.data,
            "norm.weight": self.norm.weight.data,
            "output.weight": self.output.data
        }
        for i in range(self.args.n_layers):
            layer = self.layers[i]
            tensors[f"layers.{i}.attention_norm.weight"] = layer.attention_norm.weight.data
            tensors[f"f.layers.{i}.ffn_norm.weight"] = layer.ffn_norm.weight.data
            tensors[f"layers.{i}.ffn_norm.weight"] = layer.ffn_norm.weight.data
            
            tensors[f"layers.{i}.attention.wq.weight"] = layer.attention.wq.data
            tensors[f"layers.{i}.attention.wk.weight"] = layer.attention.wk.data
            tensors[f"layers.{i}.attention.wv.weight"] = layer.attention.wv.data
            tensors[f"layers.{i}.attention.wo.weight"] = layer.attention.wo.data
            
            if layer.attention.wq_bias is not None:
                tensors[f"layers.{i}.attention.wq.bias"] = layer.attention.wq_bias.data
            if layer.attention.wk_bias is not None:
                tensors[f"layers.{i}.attention.wk.bias"] = layer.attention.wk_bias.data
            if layer.attention.wv_bias is not None:
                tensors[f"layers.{i}.attention.wv.bias"] = layer.attention.wv_bias.data
            if layer.attention.wo_bias is not None:
                tensors[f"layers.{i}.attention.wo.bias"] = layer.attention.wo_bias.data
                
            tensors[f"layers.{i}.feed_forward.w1.weight"] = layer.feed_forward.w1.data
            tensors[f"layers.{i}.feed_forward.w2.weight"] = layer.feed_forward.w2.data
            tensors[f"layers.{i}.feed_forward.w3.weight"] = layer.feed_forward.w3.data
            
            if layer.feed_forward.w1_bias is not None:
                tensors[f"layers.{i}.feed_forward.w1.bias"] = layer.feed_forward.w1_bias.data
            if layer.feed_forward.w2_bias is not None:
                tensors[f"layers.{i}.feed_forward.w2.bias"] = layer.feed_forward.w2_bias.data
            if layer.feed_forward.w3_bias is not None:
                tensors[f"layers.{i}.feed_forward.w3.bias"] = layer.feed_forward.w3_bias.data
                
        return tensors

    def init_kv_cache(self, batch_size, max_seq_len, device, dtype):
        for layer in self.layers:
            layer.attention.init_cache(batch_size, max_seq_len, device, dtype)

    def forward(self, tokens, start_pos, return_last_only=True):
        bsz, seqlen = tokens.shape
        h = nn.functional.embedding(tokens, self.tok_embeddings)
        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ])
            mask = mask.unsqueeze(0).unsqueeze(1)

        for layer in self.layers:
            h = layer(h, start_pos, self.freqs_cos, self.freqs_sin, mask)
            
        h = self.norm(h)
        if return_last_only:
            logits = nn.functional.linear(h[:, -1:, :], self.output)
            return logits.squeeze(1)
        else:
            logits = nn.functional.linear(h, self.output)
            return logits

