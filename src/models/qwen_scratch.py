"""
Qwen2.5 Causal Language Model Backbone implemented from scratch in pure PyTorch.
Architecture: Decoder-only Transformer with SwiGLU Gated MLP, RMSNorm, Rotary Position Embeddings (RoPE), and Causal Self-Attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    """SwiGLU Gated MLP block as used in Qwen2.5"""
    def __init__(self, hidden_size: int = 1536, intermediate_size: int = 8960):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    def __init__(self, hidden_size: int = 1536, num_heads: int = 12, num_kv_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = Qwen2RotaryEmbedding(self.head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat KV heads for Grouped Query Attention (GQA)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
            
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply Causal Mask
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device), diagonal=1)
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 1536, num_heads: int = 12, intermediate_size: int = 8960):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(hidden_size)
        self.self_attn = Qwen2Attention(hidden_size=hidden_size, num_heads=num_heads)
        self.post_attention_layernorm = Qwen2RMSNorm(hidden_size)
        self.mlp = Qwen2MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen2ForCausalLMScratch(nn.Module):
    def __init__(
        self, 
        vocab_size: int = 151936, 
        hidden_size: int = 1536, 
        num_layers: int = 4, 
        num_heads: int = 12, 
        intermediate_size: int = 8960
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(hidden_size=hidden_size, num_heads=num_heads, intermediate_size=intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = Qwen2RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Config object compatibility
        self.config = type('Config', (), {'hidden_size': hidden_size, 'vocab_size': vocab_size})()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return type('CausalLMOutput', (), {'logits': logits})()
