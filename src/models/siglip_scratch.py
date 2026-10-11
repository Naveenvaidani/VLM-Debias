"""
SigLIP (Sigmoid Loss Vision Language Pre-training) Vision Encoder implemented from scratch in PyTorch.
Architecture: Vision Transformer (ViT) with patch size 16x16, 224x224 input images -> 196 patch tokens.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, embed_dim: int = 768, in_channels: int = 3):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2 # 14x14 = 196
        
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        self.position_embedding = nn.Embedding(self.num_patches, embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_patches).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch, 3, 224, 224]
        patch_embeds = self.patch_embedding(pixel_values) # [batch, 768, 14, 14]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # [batch, 196, 768]
        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_probs, v) # [batch, heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class SiglipMLP(nn.Module):
    def __init__(self, embed_dim: int = 768, intermediate_size: int = 3072):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, embed_dim)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class SiglipEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, intermediate_size: int = 3072):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_attn = SiglipAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = SiglipMLP(embed_dim=embed_dim, intermediate_size=intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class SiglipVisionEncoderScratch(nn.Module):
    def __init__(
        self, 
        image_size: int = 224, 
        patch_size: int = 16, 
        embed_dim: int = 768, 
        num_layers: int = 12, 
        num_heads: int = 12, 
        intermediate_size: int = 3072
    ):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim)
        self.layers = nn.ModuleList([
            SiglipEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, intermediate_size=intermediate_size)
            for _ in range(num_layers)
        ])
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values) # [B, 196, 768]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states # Returns dense spatial patch embeddings [B, 196, 768]
