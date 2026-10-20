"""
Vision-language connectors compared in the study. All map frozen vision patch features
[B, P, vision_dim] into the LLM; parameter counts are kept in the same range (~8-11M at
llm_dim=2560) so differences come from the design, not capacity.

- MLPProjector:      LLaVA-1.5 style. Every patch becomes one LLM token (P prefix tokens).
- QFormerConnector:  BLIP-2 style. A fixed set of learned queries cross-attends to the patches;
                     the queries become prefix tokens (num_queries tokens).
- CrossAttnBottleneck: no image tokens. Each text token cross-attends to the patches and the
                     attended context is added to its embedding through a zero-initialised gate.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    kind = "prefix"

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim), nn.Linear(vision_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.net(patches), None


class QFormerConnector(nn.Module):
    """One-block Q-Former: learned queries -> self-attention -> cross-attention to patches -> MLP."""
    kind = "prefix"

    def __init__(self, vision_dim: int, llm_dim: int, num_queries: int = 32, num_heads: int = 8):
        super().__init__()
        d = vision_dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, d) * 0.02)
        self.norm_q1, self.norm_q2, self.norm_q3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, llm_dim))

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.queries.expand(patches.shape[0], -1, -1)
        h = self.norm_q1(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        kv = self.norm_kv(patches)
        attended, attn = self.cross_attn(self.norm_q2(q), kv, kv, need_weights=True)
        q = q + attended
        q = q + self.mlp(self.norm_q3(q))
        return self.out(q), attn  # attn: [B, num_queries, P]


class CrossAttnBottleneck(nn.Module):
    """Text tokens query the patches in a reduced attention space (attn_dim) and add the result
    to their own embedding through tanh(gate), gate initialised at 0."""
    kind = "fuse"

    def __init__(self, vision_dim: int, llm_dim: int, attn_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.norm_text = nn.LayerNorm(llm_dim)
        self.norm_kv = nn.LayerNorm(vision_dim)
        self.q_proj = nn.Linear(llm_dim, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads, kdim=vision_dim, vdim=vision_dim, batch_first=True)
        self.out_proj = nn.Linear(attn_dim, llm_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, patches: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kv = self.norm_kv(patches)
        q = self.q_proj(self.norm_text(text))
        attended, attn = self.attn(q, kv, kv, need_weights=True)
        return text + torch.tanh(self.gate) * self.out_proj(attended), attn  # attn: [B, T, P]


CONNECTORS = {"mlp": MLPProjector, "qformer": QFormerConnector, "bottleneck": CrossAttnBottleneck}


def build_connector(name: str, vision_dim: int, llm_dim: int) -> nn.Module:
    return CONNECTORS[name](vision_dim, llm_dim)
