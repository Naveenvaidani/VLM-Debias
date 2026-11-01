"""
Vision-language connectors compared in the study. All map frozen vision patch features
[B, P, vision_dim] into the LLM; parameter counts are kept in the same range (~8-11M at
llm_dim=2560) so differences come from the design, not capacity.

- MLPProjector:      LLaVA-1.5 style. Every patch becomes one LLM token (P prefix tokens), or one
                     token per pool x pool patch window (P / pool^2 tokens) when pool > 1.
- QFormerConnector:  BLIP-2 style. A fixed set of learned queries cross-attends to the patches;
                     the queries become prefix tokens (num_queries tokens).
- CrossAttnBottleneck: no image tokens. Each text token cross-attends to the patches and the
                     attended context is added to its embedding through a zero-initialised gate.
"""
from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    kind = "prefix"

    def __init__(self, vision_dim: int, llm_dim: int, pool: int = 1, grid: int = 0):
        super().__init__()
        self.pool, self.grid = pool, grid  # grid > 0: adaptive pooling to grid x grid tokens (overrides pool)
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim), nn.Linear(vision_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.pool > 1 or self.grid:
            b, p, d = patches.shape
            side = int(p ** 0.5)
            grid = patches.transpose(1, 2).reshape(b, d, side, side)
            grid = F.adaptive_avg_pool2d(grid, self.grid) if self.grid else F.avg_pool2d(grid, self.pool)
            patches = grid.flatten(2).transpose(1, 2)
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
    to their own embedding through tanh(gate), gate initialised at 0.

    gate="fixed" drops the learnable gate (scale 1) and zero-initialises out_proj instead, so the
    connector still starts as the identity but the optimiser cannot shut the visual pathway.
    gate="open" keeps the gate learnable but starts it open (tanh(3) = 0.995) with out_proj zero-initialised:
    it separates "the gate is trainable" from "the gate starts closed". At alpha = 3 the gate gradient is scaled by
    1 - tanh^2 = 0.01, so it barely moves; gate="half" starts at tanh(alpha) = 0.5 (gradient scale 0.75), learnable
    and free to move either way.
    Why the zero-gate arm keeps the default out_proj init: with both the gate and out_proj at zero, neither receives
    a gradient (d/d gate is proportional to out_proj, d/d out_proj to tanh(gate)) and the pathway never trains."""
    kind = "fuse"

    def __init__(self, vision_dim: int, llm_dim: int, attn_dim: int = 1024, num_heads: int = 8,
                 gate: str = "learned"):
        super().__init__()
        self.norm_text = nn.LayerNorm(llm_dim)
        self.norm_kv = nn.LayerNorm(vision_dim)
        self.q_proj = nn.Linear(llm_dim, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads, kdim=vision_dim, vdim=vision_dim, batch_first=True)
        self.out_proj = nn.Linear(attn_dim, llm_dim)
        if gate == "learned":
            self.gate = nn.Parameter(torch.zeros(1))
        elif gate in ("open", "half"):
            self.gate = nn.Parameter(torch.full((1,), 3.0 if gate == "open" else math.atanh(0.5)))
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            self.register_buffer("gate", torch.full((1,), float("inf")))  # tanh(inf) = 1
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, patches: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kv = self.norm_kv(patches)
        q = self.q_proj(self.norm_text(text))
        attended, attn = self.attn(q, kv, kv, need_weights=True)
        return text + torch.tanh(self.gate) * self.out_proj(attended), attn  # attn: [B, T, P]


class DeepGatedCrossAttn(nn.Module):
    """Per-layer injection as in Flamingo and Llama 3.2 Vision: a tanh-gated cross-attention block before every
    (L / n_blocks)-th LLM layer, starting at layer 0. The hidden states at that depth query the patches, and each
    block has its own gate. The blocks are applied by forward pre-hooks that StudyVLM registers on the decoder
    layers. attn_dim 320 with 4 blocks gives about the same parameter count as the single input-level block
    (9.3M vs 8.9M for a 2560-wide LLM). gate="fixed" zero-initialises each block's output instead."""
    kind = "deep"

    def __init__(self, vision_dim: int, llm_dim: int, gate: str = "learned", n_blocks: int = 4,
                 attn_dim: int = 320, num_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([CrossAttnBottleneck(vision_dim, llm_dim, attn_dim, num_heads, gate=gate)
                                     for _ in range(n_blocks)])

    def learned_gates(self):
        return [b.gate for b in self.blocks if isinstance(b.gate, nn.Parameter)]

    def gate_values(self):
        return [float(torch.tanh(b.gate.detach().float()).item()) for b in self.blocks]


CONNECTORS = {"mlp": MLPProjector, "qformer": QFormerConnector, "bottleneck": CrossAttnBottleneck,
              "bottleneck_fixed": lambda v, l: CrossAttnBottleneck(v, l, gate="fixed"),
              "bottleneck_open": lambda v, l: CrossAttnBottleneck(v, l, gate="open"),
              "bottleneck_half": lambda v, l: CrossAttnBottleneck(v, l, gate="half"),
              "xattn_deep": DeepGatedCrossAttn,
              "xattn_deep_fixed": lambda v, l: DeepGatedCrossAttn(v, l, gate="fixed")}
CONNECTOR_NAMES = list(CONNECTORS)


def build_connector(name: str, vision_dim: int, llm_dim: int, mlp_pool: int = 1, mlp_grid: int = 0) -> nn.Module:
    if name == "mlp":
        return MLPProjector(vision_dim, llm_dim, pool=mlp_pool, grid=mlp_grid)
    return CONNECTORS[name](vision_dim, llm_dim)
