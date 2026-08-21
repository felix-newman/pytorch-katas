"""Tiny Scalable Interpolant Transformer with per-token timesteps.

SiT (Ma et al., 2024) is DiT with a linear interpolant / flow-matching
objective. Self-Flow needs *per-token* timesteps so a mask can mix two noise
levels inside one sequence; AdaLN therefore conditions on ``(B, N, D)`` rather
than a single vector per image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Sinusoidal embedding of times in ``[0, 1]``. ``t`` may be any leading shape."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * 1000.0 * freqs
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb.to(dtype=t.dtype if t.is_floating_point() else torch.float32)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class MHA(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x_bnd: Tensor) -> Tensor:
        b, n, _ = x_bnd.shape
        qkv = self.qkv(x_bnd).reshape(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(attn.transpose(1, 2).reshape(b, n, self.dim))


class SiTBlock(nn.Module):
    """AdaLN-Zero transformer block. Conditioning ``c`` is ``(B, N, D)`` or ``(B, 1, D)``."""

    def __init__(self, dim: int, n_heads: int, exp_factor: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MHA(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = dim * exp_factor
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.proj = nn.Linear(dim, patch_dim)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


@dataclass
class SiTOutput:
    velocity: Tensor
    features: dict[int, Tensor]


class MiniSiT(nn.Module):
    """Class-conditional SiT that lives in patch-token space.

    ``forward`` takes already-noised tokens ``x_bnd`` of shape ``(B, N, patch_dim)``
    and a per-token (or broadcastable) timestep ``t`` of shape ``(B, N)``.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        n_classes: int = 10,
        exp_factor: int = 4,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.n_layers = n_layers
        self.grid = image_size // patch_size
        self.n_tokens = self.grid * self.grid
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.y_embed = nn.Embedding(n_classes, dim)
        self.blocks = nn.ModuleList(SiTBlock(dim, n_heads, exp_factor) for _ in range(n_layers))
        self.final = FinalLayer(dim, self.patch_dim)

    def patchify(self, x_bchw: Tensor) -> Tensor:
        b, c, h, w = x_bchw.shape
        p = self.patch_size
        x = x_bchw.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, self.n_tokens, self.patch_dim)
        return x

    def unpatchify(self, x_bnd: Tensor) -> Tensor:
        b = x_bnd.shape[0]
        p = self.patch_size
        g = self.grid
        x = x_bnd.reshape(b, g, g, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.in_channels, g * p, g * p)
        return x

    def condition(self, t: Tensor, y: Tensor) -> Tensor:
        """Build AdaLN condition. ``t`` is ``(B, N)`` or ``(B,)``; ``y`` is ``(B,)``."""
        if t.ndim == 1:
            t = t[:, None].expand(-1, self.n_tokens)
        t_freq = timestep_embedding(t, self.dim)
        t_emb = self.t_embed(t_freq)
        y_emb = self.y_embed(y)[:, None, :]
        return t_emb + y_emb

    def forward(
        self,
        x_bnd: Tensor,
        t: Tensor,
        y: Tensor,
        *,
        return_layers: tuple[int, ...] | None = None,
    ) -> SiTOutput:
        c = self.condition(t, y)
        h = self.patch_embed(x_bnd) + self.pos_embedding
        features: dict[int, Tensor] = {}
        for i, block in enumerate(self.blocks):
            h = block(h, c)
            if return_layers is not None and i in return_layers:
                features[i] = h
        velocity = self.final(h, c)
        return SiTOutput(velocity=velocity, features=features)
