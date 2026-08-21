"""Modded nanoGPT with optional sparse-basis token embeddings.

Tokens can be a lookup table, a dense factorisation (ALBERT-style), or a
TopK-sparse linear combination of a shared concept basis. The last option is
the sample-efficiency experiment: unused tokens still move when they share
basis vectors with tokens that appeared in the batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

EmbeddingType = Literal["dense", "factorized", "sparse"]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rope_cache(seq_len: int, head_dim: int, device: torch.device, theta: float = 10_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freq)
    return angles.cos(), angles.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim), cos/sin: (T, head_dim/2)
    b, h, t, d = x.shape
    x = x.view(b, h, t, d // 2, 2)
    x0, x1 = x.unbind(-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack((out0, out1), dim=-1).flatten(-2)


class SparseBasisEmbedding(nn.Module):
    """Token embedding as (optionally sparse) codes times a shared basis.

    ``e_i = codes[i] @ basis`` with TopK / ReLU on ``codes``. Gradients into
    ``basis`` from any used token move every other token that uses those atoms.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_basis: int,
        k_sparse: int | None = None,
        nonnegative: bool = True,
    ) -> None:
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_basis must be positive")
        if k_sparse is not None and not (1 <= k_sparse <= n_basis):
            raise ValueError(f"k_sparse must be in [1, n_basis], got {k_sparse}")
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_basis = n_basis
        self.k_sparse = k_sparse
        self.nonnegative = nonnegative
        self.basis = nn.Parameter(torch.empty(n_basis, n_embd))
        self.codes = nn.Parameter(torch.empty(vocab_size, n_basis))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.basis, mean=0.0, std=0.02)
        if self.k_sparse is None:
            nn.init.normal_(self.codes, mean=0.0, std=0.02)
            return
        nn.init.zeros_(self.codes)
        # Each token starts on k random atoms so unused tokens already share
        # some basis vectors with the rest of the vocabulary.
        idx = torch.randint(0, self.n_basis, (self.vocab_size, self.k_sparse))
        vals = torch.full((self.vocab_size, self.k_sparse), 1.0 / math.sqrt(self.k_sparse))
        self.codes.data.scatter_add_(1, idx, vals)

    def activate_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self.nonnegative:
            codes = F.relu(codes)
        if self.k_sparse is None or self.k_sparse >= codes.size(-1):
            return codes
        topv, topi = torch.topk(codes, self.k_sparse, dim=-1)
        sparse = torch.zeros_like(codes)
        return sparse.scatter(-1, topi, topv)

    def sparse_codes(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        codes = self.codes if idx is None else self.codes[idx]
        return self.activate_codes(codes)

    def embedding_matrix(self) -> torch.Tensor:
        return self.sparse_codes() @ self.basis

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.sparse_codes(idx) @ self.basis


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _c = x.shape
        qkv = self.c_attn(x).view(b, t, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = _rope_cache(t, self.head_dim, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_embd)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    embedding_type: EmbeddingType = "dense"
    n_basis: int = 64
    k_sparse: int = 8
    dropout: float = 0.0


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        if config.embedding_type == "dense":
            self.wte: nn.Module = nn.Embedding(config.vocab_size, config.n_embd)
        else:
            k_sparse = config.k_sparse if config.embedding_type == "sparse" else None
            self.wte = SparseBasisEmbedding(
                vocab_size=config.vocab_size,
                n_embd=config.n_embd,
                n_basis=config.n_basis,
                k_sparse=k_sparse,
                nonnegative=config.embedding_type == "sparse",
            )
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.apply(self._init_weights)
        for module in self.modules():
            if isinstance(module, (CausalSelfAttention, MLP)):
                nn.init.zeros_(module.c_proj.weight)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def token_embeddings(self) -> torch.Tensor:
        if isinstance(self.wte, SparseBasisEmbedding):
            return self.wte.embedding_matrix()
        return self.wte.weight

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.wte(idx)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        weight = self.token_embeddings()
        logits = F.linear(x, weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, learning_rate: float, weight_decay: float) -> list[torch.optim.Optimizer]:
        muon_params: list[nn.Parameter] = []
        adam_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            hidden_linear = param.ndim == 2 and ("wte" not in name) and ("codes" not in name) and ("basis" not in name)
            if hidden_linear:
                muon_params.append(param)
            else:
                adam_params.append(param)
        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(torch.optim.Muon(muon_params, lr=learning_rate * 10.0, weight_decay=weight_decay))
        if adam_params:
            optimizers.append(torch.optim.AdamW(adam_params, lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay))
        return optimizers
