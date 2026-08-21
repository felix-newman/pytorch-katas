from __future__ import annotations

import torch
from torch import Tensor


def collapse_metrics(tokens: Tensor) -> dict[str, float]:
    """Cheap diagnostics for whether token embeddings have collapsed.

    ``tokens`` is ``(B, N, D)`` or ``(N, D)``.
    """
    z = tokens.reshape(-1, tokens.shape[-1]).float()
    centered = z - z.mean(dim=0, keepdim=True)
    std = centered.std().item()
    # Mean absolute cosine between random pairs: ~0 for isotropic Gaussian, ~1 for collapse.
    idx = torch.randperm(z.shape[0], device=z.device)
    a, b = z[idx[: z.shape[0] // 2]], z[idx[z.shape[0] // 2 : 2 * (z.shape[0] // 2)]]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).abs().mean().item()
    return {"std": std, "abs_cosine": cos, "mean_norm": z.norm(dim=-1).mean().item()}
