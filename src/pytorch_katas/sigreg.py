"""Sketched Isotropic Gaussian Regularization (SIGReg) from LeJEPA.

Balestriero & LeCun, arXiv:2511.08544. Embeddings should be isotropic Gaussian
to minimize downstream risk; SIGReg is a sliced Epps-Pulley test that pushes a
batch of vectors toward N(0, I) without a teacher, stop-gradient, or EMA.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _unit_directions(
    dim: int,
    num_slices: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    directions = torch.randn(dim, num_slices, dtype=dtype, device=device, generator=generator)
    return directions / directions.norm(p=2, dim=0, keepdim=True)


def epps_pulley(
    projections: Tensor,
    t: Tensor,
) -> Tensor:
    """Epps-Pulley statistic for 1-D samples already projected onto slices.

    ``projections`` is ``(N, S)`` (N samples, S slices). ``t`` is the quadrature
    grid. The target characteristic function of N(0, 1) is ``exp(-t^2 / 2)``,
    and the same Gaussian is used as the integration window.

    Returns one statistic per slice, scaled by N as in LeJEPA Algorithm 1.
    """
    n_samples = projections.shape[-2]
    tw = projections.unsqueeze(-1) * t
    cos_mean = tw.cos().mean(dim=-3)
    sin_mean = tw.sin().mean(dim=-3)
    phi = torch.exp(-0.5 * t.square())
    sq_err = (cos_mean - phi).square() + sin_mean.square()
    weighted = sq_err * phi
    return torch.trapezoid(weighted, t, dim=-1) * n_samples


def sigreg_loss(
    embeddings: Tensor,
    num_slices: int = 256,
    n_knots: int = 17,
    t_max: float = 5.0,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Mean sliced Epps-Pulley distance of ``embeddings`` from N(0, I).

    SIGReg constrains the *marginal* of the batch. Perfectly identical vectors
    get identical gradients, so exact collapse cannot break symmetry; a real
    minibatch of distinct patches already has enough seed variation.

    Args:
        embeddings: ``(N, D)`` feature vectors. Flatten token grids to this
            shape yourself if you want per-token SIGReg.
        num_slices: random directions on the sphere (LeJEPA default is 256-1024).
        n_knots: trapezoid knots on ``[-t_max, t_max]``.
        t_max: integration limit used in LeJEPA Algorithm 1 (``5.0``).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected embeddings of shape (N, D), got {tuple(embeddings.shape)}")
    if embeddings.shape[0] < 2:
        raise ValueError("SIGReg needs at least 2 samples to estimate a characteristic function")

    t = torch.linspace(-t_max, t_max, n_knots, device=embeddings.device, dtype=embeddings.dtype)
    directions = _unit_directions(
        embeddings.shape[1],
        num_slices,
        dtype=embeddings.dtype,
        device=embeddings.device,
        generator=generator,
    )
    projections = embeddings @ directions
    return epps_pulley(projections, t).mean()


class SIGReg(torch.nn.Module):
    """Cached-quadrature wrapper around :func:`sigreg_loss`."""

    def __init__(self, num_slices: int = 256, n_knots: int = 17, t_max: float = 5.0) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.n_knots = n_knots
        self.t_max = t_max
        t = torch.linspace(-t_max, t_max, n_knots)
        self.register_buffer("t", t)

    def forward(self, embeddings: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"expected embeddings of shape (N, D), got {tuple(embeddings.shape)}")
        directions = _unit_directions(
            embeddings.shape[1],
            self.num_slices,
            dtype=embeddings.dtype,
            device=embeddings.device,
            generator=generator,
        )
        return epps_pulley(embeddings @ directions, self.t.to(dtype=embeddings.dtype)).mean()

    def extra_repr(self) -> str:
        return f"num_slices={self.num_slices}, n_knots={self.n_knots}, t_max={self.t_max}"
