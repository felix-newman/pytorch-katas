"""Sketched Isotropic Gaussian Regularization (SIGReg).

From LeJEPA (Balestriero & LeCun, arXiv:2511.08544): push embeddings toward
``N(0, I)`` by projecting onto random unit directions and matching each 1-D
slice to a standard normal via the Epps–Pulley characteristic-function test.

The loss is minimized when the batch has zero mean, identity covariance, and
Gaussian marginals along every direction — which is why a single SIGReg term
prevents representation collapse without stop-gradients or EMA teachers.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

Reduction = Literal["mean", "sum", "none"]


def epps_pulley_quadrature(
    n_knots: int = 17,
    t_max: float = 3.0,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Trapezoidal knots and folded Gaussian-window weights for Epps–Pulley.

    The integrand ``|φ̂(t) - φ(t)|² w(t)`` is even in ``t``, so we integrate on
    ``[0, t_max]`` and double the trapezoid weights (same as integrating on
    ``[-t_max, t_max]``). ``φ(t) = w(t) = exp(-t²/2)``.
    """
    if n_knots < 3 or n_knots % 2 == 0:
        raise ValueError(f"n_knots must be an odd integer >= 3, got {n_knots}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    t = torch.linspace(0.0, t_max, n_knots, dtype=dtype, device=device)
    dt = t_max / (n_knots - 1)

    # Folded trapezoid: endpoints get dt, interior 2*dt.
    trapezoid = torch.full((n_knots,), 2.0 * dt, dtype=dtype, device=device)
    trapezoid[0] = dt
    trapezoid[-1] = dt

    window = torch.exp(-0.5 * t.square())
    return t, trapezoid * window


def _sample_unit_directions(
    dim: int,
    num_slices: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``num_slices`` directions uniformly on the unit sphere ``S^{dim-1}``."""
    directions = torch.randn(dim, num_slices, dtype=dtype, device=device, generator=generator)
    return directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)


def _epps_pulley_stats(projections: Tensor, t: Tensor, weights: Tensor) -> Tensor:
    """Per-slice Epps–Pulley statistics for projections of shape ``(..., N, K)``."""
    n_samples = projections.shape[-2]
    # (..., N, K, T)
    xt = projections.unsqueeze(-1) * t
    # Empirical CF of the slice vs real CF of N(0, 1).
    cos_mean = xt.cos().mean(dim=-3)
    sin_mean = xt.sin().mean(dim=-3)
    phi = torch.exp(-0.5 * t.square())
    sq_err = (cos_mean - phi).square() + sin_mean.square()
    return torch.matmul(sq_err, weights) * n_samples


def sigreg_loss(
    embeddings: Tensor,
    num_slices: int = 1024,
    n_knots: int = 17,
    t_max: float = 3.0,
    *,
    reduction: Reduction = "mean",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Functional SIGReg: sliced Epps–Pulley distance to ``N(0, I)``.

    Args:
        embeddings: ``(..., N, D)`` batch of embeddings. Do **not** center/whiten
            before calling — that would mask collapse.
        num_slices: number of random 1-D projections.
        n_knots: odd quadrature resolution (paper default 17).
        t_max: folded integration upper bound (practical default 3.0).
        reduction: ``"mean"`` / ``"sum"`` over slices, or ``"none"`` for per-slice stats.
        generator: optional seeded RNG for reproducible directions.
    """
    if embeddings.ndim < 2:
        raise ValueError(f"embeddings must have shape (..., N, D), got {tuple(embeddings.shape)}")

    t, weights = epps_pulley_quadrature(
        n_knots, t_max, dtype=embeddings.dtype, device=embeddings.device
    )
    directions = _sample_unit_directions(
        embeddings.shape[-1],
        num_slices,
        dtype=embeddings.dtype,
        device=embeddings.device,
        generator=generator,
    )
    stats = _epps_pulley_stats(embeddings @ directions, t, weights)

    if reduction == "mean":
        return stats.mean(dim=-1)
    if reduction == "sum":
        return stats.sum(dim=-1)
    if reduction == "none":
        return stats
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


class SIGReg(nn.Module):
    """``nn.Module`` form of :func:`sigreg_loss` with cached quadrature buffers.

    Example::

        criterion = SIGReg(num_slices=1024)
        loss = criterion(embeddings)  # embeddings: (N, D) or (V, N, D)
        loss.backward()
    """

    t: Tensor
    weights: Tensor

    def __init__(
        self,
        num_slices: int = 1024,
        n_knots: int = 17,
        t_max: float = 3.0,
        *,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.n_knots = n_knots
        self.t_max = t_max
        self.reduction: Reduction = reduction
        t, weights = epps_pulley_quadrature(n_knots, t_max)
        self.register_buffer("t", t)
        self.register_buffer("weights", weights)

    def forward(
        self,
        embeddings: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if embeddings.ndim < 2:
            raise ValueError(
                f"embeddings must have shape (..., N, D), got {tuple(embeddings.shape)}"
            )

        t = self.t.to(dtype=embeddings.dtype)
        weights = self.weights.to(dtype=embeddings.dtype)
        directions = _sample_unit_directions(
            embeddings.shape[-1],
            self.num_slices,
            dtype=embeddings.dtype,
            device=embeddings.device,
            generator=generator,
        )
        stats = _epps_pulley_stats(embeddings @ directions, t, weights)

        if self.reduction == "mean":
            return stats.mean(dim=-1)
        if self.reduction == "sum":
            return stats.sum(dim=-1)
        if self.reduction == "none":
            return stats
        raise ValueError(f"invalid reduction {self.reduction!r}")

    def extra_repr(self) -> str:
        return (
            f"num_slices={self.num_slices}, n_knots={self.n_knots}, "
            f"t_max={self.t_max}, reduction={self.reduction!r}"
        )
