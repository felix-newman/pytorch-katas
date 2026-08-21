"""Self-Flow training recipes, including a teacher-free SIGReg variant.

Self-Flow (Chefer et al., 2026) trains a flow model with Dual-Timestep
Scheduling plus an EMA teacher whose cleaner-view features the student predicts.
That teacher/EMA pair is a JEPA mechanism. LeJEPA's SIGReg can take over the
anti-collapse job, so the SiT's own token embeddings can be forced toward
N(0, I) without an external encoder or a momentum copy.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pytorch_katas.sigreg import SIGReg
from pytorch_katas.sit import MiniSiT, SiTOutput


class Recipe(str, Enum):
    VANILLA = "vanilla"
    SELF_FLOW = "self_flow"
    SIGREG = "sigreg"
    LEJEPA_FLOW = "lejepa_flow"


class ProjectionHead(nn.Module):
    """Shallow MLP used by REPA / Self-Flow to map student features before cosine alignment."""

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class DualTimestep:
    tau: Tensor
    tau_min: Tensor
    mask: Tensor


def sample_dual_timestep(
    batch: int,
    n_tokens: int,
    *,
    mask_ratio: float = 0.25,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> DualTimestep:
    """Sample two times and assign the second one to a random token subset.

    The cleaner time ``tau_min = min(t, s)`` is what the Self-Flow teacher sees.
    ``mask_ratio`` is capped at 0.5 in the paper so inference (uniform t) stays
    on-distribution.
    """
    if not 0.0 <= mask_ratio <= 0.5:
        raise ValueError(f"mask_ratio must be in [0, 0.5], got {mask_ratio}")
    t = torch.rand(batch, 1, device=device, generator=generator)
    s = torch.rand(batch, 1, device=device, generator=generator)
    mask = torch.rand(batch, n_tokens, device=device, generator=generator) < mask_ratio
    tau = torch.where(mask, s.expand(-1, n_tokens), t.expand(-1, n_tokens))
    tau_min = torch.minimum(t, s).expand(-1, n_tokens)
    return DualTimestep(tau=tau, tau_min=tau_min, mask=mask)


def interpolate(x0: Tensor, x1: Tensor, tau: Tensor) -> Tensor:
    """Rectified-flow interpolant with per-token times. ``tau`` is ``(B, N)``."""
    w = tau.unsqueeze(-1)
    return (1.0 - w) * x0 + w * x1


@torch.no_grad()
def ema_update(ema: nn.Module, online: nn.Module, decay: float = 0.999) -> None:
    for p_ema, p_online in zip(ema.parameters(), online.parameters(), strict=True):
        p_ema.lerp_(p_online, 1.0 - decay)
    for b_ema, b_online in zip(ema.buffers(), online.buffers(), strict=True):
        b_ema.copy_(b_online)


def cosine_align(student: Tensor, teacher: Tensor) -> Tensor:
    return -F.cosine_similarity(student, teacher, dim=-1).mean()


@dataclass
class StepLosses:
    total: Tensor
    gen: Tensor
    rep: Tensor
    sigreg: Tensor


class SelfFlowTrainer(nn.Module):
    """One training step of vanilla SiT, Self-Flow, or the SIGReg replacements."""

    def __init__(
        self,
        model: MiniSiT,
        recipe: Recipe,
        *,
        student_layer: int = 1,
        teacher_layer: int | None = None,
        gamma: float = 0.5,
        lambd: float = 0.05,
        mask_ratio: float = 0.25,
        ema_decay: float = 0.999,
        sigreg_slices: int = 256,
    ) -> None:
        super().__init__()
        self.model = model
        self.recipe = Recipe(recipe)
        self.student_layer = student_layer
        self.teacher_layer = teacher_layer if teacher_layer is not None else max(student_layer, model.n_layers - 1)
        self.gamma = gamma
        self.lambd = lambd
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.proj = ProjectionHead(model.dim)
        self.sigreg = SIGReg(num_slices=sigreg_slices)

        if self.recipe is Recipe.SELF_FLOW:
            teacher = deepcopy(model)
            teacher.requires_grad_(False)
            teacher.eval()
            self.teacher = teacher
        else:
            self.teacher = None

        last = model.n_layers - 1
        if not 0 <= self.student_layer <= last:
            raise ValueError(f"student_layer {self.student_layer} out of range [0, {last}]")
        if not 0 <= self.teacher_layer <= last:
            raise ValueError(f"teacher_layer {self.teacher_layer} out of range [0, {last}]")

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def _times(self, batch: int, device: torch.device) -> DualTimestep:
        n = self.model.n_tokens
        if self.recipe is Recipe.VANILLA:
            tau = torch.rand(batch, 1, device=device).expand(batch, n)
            return DualTimestep(tau=tau, tau_min=tau, mask=torch.zeros(batch, n, dtype=torch.bool, device=device))
        return sample_dual_timestep(batch, n, mask_ratio=self.mask_ratio, device=device)

    def _needed_layers(self) -> tuple[int, ...]:
        if self.recipe is Recipe.VANILLA:
            return ()
        if self.recipe is Recipe.SIGREG:
            return (self.student_layer,)
        return (self.student_layer, self.teacher_layer)

    def _forward(self, model: MiniSiT, x: Tensor, t: Tensor, y: Tensor) -> SiTOutput:
        return model(x, t, y, return_layers=self._needed_layers())

    def _sigreg_on_tokens(self, tokens: Tensor) -> Tensor:
        return self.sigreg(tokens.reshape(-1, tokens.shape[-1]))

    def training_step(self, images: Tensor, y: Tensor) -> StepLosses:
        x0 = self.model.patchify(images)
        x1 = torch.randn_like(x0)
        times = self._times(images.shape[0], images.device)
        x_tau = interpolate(x0, x1, times.tau)
        target = x1 - x0

        student = self._forward(self.model, x_tau, times.tau, y)
        loss_gen = F.mse_loss(student.velocity, target)
        loss_rep = images.new_zeros(())
        loss_sigreg = images.new_zeros(())

        if self.recipe is Recipe.SELF_FLOW:
            assert self.teacher is not None
            x_clean = interpolate(x0, x1, times.tau_min)
            with torch.no_grad():
                teacher_out = self._forward(self.teacher, x_clean, times.tau_min, y)
            loss_rep = cosine_align(
                self.proj(student.features[self.student_layer]),
                teacher_out.features[self.teacher_layer],
            )
        elif self.recipe is Recipe.SIGREG:
            loss_sigreg = self._sigreg_on_tokens(student.features[self.student_layer])
        elif self.recipe is Recipe.LEJEPA_FLOW:
            x_clean = interpolate(x0, x1, times.tau_min)
            clean = self._forward(self.model, x_clean, times.tau_min, y)
            loss_rep = F.mse_loss(self.proj(student.features[self.student_layer]), clean.features[self.teacher_layer])
            loss_sigreg = 0.5 * (
                self._sigreg_on_tokens(student.features[self.student_layer])
                + self._sigreg_on_tokens(clean.features[self.teacher_layer])
            )

        total = loss_gen + self.gamma * loss_rep + self.lambd * loss_sigreg
        return StepLosses(total=total, gen=loss_gen, rep=loss_rep, sigreg=loss_sigreg)

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        if self.teacher is not None:
            ema_update(self.teacher, self.model, self.ema_decay)


@torch.no_grad()
def euler_sample(
    model: MiniSiT,
    y: Tensor,
    *,
    steps: int = 25,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Homogeneous-time Euler integration from noise (t=1) to data (t=0)."""
    model.eval()
    b = y.shape[0]
    x = torch.randn(b, model.n_tokens, model.patch_dim, device=y.device, generator=generator)
    dt = 1.0 / steps
    for i in range(steps, 0, -1):
        t = torch.full((b, model.n_tokens), i / steps, device=y.device)
        velocity = model(x, t, y).velocity
        x = x - dt * velocity
    return model.unpatchify(x).clamp(-1, 1)
