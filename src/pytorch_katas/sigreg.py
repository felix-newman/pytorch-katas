import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (LeJEPA / Epps–Pulley)."""

    def __init__(self, num_slices=1024, knots=17, t_max=3.0):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        phi = (-t.square() / 2).exp()
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w * phi)

    def forward(self, x):
        # x: (..., N, D)
        A = torch.randn(x.size(-1), self.num_slices, device=x.device, dtype=x.dtype)
        A = A / A.norm(2, 0, keepdim=True)
        xt = (x @ A).unsqueeze(-1) * self.t
        err = (xt.cos().mean(-3) - self.phi).square() + xt.sin().mean(-3).square()
        return (err @ self.weights).mul(x.size(-2)).mean()
