import math

import torch

from pytorch_katas.sigreg import SIGReg, sigreg_loss


def test_sigreg_lower_on_gaussian_than_on_collapse() -> None:
    g = torch.Generator().manual_seed(0)
    gaussian = torch.randn(512, 32, generator=g)
    collapsed = torch.ones(512, 32)
    scaled = 0.01 * torch.randn(512, 32, generator=g)
    gen = torch.Generator().manual_seed(1)
    loss_g = sigreg_loss(gaussian, num_slices=64, generator=gen)
    gen.manual_seed(1)
    loss_c = sigreg_loss(collapsed, num_slices=64, generator=gen)
    gen.manual_seed(1)
    loss_s = sigreg_loss(scaled, num_slices=64, generator=gen)
    assert loss_g < loss_c
    assert loss_g < loss_s


def test_sigreg_module_matches_functional() -> None:
    z = torch.randn(128, 16)
    gen = torch.Generator().manual_seed(7)
    fn = sigreg_loss(z, num_slices=32, generator=gen)
    gen.manual_seed(7)
    module = SIGReg(num_slices=32)
    assert torch.allclose(fn, module(z, generator=gen), rtol=1e-4, atol=1e-5)


def test_sigreg_gradients_uncollapse_embeddings() -> None:
    # Exact collapse is a symmetry: identical vectors get identical gradients and
    # never spread. A low-variance cloud is the realistic case (different patches).
    torch.manual_seed(0)
    z = (0.05 * torch.randn(256, 8)).requires_grad_(True)
    opt = torch.optim.SGD([z], lr=1.0)
    start_std = z.detach().std().item()
    start_loss = sigreg_loss(z.detach(), num_slices=64).item()
    for _ in range(80):
        opt.zero_grad()
        sigreg_loss(z, num_slices=64).backward()
        opt.step()
    end_loss = sigreg_loss(z.detach(), num_slices=64).item()
    end_std = z.detach().std().item()
    assert end_loss < start_loss
    assert end_std > start_std
    assert math.isfinite(end_loss)
