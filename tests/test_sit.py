import torch

from pytorch_katas.sit import MiniSiT


def _tiny_sit() -> MiniSiT:
    return MiniSiT(image_size=8, patch_size=4, dim=32, n_heads=4, n_layers=3, n_classes=4)


def test_patchify_roundtrip() -> None:
    model = _tiny_sit()
    x = torch.randn(2, 3, 8, 8)
    tokens = model.patchify(x)
    assert tokens.shape == (2, 4, 48)
    assert torch.allclose(model.unpatchify(tokens), x, atol=1e-6)


def test_forward_per_token_timestep_and_features() -> None:
    model = _tiny_sit()
    x = torch.randn(3, 4, 48)
    t = torch.rand(3, 4)
    y = torch.tensor([0, 1, 2])
    out = model(x, t, y, return_layers=(0, 2))
    assert out.velocity.shape == x.shape
    assert set(out.features) == {0, 2}
    assert out.features[0].shape == (3, 4, 32)


def test_scalar_timestep_broadcasts() -> None:
    model = _tiny_sit()
    x = torch.randn(2, 4, 48)
    t = torch.rand(2)
    y = torch.zeros(2, dtype=torch.long)
    out = model(x, t, y)
    assert out.velocity.shape == x.shape
