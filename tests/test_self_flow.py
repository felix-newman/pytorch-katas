import torch

from pytorch_katas.metrics import collapse_metrics
from pytorch_katas.self_flow import Recipe, SelfFlowTrainer, euler_sample, sample_dual_timestep
from pytorch_katas.sit import MiniSiT


def _trainer(recipe: Recipe) -> SelfFlowTrainer:
    model = MiniSiT(image_size=8, patch_size=4, dim=32, n_heads=4, n_layers=3, n_classes=4)
    return SelfFlowTrainer(model, recipe, student_layer=0, teacher_layer=2, sigreg_slices=32, gamma=0.5, lambd=0.05)


def test_dual_timestep_shapes_and_mask_ratio() -> None:
    times = sample_dual_timestep(64, 16, mask_ratio=0.25)
    assert times.tau.shape == (64, 16)
    assert times.tau_min.shape == (64, 16)
    assert 0.15 < times.mask.float().mean().item() < 0.35
    assert torch.all(times.tau_min <= times.tau + 1e-6)
    # tau_min is min(t, s) on every token, so it is constant along the sequence.
    assert torch.allclose(times.tau_min, times.tau_min[:, :1].expand_as(times.tau_min))


def test_vanilla_has_no_teacher() -> None:
    trainer = _trainer(Recipe.VANILLA)
    assert trainer.teacher is None
    images = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 4, (4,))
    losses = trainer.training_step(images, y)
    losses.total.backward()
    assert losses.rep.item() == 0.0
    assert losses.sigreg.item() == 0.0


def test_self_flow_uses_ema_teacher() -> None:
    trainer = _trainer(Recipe.SELF_FLOW)
    assert trainer.teacher is not None
    for p in trainer.teacher.parameters():
        assert not p.requires_grad
    images = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 4, (4,))
    losses = trainer.training_step(images, y)
    losses.total.backward()
    trainer.after_optimizer_step()
    assert losses.rep.abs().item() > 0.0
    assert losses.sigreg.item() == 0.0
    assert not any(p.requires_grad for p in trainer.teacher.parameters())
    trainable = {id(p) for p in trainer.trainable_parameters()}
    assert {id(p) for p in trainer.teacher.parameters()}.isdisjoint(trainable)


def test_sigreg_recipe_has_no_teacher() -> None:
    trainer = _trainer(Recipe.SIGREG)
    assert trainer.teacher is None
    images = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 4, (4,))
    losses = trainer.training_step(images, y)
    losses.total.backward()
    assert losses.sigreg.item() > 0.0
    assert losses.rep.item() == 0.0
    assert trainer.teacher is None


def test_lejepa_flow_predicts_without_ema() -> None:
    trainer = _trainer(Recipe.LEJEPA_FLOW)
    assert trainer.teacher is None
    images = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 4, (4,))
    losses = trainer.training_step(images, y)
    losses.total.backward()
    assert losses.rep.item() > 0.0
    assert losses.sigreg.item() > 0.0


def test_euler_sample_shape() -> None:
    model = MiniSiT(image_size=8, patch_size=4, dim=32, n_heads=4, n_layers=2, n_classes=4)
    y = torch.zeros(2, dtype=torch.long)
    samples = euler_sample(model, y, steps=3)
    assert samples.shape == (2, 3, 8, 8)


def test_collapse_metrics_distinguish_gaussian_from_constant() -> None:
    gaussian = torch.randn(8, 16, 32)
    constant = torch.ones(8, 16, 32)
    g = collapse_metrics(gaussian)
    c = collapse_metrics(constant)
    assert g["std"] > c["std"]
    assert c["abs_cosine"] > g["abs_cosine"]
