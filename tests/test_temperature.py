import torch

from evaluation.temperature import apply_temperature, fit_temperature, scale_nll


def _toy(n=128, d=3):
    g = torch.Generator().manual_seed(7)
    pred = torch.zeros(n, d)
    target = torch.randn(n, d, generator=g) * 2.0
    scale = torch.eye(d).expand(n, d, d).clone()
    return pred, target, scale


def test_temperature_fit_reduces_gaussian_nll():
    pred, target, scale = _toy()
    before = scale_nll(pred, target, scale)
    temperature = fit_temperature(pred, target, scale, steps=40)
    after = scale_nll(pred, target, apply_temperature(scale, temperature))
    assert temperature > 1.0
    assert after < before


def test_student_t_temperature_is_finite():
    pred, target, scale = _toy()
    temperature = fit_temperature(
        pred, target, scale, distribution="student_t", student_t_dof=5.0, steps=40
    )
    assert torch.isfinite(torch.tensor(temperature))
    assert scale_nll(
        pred, target, apply_temperature(scale, temperature),
        distribution="student_t", student_t_dof=5.0,
    ).isfinite()

