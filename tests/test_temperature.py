import math

import pytest
import torch

from evaluation.temperature import (
    apply_block_temperature,
    apply_temperature,
    fit_block_temperatures,
    fit_temperature,
    scale_nll,
)


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


def test_student_t_nll_uses_scale_parameterization_constant():
    pred = torch.zeros(1, 2)
    target = torch.zeros(1, 2)
    scale = torch.eye(2).unsqueeze(0)
    expected = -math.lgamma(3.5) + math.lgamma(2.5) + math.log(5.0 * math.pi)
    assert scale_nll(
        pred, target, scale, distribution="student_t", student_t_dof=5.0
    ).item() == pytest.approx(expected)


def test_block_temperature_preserves_spd_and_fits_positive_scales():
    pred, target, scale = _toy(d=3)
    block_ids = torch.tensor([0, 1, 1])
    calibrated = apply_block_temperature(scale, block_ids, [2.0, 0.5])
    assert torch.all(torch.linalg.eigvalsh(calibrated) > 0)
    temperatures = fit_block_temperatures(
        pred, target, scale, block_ids, steps=20
    )
    assert len(temperatures) == 2
    assert all(t > 0 for t in temperatures)
