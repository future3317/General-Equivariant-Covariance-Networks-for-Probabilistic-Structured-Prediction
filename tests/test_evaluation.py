"""Tests for evaluation metrics, calibration, and equivariance validators."""

import pytest
import torch

from evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    empirical_coverage,
    negative_log_likelihood_gaussian,
    energy_score,
    covariance_relative_error,
    log_euclidean_error,
    eigenvalue_error,
    whitened_residual_covariance,
    calibration_error,
    sharpness,
)


def _make_gaussian_data(batch=16, d=6, seed=0, well_specified: bool = False):
    torch.manual_seed(seed)
    mu = torch.randn(batch, d)
    noise = torch.randn(batch, d)
    target = mu + 0.5 * noise
    if well_specified:
        # S matches the true noise covariance (0.25 * I).
        S = 0.25 * torch.eye(d).unsqueeze(0).expand(batch, d, d)
    else:
        A = torch.randn(batch, d, d)
        S = torch.matmul(A, A.transpose(-1, -2)) + 0.1 * torch.eye(d)
    return mu, target, S


def test_mae_rmse():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    assert mean_absolute_error(pred, target).item() == pytest.approx(0.5)
    assert root_mean_squared_error(pred, target).item() == pytest.approx(0.5)


def test_r2_score():
    pred = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    score = r2_score(pred, target)
    assert torch.allclose(score, torch.ones(2))


def test_empirical_coverage():
    mu, target, S = _make_gaussian_data(batch=2000, d=6, well_specified=True)
    cov = empirical_coverage(mu, target, S)
    # For a well-specified Gaussian, coverage should be close to nominal levels.
    assert 0.45 < cov["coverage_50"] < 0.65
    assert 0.75 < cov["coverage_80"] < 0.90
    assert 0.85 < cov["coverage_90"] < 0.98


def test_negative_log_likelihood_gaussian_finite():
    mu, target, S = _make_gaussian_data()
    nll = negative_log_likelihood_gaussian(mu, target, S)
    assert torch.isfinite(nll)
    assert nll.item() > 0


def test_energy_score_finite():
    mu, target, S = _make_gaussian_data()
    score = energy_score(mu, S, target, num_samples=20)
    assert torch.isfinite(score)
    assert score.item() > 0


def test_covariance_relative_error_zero_when_equal():
    S = torch.randn(4, 6, 6)
    S = torch.matmul(S, S.transpose(-1, -2)) + 0.1 * torch.eye(6)
    err = covariance_relative_error(S, S)
    assert err.item() < 1e-5


def test_log_euclidean_error():
    A = torch.randn(4, 6, 6)
    A = 0.5 * (A + A.transpose(-1, -2))
    err = log_euclidean_error(A, A)
    assert err.item() < 1e-5


def test_eigenvalue_error_zero():
    S = torch.eye(6).unsqueeze(0).expand(4, 6, 6)
    err = eigenvalue_error(S, S)
    assert err.item() < 1e-5


def test_whitened_residual_covariance_identity_case():
    # If pred == target and S = I, whitened residuals are zero -> trace is zero.
    mu = torch.zeros(10, 6)
    target = torch.zeros(10, 6)
    S = torch.eye(6).unsqueeze(0).expand(10, 6, 6)
    trace = whitened_residual_covariance(mu, target, S)
    assert trace.item() < 1e-4


def test_whitened_residual_covariance_well_specified():
    # For identity scale and unit-Gaussian residuals, the trace should be close to d.
    torch.manual_seed(0)
    d = 6
    batch = 2000
    mu = torch.zeros(batch, d)
    target = torch.randn(batch, d)
    S = torch.eye(d).unsqueeze(0).expand(batch, d, d)
    trace = whitened_residual_covariance(mu, target, S)
    assert trace.item() == pytest.approx(float(d), rel=0.1)


def test_calibration_error():
    mu, target, S = _make_gaussian_data(batch=1000, d=6)
    cal = calibration_error(mu, target, S)
    assert "ece" in cal
    assert "ace" in cal
    assert 0 <= cal["ace"] <= 1


def test_sharpness():
    S = torch.eye(6).unsqueeze(0).expand(4, 6, 6)
    sh = sharpness(S)
    assert "mean_logdet" in sh
    assert "mean_trace" in sh
