"""Tests for distribution losses."""

import math
import pytest
import torch
from torch.distributions import MultivariateNormal

from spd_maps import MatrixExponentialMap
from distributions import GaussianNLL, StudentTNLL


def _make_test_tensors(batch=8, d=6, dtype=torch.float32):
    mu = torch.randn(batch, d, dtype=dtype)
    target = torch.randn(batch, d, dtype=dtype)
    A = torch.randn(batch, d, d, dtype=dtype)
    A = 0.5 * (A + A.transpose(-1, -2))
    A.requires_grad_(True)
    return mu, target, A


def test_gaussian_matches_pytorch():
    mu, target, A = _make_test_tensors()
    spdm = MatrixExponentialMap()
    loss, _ = GaussianNLL().forward(mu, A, target, spdm)

    S = spdm(A)
    mvn = MultivariateNormal(mu.double(), scale_tril=torch.linalg.cholesky(S.double()))
    ref = -mvn.log_prob(target.double()).mean()
    assert abs(loss.item() - ref.item()) < 1e-3


def test_gaussian_gradient_finite():
    mu, target, A = _make_test_tensors()
    loss, _ = GaussianNLL().forward(mu, A, target, MatrixExponentialMap())
    loss.backward()
    assert torch.isfinite(A.grad).all()


def test_student_t_gradient_finite():
    mu, target, A = _make_test_tensors()
    loss, _ = StudentTNLL(nu=5.0).forward(mu, A, target, MatrixExponentialMap())
    loss.backward()
    assert torch.isfinite(A.grad).all()


def test_student_t_approaches_gaussian():
    """Use float64 to avoid loss of precision in lgamma for large nu."""
    mu = torch.randn(8, 6, dtype=torch.float64)
    target = torch.randn(8, 6, dtype=torch.float64)
    A = torch.randn(8, 6, 6, dtype=torch.float64)
    A = 0.5 * (A + A.transpose(-1, -2))
    A.requires_grad_(True)
    spdm = MatrixExponentialMap()

    loss_g, _ = GaussianNLL().forward(mu, A, target, spdm)
    loss_t, _ = StudentTNLL(nu=1e6).forward(mu, A, target, spdm)
    assert abs(loss_t.item() - loss_g.item()) < 1e-2


def test_student_t_covariance_conversion():
    S = torch.eye(6) * 3.0
    S_cov = StudentTNLL.scale_to_covariance(S, nu=5.0)
    expected = (5.0 / 3.0) * S
    assert torch.allclose(S_cov, expected)
