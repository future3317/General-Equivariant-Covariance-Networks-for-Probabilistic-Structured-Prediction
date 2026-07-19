"""Tests for low-rank-plus-isotropic SPD map using Woodbury identity."""

import pytest
import torch

from spd_maps import LowRankPlusIsotropicMap


def _make_params(batch=4, dim=21, rank=4, seed=0):
    torch.manual_seed(seed)
    L = torch.randn(batch, dim, rank)
    log_sigma2 = torch.randn(batch)
    return torch.cat([L.reshape(batch, dim * rank), log_sigma2.unsqueeze(-1)], dim=-1)


def test_woodbury_logdet_matches_dense():
    """Woodbury logdet must match dense logdet."""
    params = _make_params(batch=8, dim=21, rank=4)
    mp = LowRankPlusIsotropicMap(dim=21, rank=4)

    logdet_woodbury = mp.logdet(params)
    S = mp.forward(params)
    logdet_dense = torch.logdet(S)

    assert torch.allclose(logdet_woodbury, logdet_dense, atol=1e-4, rtol=1e-4)


def test_woodbury_precision_action_matches_dense():
    """Woodbury precision action must match dense solve."""
    params = _make_params(batch=8, dim=21, rank=4)
    residual = torch.randn(8, 21)
    mp = LowRankPlusIsotropicMap(dim=21, rank=4)

    action_woodbury = mp.precision_action(params, residual)
    S = mp.forward(params)
    x = torch.linalg.solve(S, residual.unsqueeze(-1))
    action_dense = torch.sum(residual * x.squeeze(-1), dim=-1)

    assert torch.allclose(action_woodbury, action_dense, atol=1e-3, rtol=1e-3)


def test_woodbury_gradients_finite():
    """Gradients through Woodbury logdet and precision_action must be finite."""
    params = _make_params(batch=4, dim=21, rank=4).requires_grad_(True)
    residual = torch.randn(4, 21)
    mp = LowRankPlusIsotropicMap(dim=21, rank=4)

    loss = mp.logdet(params).sum() + mp.precision_action(params, residual).sum()
    loss.backward()

    assert params.grad is not None
    assert torch.isfinite(params.grad).all()
