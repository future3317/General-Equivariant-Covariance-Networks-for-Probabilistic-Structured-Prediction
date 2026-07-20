"""Tests for low-rank-plus-isotropic SPD map using Woodbury identity."""

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


def test_joint_statistics_match_separate_values_and_gradients():
    """One-factorization statistics must preserve both outputs and gradients."""
    residual_old = torch.randn(4, 21, dtype=torch.float64, requires_grad=True)
    residual_new = residual_old.detach().clone().requires_grad_(True)
    params_old = _make_params(batch=4, dim=21, rank=4).double().requires_grad_(True)
    params_new = params_old.detach().clone().requires_grad_(True)
    spd_map = LowRankPlusIsotropicMap(dim=21, rank=4)

    old_logdet = spd_map.logdet(params_old)
    old_quadratic = spd_map.precision_action(params_old, residual_old)
    new_logdet, new_quadratic = spd_map.statistics(params_new, residual_new)

    torch.testing.assert_close(new_logdet, old_logdet, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(new_quadratic, old_quadratic, atol=1e-10, rtol=1e-10)

    (old_logdet + old_quadratic).sum().backward()
    (new_logdet + new_quadratic).sum().backward()
    torch.testing.assert_close(params_new.grad, params_old.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(
        residual_new.grad, residual_old.grad, atol=1e-10, rtol=1e-10
    )


def test_joint_statistics_build_factor_system_once():
    class CountingLowRankMap(LowRankPlusIsotropicMap):
        calls = 0

        def _factor_system(self, params):
            self.calls += 1
            return super()._factor_system(params)

    spd_map = CountingLowRankMap(dim=21, rank=4)
    params = _make_params(batch=2, dim=21, rank=4)
    residual = torch.randn(2, 21)
    spd_map.statistics(params, residual)
    assert spd_map.calls == 1
