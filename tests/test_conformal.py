import math

import pytest
import torch

from evaluation.conformal import evaluate_region, fit_split_conformal


def test_split_conformal_uses_finite_sample_order_statistic():
    means = torch.zeros(9, 1)
    targets = torch.arange(1.0, 10.0).reshape(-1, 1)
    shapes = torch.eye(1).expand(9, 1, 1).clone()

    region = fit_split_conformal(means, shapes, targets, alpha=0.1)

    assert region.calibration_size == 9
    assert region.rank == 9
    assert region.threshold == pytest.approx(81.0)

    tiny = fit_split_conformal(
        means[:3], shapes[:3], targets[:3], alpha=0.1
    )
    assert tiny.rank == 4
    assert math.isinf(tiny.threshold)


def test_split_conformal_score_and_membership_are_orthogonal_invariant():
    generator = torch.Generator().manual_seed(12)
    n_cal, n_test, d = 32, 19, 3
    means_cal = torch.randn(n_cal, d, generator=generator)
    means_test = torch.randn(n_test, d, generator=generator)
    targets_cal = means_cal + torch.randn(n_cal, d, generator=generator)
    targets_test = means_test + torch.randn(n_test, d, generator=generator)
    diagonal_cal = 0.5 + torch.rand(n_cal, d, generator=generator)
    diagonal_test = 0.5 + torch.rand(n_test, d, generator=generator)
    shapes_cal = torch.diag_embed(diagonal_cal)
    shapes_test = torch.diag_embed(diagonal_test)
    rotation, _ = torch.linalg.qr(torch.randn(d, d, generator=generator))

    def transform(means, shapes, targets):
        return (
            means @ rotation.T,
            rotation @ shapes @ rotation.T,
            targets @ rotation.T,
        )

    region = fit_split_conformal(
        means_cal, shapes_cal, targets_cal, alpha=0.2
    )
    rotated = fit_split_conformal(
        *transform(means_cal, shapes_cal, targets_cal), alpha=0.2
    )
    assert rotated.threshold == pytest.approx(region.threshold, rel=1e-5, abs=1e-6)
    torch.testing.assert_close(
        region.contains(means_test, shapes_test, targets_test),
        rotated.contains(*transform(means_test, shapes_test, targets_test)),
    )


def test_region_summary_reports_shape_volume_not_density():
    means = torch.zeros(16, 2)
    targets = torch.randn(16, 2, generator=torch.Generator().manual_seed(4))
    shapes = torch.eye(2).expand(16, 2, 2).clone()
    region = fit_split_conformal(means, shapes, targets, alpha=0.25)
    summary = evaluate_region(region, means, shapes, targets)

    assert 0.0 <= summary["coverage"] <= 1.0
    assert summary["n"] == 16.0
    assert math.isfinite(summary["mean_log_volume"])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_split_conformal_rejects_invalid_alpha(alpha):
    means = torch.zeros(2, 1)
    shapes = torch.eye(1).expand(2, 1, 1).clone()
    targets = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="alpha"):
        fit_split_conformal(means, shapes, targets, alpha=alpha)
