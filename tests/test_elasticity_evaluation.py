from __future__ import annotations

import torch
import pytest

from scripts.evaluate_elasticity import evaluate_elasticity_predictions


def test_deterministic_evaluation_reports_point_metrics_only():
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mean = target + 1.0

    result = evaluate_elasticity_predictions(
        {"sample_id": torch.arange(2), "mean": mean, "target": target},
        arm="deterministic",
    )

    assert result["sample_count"] == 2
    assert result["finite"] is True
    assert result["mae_normalized"] == 1.0
    assert result["rmse_normalized"] == 1.0
    assert "nll" not in result


@pytest.mark.parametrize("arm", ["full_student_t", "full_asinh_exp_student_t"])
def test_student_t_evaluation_reuses_proper_metrics_and_falsification(arm):
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(64, 3, generator=generator, dtype=torch.float64)
    mean = torch.zeros_like(target)
    scale = torch.eye(3, dtype=torch.float64).expand(64, -1, -1).clone()

    result = evaluate_elasticity_predictions(
        {
            "sample_id": torch.arange(64),
            "mean": mean,
            "target": target,
            "scale": scale,
        },
        arm=arm,
        student_t_dof=5.0,
        energy_samples=16,
        diagnostic_directions=8,
        diagnostic_permutations=19,
        seed=11,
    )

    assert result["finite"] is True
    assert result["fp64_scatter"]["strict_spd"] is True
    assert result["nll"] > 0.0
    assert result["energy_score"] > 0.0
    assert 0.0 <= result["coverage"]["coverage_90"] <= 1.0
    assert result["elliptical"]["sample_count"] == 64


def test_probabilistic_evaluation_rejects_non_spd_scatter():
    target = torch.zeros(20, 2, dtype=torch.float64)
    scale = torch.eye(2, dtype=torch.float64).expand(20, -1, -1).clone()
    scale[0, 0, 0] = -1.0

    try:
        evaluate_elasticity_predictions(
            {
                "sample_id": torch.arange(20),
                "mean": torch.zeros_like(target),
                "target": target,
                "scale": scale,
            },
            arm="low_rank_student_t",
            diagnostic_directions=4,
            diagnostic_permutations=19,
        )
    except ValueError as error:
        assert "strictly SPD" in str(error)
    else:
        raise AssertionError("non-SPD scatter was accepted")
