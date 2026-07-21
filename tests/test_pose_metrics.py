"""Tests for structured pose uncertainty metrics."""

import torch

from evaluation.pose import (
    binary_auroc,
    bone_length_error,
    calibration_absolute_error,
    joint_mahalanobis_squared,
    joint_residual_correlation,
    marginal_joint_covariances,
    mpjpe,
    occlusion_uncertainty_ratio,
    pck,
    per_joint_marginal_coverage,
    residual_correlation_by_graph_distance,
    risk_coverage_auc,
)
from scripts.evaluate_itop_ensemble import _evaluate as evaluate_ensemble


def test_pose_accuracy_and_risk_coverage_metrics():
    target = torch.zeros(2, 45)
    prediction = target.clone()
    prediction[1] = 0.1
    assert torch.allclose(mpjpe(prediction, target), torch.tensor(0.08660254))
    assert pck(prediction, target, 0.05).item() == 0.5
    risk = torch.tensor([0.1, 0.9])
    error = torch.tensor([0.0, 1.0])
    assert torch.allclose(risk_coverage_auc(risk, error), torch.tensor(0.25))


def test_joint_marginals_and_mahalanobis():
    covariance = torch.eye(45).repeat(2, 1, 1)
    prediction = torch.zeros(2, 45)
    target = torch.ones(2, 45)
    marginal = marginal_joint_covariances(covariance)
    assert marginal.shape == (2, 15, 3, 3)
    assert torch.allclose(
        joint_mahalanobis_squared(prediction, target, covariance),
        torch.full((2, 15), 3.0),
    )


def test_occlusion_uncertainty_and_bone_consistency():
    covariance = torch.eye(45).unsqueeze(0)
    covariance[:, 3:6, 3:6] *= 4.0
    visible = torch.ones(1, 15, dtype=torch.bool)
    visible[:, 1] = False
    assert occlusion_uncertainty_ratio(covariance, visible) > 1.0

    target = torch.zeros(1, 45)
    target[:, 3] = 1.0
    samples = target.reshape(1, 15, 3).unsqueeze(1).repeat(1, 4, 1, 1)
    assert bone_length_error(samples, target, ((0, 1),)).item() == 0.0


def test_student_t_calibration_and_per_joint_coverage_are_finite():
    mahalanobis = torch.full((20, 15), 3.0)
    assert (
        0.0
        <= calibration_absolute_error(
            mahalanobis,
            3,
            student_t_dof=5.0,
        )
        <= 1.0
    )
    record = per_joint_marginal_coverage(
        mahalanobis,
        student_t_dof=5.0,
    )
    assert len(record["mace_by_joint"]) == 15
    assert len(record["coverage_by_level_and_joint"]) == 4


def test_residual_correlation_graph_distance_and_ood_auroc():
    prediction = torch.zeros(4, 9)
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ]
    )
    correlation = joint_residual_correlation(prediction, target)
    torch.testing.assert_close(correlation[0, 1], torch.tensor(1.0))
    torch.testing.assert_close(correlation[0, 2], torch.tensor(-1.0))
    grouped = residual_correlation_by_graph_distance(
        correlation,
        ((0, 1), (1, 2)),
    )
    assert set(grouped) == {"1", "2"}
    assert (
        binary_auroc(
            torch.tensor([0.1, 0.2, 0.8, 0.9]),
            torch.tensor([0, 0, 1, 1]),
        )
        == 1.0
    )


def test_deterministic_ensemble_is_not_mislabeled_as_continuous_density():
    target = torch.zeros(2, 45)
    visible = torch.ones(2, 15, dtype=torch.bool)
    records = [
        {
            "mean": target + offset,
            "target": target,
            "visible_joints": visible,
            "frame_index": torch.arange(2),
            "view_id": torch.zeros(2, dtype=torch.long),
        }
        for offset in (-0.1, 0.0, 0.1)
    ]
    metrics, artifact = evaluate_ensemble(records)
    assert metrics["members"] == 3
    assert metrics["nll"] is None
    assert metrics["mace"] is None
    assert artifact["members"].shape == (2, 3, 45)
