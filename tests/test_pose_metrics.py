"""Tests for structured pose uncertainty metrics."""

import torch

from evaluation.pose import (
    bone_length_error,
    joint_mahalanobis_squared,
    marginal_joint_covariances,
    mpjpe,
    occlusion_uncertainty_ratio,
    pck,
    risk_coverage_auc,
)


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
