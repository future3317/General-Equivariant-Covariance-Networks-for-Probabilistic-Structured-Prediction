"""Reusable evaluation for trained rank-4 elasticity prediction artifacts."""

from __future__ import annotations

from typing import Any

import torch

from distributions.student_t import student_t_log_prob_from_statistics
from evaluation import (
    calibration_error,
    elliptical_falsification,
    empirical_coverage,
    energy_score,
    mahalanobis_distance_squared,
    mean_absolute_error,
    root_mean_squared_error,
)


def _validated_prediction_tensors(
    predictions: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    required = ("sample_id", "mean", "target")
    missing = [name for name in required if name not in predictions]
    if missing:
        raise KeyError(f"prediction artifact is missing {missing}")
    mean = predictions["mean"].detach().double().cpu()
    target = predictions["target"].detach().double().cpu()
    if mean.shape != target.shape or mean.ndim != 2:
        raise ValueError("mean and target must have matching (samples, dimension) shapes")
    if predictions["sample_id"].numel() != mean.shape[0]:
        raise ValueError("sample IDs do not match prediction count")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(target).all()):
        raise ValueError("mean or target contains non-finite values")
    return mean, target


def evaluate_elasticity_predictions(
    predictions: dict[str, torch.Tensor],
    *,
    arm: str,
    student_t_dof: float = 5.0,
    energy_samples: int = 64,
    diagnostic_directions: int = 64,
    diagnostic_permutations: int = 199,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate one deterministic or Student-t elasticity prediction artifact."""

    mean, target = _validated_prediction_tensors(predictions)
    result: dict[str, Any] = {
        "arm": arm,
        "sample_count": int(mean.shape[0]),
        "dimension": int(mean.shape[1]),
        "finite": True,
        "mae_normalized": float(mean_absolute_error(mean, target)),
        "rmse_normalized": float(root_mean_squared_error(mean, target)),
    }
    if arm == "deterministic":
        return result
    if arm not in {"low_rank_student_t", "full_student_t"}:
        raise ValueError(f"unsupported elasticity arm: {arm}")
    if "scale" not in predictions:
        raise KeyError("probabilistic prediction artifact is missing scale")

    scale = predictions["scale"].detach().double().cpu()
    expected_shape = (mean.shape[0], mean.shape[1], mean.shape[1])
    if scale.shape != expected_shape or not bool(torch.isfinite(scale).all()):
        raise ValueError("scatter has invalid shape or non-finite values")
    symmetry_error = float((scale - scale.transpose(-1, -2)).abs().max())
    eigenvalues = torch.linalg.eigvalsh(0.5 * (scale + scale.transpose(-1, -2)))
    minimum_eigenvalue = float(eigenvalues.min())
    if minimum_eigenvalue <= 0.0:
        raise ValueError("scatter is not strictly SPD in FP64")

    residual = target - mean
    mahalanobis2 = mahalanobis_distance_squared(residual, scale)
    _, logdet = torch.linalg.slogdet(scale)
    log_prob = student_t_log_prob_from_statistics(
        logdet, mahalanobis2, mean.shape[-1], student_t_dof
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        score = energy_score(
            mean,
            scale,
            target,
            num_samples=energy_samples,
            distribution="student_t",
            student_t_dof=student_t_dof,
        )

    result.update(
        {
            "nll": float(-log_prob.mean()),
            "energy_score": float(score),
            "coverage": empirical_coverage(
                mean,
                target,
                scale,
                levels=[0.9, 0.95],
                reference="student_t",
                student_t_dof=student_t_dof,
            ),
            "calibration": calibration_error(
                mean,
                target,
                scale,
                reference="student_t",
                student_t_dof=student_t_dof,
            ),
            "elliptical": elliptical_falsification(
                mean,
                target,
                scale,
                reference="student_t",
                student_t_dof=student_t_dof,
                num_directions=diagnostic_directions,
                permutations=diagnostic_permutations,
                seed=seed,
            ),
            "fp64_scatter": {
                "strict_spd": True,
                "minimum_eigenvalue": minimum_eigenvalue,
                "maximum_eigenvalue": float(eigenvalues.max()),
                "maximum_symmetry_error": symmetry_error,
            },
        }
    )
    return result
