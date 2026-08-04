"""Audit split-conformal coverage and orthogonal-coordinate invariance."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from evaluation.conformal import evaluate_region, fit_split_conformal


def _rotate(
    means: torch.Tensor,
    shapes: torch.Tensor,
    targets: torch.Tensor,
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotated_shapes = rotation @ shapes @ rotation.T
    rotated_shapes = 0.5 * (rotated_shapes + rotated_shapes.transpose(-1, -2))
    return (
        means @ rotation.T,
        rotated_shapes,
        targets @ rotation.T,
    )


def run_audit(
    *,
    seed: int = 42,
    calibration_size: int = 255,
    test_size: int = 10000,
    dimension: int = 3,
    alpha: float = 0.1,
    student_t_dof: float = 5.0,
) -> dict[str, object]:
    if calibration_size < 1 or test_size < 1 or dimension < 1:
        raise ValueError("sizes and dimension must be positive")
    if student_t_dof <= 0:
        raise ValueError("student_t_dof must be positive")
    torch.manual_seed(seed)
    means_cal = torch.randn(calibration_size, dimension) * 0.2
    means_test = torch.randn(test_size, dimension) * 0.2
    diagonal_cal = 0.5 + torch.rand(calibration_size, dimension)
    diagonal_test = 0.5 + torch.rand(test_size, dimension)
    shapes_cal = torch.diag_embed(diagonal_cal)
    shapes_test = torch.diag_embed(diagonal_test)
    student_t = torch.distributions.StudentT(student_t_dof)
    targets_cal = means_cal + torch.linalg.cholesky(shapes_cal).bmm(
        student_t.sample((calibration_size, dimension, 1))
    ).squeeze(-1)
    targets_test = means_test + torch.linalg.cholesky(shapes_test).bmm(
        student_t.sample((test_size, dimension, 1))
    ).squeeze(-1)

    region = fit_split_conformal(
        means_cal, shapes_cal, targets_cal, alpha=alpha
    )
    original = evaluate_region(region, means_test, shapes_test, targets_test)
    rotation, _ = torch.linalg.qr(torch.randn(dimension, dimension))
    rotated_region = fit_split_conformal(
        *_rotate(means_cal, shapes_cal, targets_cal, rotation), alpha=alpha
    )
    rotated = evaluate_region(
        rotated_region,
        *_rotate(means_test, shapes_test, targets_test, rotation),
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "audit": "split_conformal_student_t_shape",
        "source_commit": source_commit,
        "seed": seed,
        "dimension": dimension,
        "calibration_size": calibration_size,
        "test_size": test_size,
        "alpha": alpha,
        "student_t_dof": student_t_dof,
        "rank": region.rank,
        "threshold": region.threshold,
        "original": original,
        "rotated": rotated,
        "threshold_abs_difference": abs(
            region.threshold - rotated_region.threshold
        ),
        "coverage_abs_difference": abs(
            original["coverage"] - rotated["coverage"]
        ),
        "interpretation": (
            "Shape-based split conformal supplies marginal coverage only; "
            "the shape is not relabeled as a covariance or Student-t scale."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-size", type=int, default=255)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--student-t-dof", type=float, default=5.0)
    args = parser.parse_args()
    result = run_audit(
        seed=args.seed,
        calibration_size=args.calibration_size,
        test_size=args.test_size,
        dimension=args.dimension,
        alpha=args.alpha,
        student_t_dof=args.student_t_dof,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
