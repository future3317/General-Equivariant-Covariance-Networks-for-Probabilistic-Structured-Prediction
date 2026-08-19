"""Compute paired dielectric factorial contrasts from saved predictions.

The audit is read-only: it does not train, select checkpoints, or rewrite
prediction artifacts.  Bootstrap resampling is clustered by test structure so
that the three initialization seeds are averaged within each resampled
structure rather than treated as independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"prediction artifact is not a dictionary: {path}")
    required = {"sample_id", "target", "mean", "scale"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"{path} is missing prediction fields: {sorted(missing)}")
    return payload


def _as_float(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def _validate_pair(left: dict[str, Any], right: dict[str, Any]) -> None:
    left_ids = _as_float(left["sample_id"])
    right_ids = _as_float(right["sample_id"])
    if left_ids.shape != right_ids.shape or not torch.equal(left_ids, right_ids):
        raise ValueError("paired dielectric predictions have different sample IDs")
    left_target = _as_float(left["target"])
    right_target = _as_float(right["target"])
    if left_target.shape != right_target.shape or not torch.equal(left_target, right_target):
        raise ValueError("paired dielectric predictions have different targets")
    if not bool(torch.isfinite(left_target).all()):
        raise ValueError("dielectric targets contain non-finite values")


def _statistics(prediction: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    mean = _as_float(prediction["mean"])
    target = _as_float(prediction["target"])
    scale = _as_float(prediction["scale"])
    if mean.shape != target.shape or mean.ndim != 2 or mean.shape[1] != 6:
        raise ValueError("dielectric predictions must have shape (N, 6)")
    if scale.shape != (mean.shape[0], 6, 6):
        raise ValueError("dielectric scale must have shape (N, 6, 6)")
    sign, logdet = torch.linalg.slogdet(scale)
    if not bool(torch.all(sign > 0)):
        raise ValueError("dielectric scale is not strictly SPD")
    residual = target - mean
    solved = torch.linalg.solve(scale, residual.unsqueeze(-1)).squeeze(-1)
    mahalanobis2 = (residual * solved).sum(dim=-1)
    if not bool(torch.isfinite(logdet).all() and torch.isfinite(mahalanobis2).all()):
        raise ValueError("dielectric sufficient statistics are non-finite")
    return logdet, mahalanobis2


def _nll(prediction: dict[str, Any], law: str, nu: float = 5.0) -> np.ndarray:
    logdet, mahalanobis2 = _statistics(prediction)
    dimension = 6
    if law == "gaussian":
        values = 0.5 * (dimension * math.log(2.0 * math.pi) + logdet + mahalanobis2)
    elif law == "student_t":
        normalization = (
            torch.lgamma(
                torch.as_tensor((nu + dimension) / 2.0, dtype=torch.float64)
            )
            - torch.lgamma(torch.as_tensor(nu / 2.0, dtype=torch.float64))
            - 0.5 * dimension * math.log(nu * math.pi)
        )
        values = -normalization + 0.5 * logdet + 0.5 * (nu + dimension) * torch.log1p(
            mahalanobis2 / nu
        )
    else:
        raise ValueError(f"unsupported law: {law}")
    result = values.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("dielectric NLL is non-finite")
    return result


def paired_difference(
    left: dict[str, Any], right: dict[str, Any], *, left_law: str, right_law: str
) -> np.ndarray:
    """Return right-minus-left per-structure NLL after validating pairing."""

    _validate_pair(left, right)
    return _nll(right, right_law) - _nll(left, left_law)


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite and one-dimensional")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def cluster_bootstrap_mean_interval(
    values_by_seed: np.ndarray,
    *,
    seed: int,
    samples: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a mean after averaging repeated seeds within each structure."""

    values = np.asarray(values_by_seed, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("cluster bootstrap input must have shape (seeds, structures)")
    if not np.isfinite(values).all():
        raise ValueError("cluster bootstrap input is non-finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[1], size=(samples, values.shape[1]))
    # ``values[:, indices]`` has shape (seeds, bootstrap_draws, structures).
    # Average seeds and resampled structures while retaining one value per
    # bootstrap draw.
    draws = values[:, indices].mean(axis=(0, 2))
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha)))


def _contrast(
    *,
    factorial_root: Path,
    left_dir: str,
    right_dir: str,
    left_law: str,
    right_law: str,
    left_label: str,
    right_label: str,
    seeds: tuple[int, ...],
    bootstrap_seed: int,
    samples: int,
) -> dict[str, Any]:
    per_seed: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        left = _load(
            factorial_root
            / left_dir
            / f"seed_{seed}"
            / "predictions_test.pt"
        )
        right = _load(
            factorial_root
            / right_dir
            / f"seed_{seed}"
            / "predictions_test.pt"
        )
        values = paired_difference(
            left,
            right,
            left_law=left_law,
            right_law=right_law,
        )
        per_seed.append(values)
        rows.append(
            {
                "seed": seed,
                "mean": float(values.mean()),
                "bootstrap_95": list(
                    bootstrap_mean_interval(values, seed=bootstrap_seed + seed, samples=samples)
                ),
            }
        )
    matrix = np.stack(per_seed, axis=0)
    return {
        "left": left_label,
        "right": right_label,
        "direction": "right_minus_left",
        "per_seed": rows,
        "mean_over_seed_structure_pairs": float(matrix.mean()),
        "cluster_bootstrap_95": list(
            cluster_bootstrap_mean_interval(matrix, seed=bootstrap_seed, samples=samples)
        ),
        "seeds": list(seeds),
        "structures": int(matrix.shape[1]),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    seeds = tuple(args.seeds)
    factorial_root = Path(args.factorial_root)
    contrasts = [
        _contrast(
            factorial_root=factorial_root,
            left_dir="full/gaussian",
            right_dir="full/student_t",
            left_law="gaussian",
            right_law="student_t",
            left_label="full_gaussian",
            right_label="full_student_t",
            seeds=seeds,
            bootstrap_seed=args.bootstrap_seed,
            samples=args.samples,
        )
    ]
    isotropic = _contrast(
        factorial_root=factorial_root,
        left_dir="isotropic/student_t",
        right_dir="full/student_t",
        left_law="student_t",
        right_law="student_t",
        left_label="isotropic_student_t",
        right_label="full_student_t",
        seeds=seeds,
        bootstrap_seed=args.bootstrap_seed + 100,
        samples=args.samples,
    )
    return {
        "schema_version": 1,
        "kind": "dielectric_paired_factorial_bootstrap",
        "law_parameterization": "full normalized Gaussian/Student-t NLL; Student-t nu=5",
        "bootstrap": {
            "samples": args.samples,
            "confidence": 0.95,
            "unit": "test structure; repeated seeds averaged within structure",
            "seed": args.bootstrap_seed,
        },
        "contrasts": [contrasts[0], isotropic],
        "provenance": {
            "factorial_root": str(factorial_root),
            "selection": "existing validation-only checkpoints; no test selection",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    result = audit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
