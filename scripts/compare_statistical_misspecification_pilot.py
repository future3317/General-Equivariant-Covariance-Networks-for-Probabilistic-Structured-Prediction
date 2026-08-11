"""Summarize the first gated statistical-misspecification pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

LEVELS = tuple(0.1 * index for index in range(1, 10)) + (0.95,)


def _mahalanobis2(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor):
    residual = target - pred
    solved = torch.linalg.solve(scale, residual.unsqueeze(-1)).squeeze(-1)
    return (residual * solved).sum(-1).double().numpy()


def _single_calibration(prediction: dict[str, torch.Tensor]) -> dict:
    pred = prediction["mean"].double()
    target = prediction["target"].double()
    scale = prediction["scale"].double()
    nu = torch.as_tensor(prediction.get("nu", 5.0), dtype=torch.float64)
    if nu.ndim == 0:
        nu = nu.expand(pred.shape[0])
    values = _mahalanobis2(pred, target, scale)
    dimension = pred.shape[-1]
    observed = []
    for level in LEVELS:
        threshold = dimension * stats.f.ppf(level, dimension, nu.numpy())
        observed.append(float(np.mean(values < threshold)))
    observed_array = np.asarray(observed)
    return {
        "coverage90": observed[8],
        "coverage95": float(
            np.mean(
                values
                < dimension * stats.f.ppf(0.95, dimension, nu.numpy())
            )
        ),
        "mace": float(np.mean(np.abs(observed_array - np.asarray(LEVELS)))),
        "coverage_levels": list(LEVELS),
        "observed_coverages": observed,
    }


def _mixture_marginal_quantile(
    means: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    nu: np.ndarray,
    probability: float,
) -> np.ndarray:
    """Compute exact component-mixture marginal quantiles by bisection."""
    lower = np.min(means - 100.0 * scales, axis=0)
    upper = np.max(means + 100.0 * scales, axis=0)

    def cdf(value: np.ndarray) -> np.ndarray:
        standardized = (value[None, :] - means) / scales
        return np.sum(weights * stats.t.cdf(standardized, df=nu), axis=0)

    lo = lower.copy()
    hi = upper.copy()
    for _ in range(80):
        mid = (lo + hi) / 2.0
        below = cdf(mid) < probability
        lo[below] = mid[below]
        hi[~below] = mid[~below]
    return (lo + hi) / 2.0


def _mixture_calibration(prediction: dict[str, torch.Tensor]) -> dict:
    means = prediction["component_means"].double().numpy()
    scales = prediction["component_scales"].double().numpy()
    weights = prediction["weights"].double().numpy()
    nu = prediction["nu"].double().numpy()
    target = prediction["target"].double().numpy()
    component_count, sample_count, dimension = means.shape
    observed = []
    for level in LEVELS:
        covered = np.zeros((sample_count, dimension), dtype=bool)
        for coordinate in range(dimension):
            marginal_means = means[:, :, coordinate]
            marginal_scales = np.sqrt(scales[:, :, coordinate, coordinate])
            marginal_nu = nu
            lower = _mixture_marginal_quantile(
                marginal_means,
                marginal_scales,
                weights,
                marginal_nu,
                (1.0 - level) / 2.0,
            )
            upper = _mixture_marginal_quantile(
                marginal_means,
                marginal_scales,
                weights,
                marginal_nu,
                (1.0 + level) / 2.0,
            )
            covered[:, coordinate] = (target[:, coordinate] >= lower) & (
                target[:, coordinate] <= upper
            )
        observed.append(float(covered.mean()))
    observed_array = np.asarray(observed)
    return {
        "marginal_coverage90": observed[8],
        "marginal_coverage95": observed[9],
        "marginal_mace": float(
            np.mean(np.abs(observed_array - np.asarray(LEVELS)))
        ),
        "coverage_levels": list(LEVELS),
        "observed_coverages": observed,
        "coverage_semantics": "exact_component_mixture_marginal_intervals",
        "joint_elliptical_coverage": None,
        "components": component_count,
    }


def summarize_run(run_dir: Path) -> dict:
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text())
    prediction = torch.load(
        run_dir / "predictions_test.pt", map_location="cpu", weights_only=True
    )
    if "component_means" in prediction:
        calibration = _mixture_calibration(prediction)
        semantics = diagnostics["test"]["nll_semantics"]
    else:
        calibration = _single_calibration(prediction)
        semantics = diagnostics["test"]["nll_semantics"]
    elliptical = diagnostics["test"].get("elliptical_falsification", {})
    return {
        "run": run_dir.name,
        "nll": diagnostics["test"]["nll"],
        "nll_semantics": semantics,
        "energy_score": diagnostics["test"]["energy_score"],
        "calibration": calibration,
        "radial_pit": elliptical.get("radial_pit"),
        "whitened_second_moment_defect": elliptical.get(
            "whitened_second_moment_defect"
        ),
        "radius_direction_dependence": elliptical.get(
            "radius_direction_dependence"
        ),
        "mixture_projection_pit": diagnostics["test"].get("mixture_projection_pit"),
        "degrees_of_freedom": diagnostics["test"].get("degrees_of_freedom")
        or elliptical.get("student_t_dof"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        summarize_run(path)
        for path in sorted(args.root.iterdir())
        if path.is_dir() and (path / "diagnostics.json").is_file()
    ]
    output = {
        "kind": "statistical_misspecification_repair_pilot",
        "root": str(args.root.resolve()),
        "runs": records,
        "itop": {
            "status": "blocked",
            "reason": "current workspace exposes frozen Full H,mu cache but no raw geometry/depth cache or Graph frozen cache",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
