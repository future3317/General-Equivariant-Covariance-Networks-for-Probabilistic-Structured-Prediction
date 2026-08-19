"""Audit existing ITOP artifacts for reviewer-facing paired evidence.

This module does not train models or implement a new predictive density.  It
reuses the saved prediction sufficient statistics and the production
Student-t log-probability helper to compute paired per-frame NLL contrasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from distributions.student_t import student_t_log_prob_from_statistics


SIX_HEADS = (
    ("frozen_full_student_t", "Full-t"),
    ("frozen_independent_gaussian", "Indep-G"),
    ("frozen_independent_student_t", "Indep-t"),
    ("frozen_low_rank_student_t", "LR-t"),
    ("frozen_graph_gaussian", "Graph-G"),
    ("frozen_graph_student_t", "Graph-t"),
)
FAMILIES = ("full_student_t", "low_rank_student_t", "graph_student_t")
SEEDS = (42, 43, 44)
FAMILY_TO_MODEL = {
    "full_student_t": "frozen_full_student_t",
    "low_rank_student_t": "frozen_low_rank_student_t",
    "graph_student_t": "frozen_graph_student_t",
}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction_pair(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Require identical frame order and targets for a paired comparison."""

    left_frame = _as_numpy(left["frame_index"])
    right_frame = _as_numpy(right["frame_index"])
    if left_frame.shape != right_frame.shape or not np.array_equal(left_frame, right_frame):
        raise ValueError("frame_index mismatch in paired prediction artifacts")
    left_target = _as_numpy(left["target"])
    right_target = _as_numpy(right["target"])
    if left_target.shape != right_target.shape or not np.array_equal(left_target, right_target):
        raise ValueError("target mismatch in paired prediction artifacts")
    if not np.isfinite(left_target).all() or not np.isfinite(right_target).all():
        raise ValueError("non-finite target in paired prediction artifacts")


def _frame_mpjpe_cm(prediction: dict[str, Any]) -> np.ndarray:
    mean = _as_numpy(prediction["mean"]).astype(np.float64)
    target = _as_numpy(prediction["target"]).astype(np.float64)
    if mean.shape != target.shape or mean.shape[-1] != 45:
        raise ValueError("ITOP prediction must have shape (N, 45)")
    return np.linalg.norm((mean - target).reshape(-1, 15, 3), axis=-1).mean(axis=-1) * 100.0


def _frame_nll(prediction: dict[str, Any], nu: float = 5.0) -> np.ndarray:
    """Return exact per-frame Student-t NLL from saved sufficient statistics."""

    if "scale" in prediction:
        scale = torch.as_tensor(prediction["scale"], dtype=torch.float64)
        logdet = torch.linalg.slogdet(scale)[1]
    else:
        # train_itop stores frame_uncertainty as log|Cov| for Student-t
        # diagnostics, while the proper law is parameterized by scatter S.
        covariance_logdet = torch.as_tensor(
            prediction["frame_uncertainty"], dtype=torch.float64
        )
        dimension = 45
        scatter_factor = nu / (nu - 2.0)
        logdet = covariance_logdet - dimension * math.log(scatter_factor)
    mahalanobis2 = torch.as_tensor(prediction["frame_mahalanobis2"], dtype=torch.float64)
    log_prob = student_t_log_prob_from_statistics(logdet, mahalanobis2, 45, nu)
    values = (-log_prob).detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("non-finite per-frame NLL")
    return values


def paired_difference(
    left: dict[str, Any], right: dict[str, Any], metric: str = "frame_mpjpe_cm"
) -> np.ndarray:
    """Return ``right - left`` per-frame differences after pair validation."""

    validate_prediction_pair(left, right)
    if metric == "frame_mpjpe_cm":
        return _frame_mpjpe_cm(right) - _frame_mpjpe_cm(left)
    if metric == "nll":
        return _frame_nll(right) - _frame_nll(left)
    raise ValueError(f"unsupported paired metric: {metric}")


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of paired differences."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite, non-empty vector")
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap samples or confidence")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def _load_prediction(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"prediction artifact is not a dictionary: {path}")
    return payload


def _metric_summary(run: Path, label: str) -> dict[str, Any]:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    rows: dict[str, Any] = {"model": label, "run_dir": str(run)}
    for view in ("side", "top"):
        payload = metrics[view]
        coverage = payload.get("per_joint_marginal_coverage", {})
        rows[f"{view}_nll"] = payload.get("nll")
        rows[f"{view}_energy_score"] = payload.get("energy_score_m")
        rows[f"{view}_mace"] = payload.get("mace")
        rows[f"{view}_coverage_90"] = _coverage_level(coverage, 0.9)
        rows[f"{view}_coverage_95"] = _coverage_level(coverage, 0.95)
        prediction = _load_prediction(run / f"predictions_{view}.pt")
        rows["active_coordinates"] = int(_as_numpy(prediction["params"]).shape[-1]) if "params" in prediction else None
        rows[f"{view}_samples"] = int(_as_numpy(prediction["target"]).shape[0])
    return rows


def _coverage_level(payload: dict[str, Any], level: float) -> float | None:
    levels = payload.get("levels", [])
    values = payload.get("coverage_by_level_and_joint", [])
    if not levels or not values:
        return None
    index = int(np.argmin(np.abs(np.asarray(levels, dtype=float) - level)))
    return float(np.asarray(values[index], dtype=float).mean())


def _complete(run: Path) -> bool:
    return all(
        (run / name).is_file()
        for name in ("metrics.json", "args.json", "environment.json")
    ) and all(
        (run / f"predictions_{view}.pt").is_file() for view in ("side", "top")
    )


def _single_seed_rows(factorial_root: Path, full_root: Path) -> list[dict[str, Any]]:
    rows = []
    references: dict[str, dict[str, Any]] = {}
    backbone_hashes: set[str] = set()
    for model, label in SIX_HEADS:
        if model == "frozen_full_student_t" or model == "frozen_graph_student_t":
            run = full_root / "seed_42" / model
        else:
            run = factorial_root / model
        if _complete(run):
            feature_cache = run / "feature_cache.json"
            if feature_cache.is_file():
                record = json.loads(feature_cache.read_text(encoding="utf-8"))
                backbone_hash = record.get("backbone_checkpoint_sha256")
                if backbone_hash:
                    backbone_hashes.add(str(backbone_hash))
            for view in ("side", "top"):
                prediction = _load_prediction(run / f"predictions_{view}.pt")
                if view in references:
                    validate_prediction_pair(references[view], prediction)
                else:
                    references[view] = prediction
            rows.append(_metric_summary(run, label))
    if len(backbone_hashes) > 1:
        raise ValueError(f"single-seed six-head rows do not share a backbone: {backbone_hashes}")
    return rows


def _family_run(
    family: str,
    seed: int,
    *,
    factorial_root: Path,
    full_root: Path,
    family_root: Path,
    graph_root: Path,
) -> Path:
    model = FAMILY_TO_MODEL[family]
    if family == "full_student_t" and seed == 42:
        return full_root / "seed_42" / model
    if family == "low_rank_student_t" and seed == 42:
        return factorial_root / model
    if family == "graph_student_t":
        return graph_root / f"seed_{seed}" / model
    return family_root / f"seed_{seed}" / model


def _bootstrap_contrast(
    left: dict[str, Any], right: dict[str, Any], *, seed: int, view: str, metric: str
) -> dict[str, Any]:
    left_view = _load_prediction(left["path"] / f"predictions_{view}.pt")
    right_view = _load_prediction(right["path"] / f"predictions_{view}.pt")
    values = paired_difference(left_view, right_view, metric=metric)
    return {
        "metric": metric,
        "view": view,
        "direction": "right_minus_left",
        "mean": float(values.mean()),
        "bootstrap_95": list(bootstrap_mean_interval(values, seed=seed)),
        "samples": int(values.size),
    }


def _three_seed_bootstrap(
    *, factorial_root: Path, full_root: Path, family_root: Path, graph_root: Path
) -> list[dict[str, Any]]:
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for family in FAMILIES:
        for seed in SEEDS:
            path = _family_run(
                family,
                seed,
                factorial_root=factorial_root,
                full_root=full_root,
                family_root=family_root,
                graph_root=graph_root,
            )
            if not _complete(path):
                raise FileNotFoundError(f"incomplete paired artifact: {path}")
            runs[(family, seed)] = {"path": path}
    contrasts = []
    for right, left in (("graph_student_t", "full_student_t"), ("graph_student_t", "low_rank_student_t"), ("low_rank_student_t", "full_student_t")):
        for view in ("side", "top"):
            for metric in ("nll", "frame_mpjpe_cm"):
                per_seed = [
                    _bootstrap_contrast(
                        runs[(left, seed)],
                        runs[(right, seed)],
                        seed=1000 + seed,
                        view=view,
                        metric=metric,
                    )
                    for seed in SEEDS
                ]
                all_values = []
                for seed in SEEDS:
                    left_prediction = _load_prediction(runs[(left, seed)]["path"] / f"predictions_{view}.pt")
                    right_prediction = _load_prediction(runs[(right, seed)]["path"] / f"predictions_{view}.pt")
                    all_values.append(paired_difference(left_prediction, right_prediction, metric=metric))
                pooled = np.concatenate(all_values)
                contrasts.append(
                    {
                        "left": left,
                        "right": right,
                        "view": view,
                        "metric": metric,
                        "direction": "right_minus_left",
                        "per_seed": per_seed,
                        "pooled_mean": float(pooled.mean()),
                        "pooled_bootstrap_95": list(
                            bootstrap_mean_interval(pooled, seed=20260812, samples=4000)
                        ),
                    }
                )
    return contrasts


def audit(args: argparse.Namespace) -> dict[str, Any]:
    factorial_root = Path(args.itop_factorial_root)
    full_root = Path(args.itop_full_root)
    family_root = Path(args.itop_family_robustness_root)
    graph_root = Path(args.itop_graph_robustness_root)
    result = {
        "schema_version": 1,
        "kind": "review_evidence_completion_itop",
        "single_seed_six_head": _single_seed_rows(factorial_root, full_root),
        "three_seed_paired_bootstrap": _three_seed_bootstrap(
            factorial_root=factorial_root,
            full_root=full_root,
            family_root=family_root,
            graph_root=graph_root,
        ),
        "provenance": {
            "factorial_root": str(factorial_root),
            "full_root": str(full_root),
            "family_robustness_root": str(family_root),
            "graph_robustness_root": str(graph_root),
            "bootstrap_samples_pooled": 4000,
            "selection": "existing validation-only checkpoints; no test selection",
        },
        "evidence_status": {
            "existing_evidence": [
                "six-head ITOP factorial artifacts are complete for one seed",
                "Full/LR/Graph Student-t artifacts are complete for seeds 42/43/44",
            ],
            "new_evidence": [
                "paired per-frame NLL and MPJPE bootstrap contrasts from saved predictions"
            ],
            "supported_inference": [
                "family contrasts are evaluated on matched frame IDs within each seed"
            ],
            "unresolved": [
                "six-head three-seed factorial was not run and is not inferred"
            ],
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itop-factorial-root", required=True)
    parser.add_argument("--itop-full-root", required=True)
    parser.add_argument("--itop-family-robustness-root", required=True)
    parser.add_argument("--itop-graph-robustness-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
