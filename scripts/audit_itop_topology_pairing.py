"""Audit matched true-vs-shuffled ITOP topology artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.review_evidence_audit import (
    _as_numpy,
    _coverage_level,
    _frame_nll,
    bootstrap_mean_interval,
    validate_prediction_pair,
)

SEEDS = (42, 43, 44)
REQUIRED = (
    "args.json",
    "environment.json",
    "compilation.json",
    "history.json",
    "metrics.json",
    "predictions_side.pt",
    "predictions_top.pt",
    "best_model.pt",
    "train.log",
    "feature_cache.json",
)
PROTOCOL_FIELDS = (
    "backbone_checkpoint",
    "feature_cache",
    "backbone_precision",
    "batch_size",
    "compile_tp",
    "cueq_method",
    "data_dir",
    "hidden_dim",
    "lmax",
    "lr",
    "max_radius",
    "num_basis",
    "num_layers",
    "num_neighbors",
    "num_points",
    "num_workers",
    "patience",
    "phase",
    "prefetch_factor",
    "representation_metric",
    "student_t_dof",
    "tp_backend",
    "train_cache_sample_limit",
    "weight_decay",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _load_prediction(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected prediction dictionary: {path}")
    return payload


def _subject_ids(label_path: Path, frame_index: np.ndarray) -> np.ndarray:
    labels = np.load(label_path, allow_pickle=False)
    ids = np.asarray(labels["id"]).astype("U")
    indices = np.asarray(frame_index, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("frame_index must be a non-empty one-dimensional array")
    if np.any(indices < 0) or np.any(indices >= ids.size):
        raise ValueError("prediction frame_index is outside the label table")
    sample_ids = ids[indices]
    subjects = np.asarray([value.split("_", 1)[0] for value in sample_ids])
    if np.any(subjects == ""):
        raise ValueError("label IDs do not contain a subject prefix")
    return subjects


def cluster_bootstrap_mean_interval(
    values_by_seed: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    samples: int = 4000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a frame mean by resampling complete subject clusters.

    Repeated seeds are averaged within frame before the cluster resampling, so
    the bootstrap unit is the subject rather than an individual frame.
    """

    values = np.asarray(values_by_seed, dtype=np.float64)
    clusters = np.asarray(cluster_ids)
    if values.ndim != 2 or values.shape[1] != clusters.size or values.shape[1] == 0:
        raise ValueError("cluster bootstrap input must be (seeds, frames)")
    if not np.isfinite(values).all():
        raise ValueError("cluster bootstrap input is non-finite")
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    if unique_clusters.size < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    frame_values = values.mean(axis=0)
    cluster_sums = np.bincount(inverse, weights=frame_values)
    cluster_sizes = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, unique_clusters.size, size=(samples, unique_clusters.size))
    means = cluster_sums[draws].sum(axis=1) / cluster_sizes[draws].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def _parse_runs(values: list[str]) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed not in SEEDS or seed in runs:
            raise ValueError(f"invalid or duplicate seed in run: {value}")
        path = Path(path_text)
        if not path.is_dir():
            raise FileNotFoundError(path)
        runs[seed] = path
    if tuple(sorted(runs)) != SEEDS:
        raise ValueError(f"expected one run for each seed {SEEDS}, got {sorted(runs)}")
    return runs


def _complete(path: Path) -> None:
    missing = [name for name in REQUIRED if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing {missing}")


def _run_record(path: Path, expected_seed: int) -> dict[str, Any]:
    _complete(path)
    args = _read_json(path / "args.json")
    effective_split_seed = int(args.get("split_seed", args.get("seed", -1)))
    if int(args.get("seed", -1)) != expected_seed:
        raise ValueError(f"{path}: model seed is not {expected_seed}")
    if args.get("phase") != "frozen_head":
        raise ValueError(f"{path}: topology pairing requires frozen_head")
    cache = _read_json(path / "feature_cache.json")
    predictions = {
        view: _load_prediction(path / f"predictions_{view}.pt")
        for view in ("side", "top")
    }
    finite = all(
        bool(torch.isfinite(value).all())
        for prediction in predictions.values()
        for value in prediction.values()
        if isinstance(value, torch.Tensor)
    )
    if not finite:
        raise FloatingPointError(f"{path}: prediction artifact is non-finite")
    return {
        "path": str(path),
        "args": args,
        "effective_split_seed": effective_split_seed,
        "cache": cache,
        "predictions": predictions,
        "finite": finite,
    }


def _protocol_comparison(true: dict[str, Any], shuffled: dict[str, Any]) -> dict[str, Any]:
    true_args = true["args"]
    shuffled_args = shuffled["args"]
    mismatches = {
        field: {"true": true_args.get(field), "shuffled": shuffled_args.get(field)}
        for field in PROTOCOL_FIELDS
        if true_args.get(field) != shuffled_args.get(field)
    }
    true_cache = true["cache"]
    shuffled_cache = shuffled["cache"]
    cache_fields = (
        "backbone_checkpoint_sha256",
        "backbone_irreps",
        "feature_dimension",
        "backbone_precision",
        "num_points",
        "num_neighbors",
        "train_cache_sample_limit",
        "counts",
    )
    cache_mismatches = {
        field: {"true": true_cache.get(field), "shuffled": shuffled_cache.get(field)}
        for field in cache_fields
        if true_cache.get(field) != shuffled_cache.get(field)
    }
    return {
        "protocol_fields_match": not mismatches,
        "protocol_mismatches": mismatches,
        "cache_fields_match": not cache_mismatches,
        "cache_mismatches": cache_mismatches,
        "effective_split_seed_match": (
            true["effective_split_seed"] == shuffled["effective_split_seed"]
        ),
        "true_effective_split_seed": true["effective_split_seed"],
        "shuffled_effective_split_seed": shuffled["effective_split_seed"],
    }


def _metric_summary(run: dict[str, Any], view: str) -> dict[str, float | int | None]:
    metrics = _read_json(Path(run["path"]) / "metrics.json")[view]
    coverage = metrics.get("per_joint_marginal_coverage", {})
    prediction = run["predictions"][view]
    return {
        "nll": float(metrics["nll"]),
        "energy_score": float(metrics["energy_score_m"]),
        "joint_mace": float(metrics["joint_mace"]),
        "coverage90": _coverage_level(coverage, 0.9),
        "coverage95": _coverage_level(coverage, 0.95),
        "risk_coverage_auc_cm": float(metrics["frame_risk_coverage_auc_cm"]),
        "samples": int(_as_numpy(prediction["target"]).shape[0]),
    }


def _paired_nll(
    true: dict[str, Any],
    shuffled: dict[str, Any],
    view: str,
    seed: int,
    subject_ids: np.ndarray,
) -> dict[str, Any]:
    true_prediction = true["predictions"][view]
    shuffled_prediction = shuffled["predictions"][view]
    validate_prediction_pair(true_prediction, shuffled_prediction)
    values = _frame_nll(shuffled_prediction) - _frame_nll(true_prediction)
    return {
        "direction": "shuffled_minus_true",
        "true_frame_nll_mean": float(_frame_nll(true_prediction).mean()),
        "shuffled_frame_nll_mean": float(_frame_nll(shuffled_prediction).mean()),
        "mean": float(values.mean()),
        "bootstrap_95": list(bootstrap_mean_interval(values, seed=seed, samples=4000)),
        "subject_cluster_bootstrap_95": list(
            cluster_bootstrap_mean_interval(
                values[None, :], subject_ids, seed=seed, samples=4000
            )
        ),
        "samples": int(values.size),
    }


def audit(
    true_runs: dict[int, Path],
    shuffled_runs: dict[int, Path],
    labels_path: Path,
) -> dict[str, Any]:
    records = []
    pooled: dict[str, list[np.ndarray]] = {"side": [], "top": []}
    pooled_true: dict[str, list[np.ndarray]] = {"side": [], "top": []}
    pooled_shuffled: dict[str, list[np.ndarray]] = {"side": [], "top": []}
    subject_ids_by_seed: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        true = _run_record(true_runs[seed], seed)
        shuffled = _run_record(shuffled_runs[seed], seed)
        protocol = _protocol_comparison(true, shuffled)
        if not (
            protocol["protocol_fields_match"]
            and protocol["cache_fields_match"]
            and protocol["effective_split_seed_match"]
        ):
            raise ValueError(f"seed {seed}: true/shuffled artifacts are not matched: {protocol}")
        paired = {}
        for view in ("side", "top"):
            true_prediction = true["predictions"][view]
            shuffled_prediction = shuffled["predictions"][view]
            validate_prediction_pair(true_prediction, shuffled_prediction)
            values = _frame_nll(shuffled_prediction) - _frame_nll(true_prediction)
            subject_ids = _subject_ids(labels_path, true_prediction["frame_index"].numpy())
            subject_ids_by_seed[seed] = subject_ids
            pooled[view].append(values)
            pooled_true[view].append(_frame_nll(true_prediction))
            pooled_shuffled[view].append(_frame_nll(shuffled_prediction))
            paired[view] = _paired_nll(
                true,
                shuffled,
                view,
                seed=20260816 + seed,
                subject_ids=subject_ids,
            )
        records.append(
            {
                "seed": seed,
                "true_run": true["path"],
                "shuffled_run": shuffled["path"],
                "protocol": protocol,
                "true_metrics": {view: _metric_summary(true, view) for view in ("side", "top")},
                "shuffled_metrics": {
                    view: _metric_summary(shuffled, view) for view in ("side", "top")
                },
                "paired_nll": paired,
            }
        )
    pooled_result = {}
    for view in ("side", "top"):
        values = np.concatenate(pooled[view])
        subject_ids = np.concatenate([subject_ids_by_seed[seed] for seed in SEEDS])
        pooled_result[view] = {
            "direction": "shuffled_minus_true",
            "true_frame_nll_mean": float(np.concatenate(pooled_true[view]).mean()),
            "shuffled_frame_nll_mean": float(
                np.concatenate(pooled_shuffled[view]).mean()
            ),
            "mean": float(values.mean()),
            "bootstrap_95": list(
                bootstrap_mean_interval(values, seed=20260816, samples=4000)
            ),
            "samples": int(values.size),
            "subject_cluster_bootstrap_95": list(
                cluster_bootstrap_mean_interval(
                    np.stack(pooled[view]),
                    subject_ids_by_seed[SEEDS[0]],
                    seed=20260817,
                    samples=4000,
                )
            ),
            "cluster_unit": "subject prefix from official compact label ID",
            "cluster_count": int(np.unique(subject_ids).size),
            "action_sequence_metadata_available": False,
        }
    return {
        "schema_version": 1,
        "kind": "itop_matched_topology_pairing_audit",
        "seeds": list(SEEDS),
        "selection": "Side validation only; Top evaluation only; no test selection",
        "label_artifact": str(labels_path),
        "cluster_inference_note": (
            "Subject clusters are recovered from the official compact label ID prefix. "
            "The official ITOP schema defines IDs as XX_YYYYY person/frame identifiers "
            "and provides no action-sequence field; sequence boundaries are not inferred "
            "from frame-number gaps or validity masks."
        ),
        "paired_frames_per_view": 4863,
        "per_seed": records,
        "pooled_paired_nll": pooled_result,
        "conclusion": "true and shuffled Graph-t differ only in declared topology within each matched seed/split",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", action="append", required=True, metavar="SEED=RUN_DIR")
    parser.add_argument("--shuffled", action="append", required=True, metavar="SEED=RUN_DIR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", required=True, help="official compact test-label NPZ")
    args = parser.parse_args()
    result = audit(_parse_runs(args.true), _parse_runs(args.shuffled), Path(args.labels))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
