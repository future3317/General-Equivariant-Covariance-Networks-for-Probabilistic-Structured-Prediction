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


def _paired_nll(true: dict[str, Any], shuffled: dict[str, Any], view: str, seed: int) -> dict[str, Any]:
    true_prediction = true["predictions"][view]
    shuffled_prediction = shuffled["predictions"][view]
    validate_prediction_pair(true_prediction, shuffled_prediction)
    values = _frame_nll(shuffled_prediction) - _frame_nll(true_prediction)
    return {
        "direction": "shuffled_minus_true",
        "mean": float(values.mean()),
        "bootstrap_95": list(bootstrap_mean_interval(values, seed=seed, samples=4000)),
        "samples": int(values.size),
    }


def audit(true_runs: dict[int, Path], shuffled_runs: dict[int, Path]) -> dict[str, Any]:
    records = []
    pooled: dict[str, list[np.ndarray]] = {"side": [], "top": []}
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
            pooled[view].append(values)
            paired[view] = _paired_nll(true, shuffled, view, seed=20260816 + seed)
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
        pooled_result[view] = {
            "direction": "shuffled_minus_true",
            "mean": float(values.mean()),
            "bootstrap_95": list(
                bootstrap_mean_interval(values, seed=20260816, samples=4000)
            ),
            "samples": int(values.size),
        }
    return {
        "schema_version": 1,
        "kind": "itop_matched_topology_pairing_audit",
        "seeds": list(SEEDS),
        "selection": "Side validation only; Top evaluation only; no test selection",
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
    args = parser.parse_args()
    result = audit(_parse_runs(args.true), _parse_runs(args.shuffled))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
