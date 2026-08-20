"""Audit a preregistered ITOP degree-matched topology-null campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.review_evidence_audit import (
    _coverage_level,
    _frame_nll,
    _load_prediction,
    validate_prediction_pair,
)

REQUIRED_FILES = (
    "args.json",
    "environment.json",
    "compilation.json",
    "history.json",
    "metrics.json",
    "predictions_side.pt",
    "predictions_top.pt",
    "best_model.pt",
    "train.log",
    "provenance.json",
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


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _edge_set(edges: list[list[int]]) -> set[tuple[int, int]]:
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}


def edge_overlap(reference: list[list[int]], candidate: list[list[int]]) -> int:
    """Return the number of shared undirected edges."""

    return len(_edge_set(reference) & _edge_set(candidate))


def _missing_required_files(
    run: Path, *, allow_missing_provenance: bool = False
) -> list[str]:
    missing = [name for name in REQUIRED_FILES if not (run / name).is_file()]
    if missing and (
        not allow_missing_provenance
        or any(name != "provenance.json" for name in missing)
    ):
        raise FileNotFoundError(f"{run}: missing {missing}")
    return missing


def _metric_row(run: Path, view: str) -> dict[str, float | None]:
    metrics = _json(run / "metrics.json")[view]
    coverage = metrics.get("per_joint_marginal_coverage", {})
    return {
        "nll": float(metrics["nll"]),
        "energy_score": float(metrics["energy_score_m"]),
        "joint_mace": float(metrics["joint_mace"]),
        "coverage90": _coverage_level(coverage, 0.9),
        "coverage95": _coverage_level(coverage, 0.95),
        "risk_coverage_auc_cm": float(metrics["frame_risk_coverage_auc_cm"]),
        "minimum_eigenvalue": float(
            metrics["scale_materialization"]["minimum_eigenvalue"]
        ),
    }


def _load_run(
    run: Path, *, topology_index: int | None, allow_missing_provenance: bool = False
) -> dict[str, Any]:
    missing = _missing_required_files(
        run, allow_missing_provenance=allow_missing_provenance
    )
    args = _json(run / "args.json")
    if topology_index is not None and int(args.get("topology_index", -1)) != topology_index:
        raise ValueError(f"{run}: topology index does not match manifest")
    if int(args.get("seed", -1)) != 42 or int(args.get("split_seed", -1)) != 42:
        raise ValueError(f"{run}: expected model and split seed 42")
    if args.get("phase") != "frozen_head":
        raise ValueError(f"{run}: expected frozen_head phase")
    predictions = {
        view: _load_prediction(run / f"predictions_{view}.pt")
        for view in ("side", "top")
    }
    finite = all(
        bool(torch.isfinite(value).all())
        for prediction in predictions.values()
        for value in prediction.values()
        if isinstance(value, torch.Tensor)
    )
    if not finite:
        raise FloatingPointError(f"{run}: prediction artifact is non-finite")
    metrics = {view: _metric_row(run, view) for view in ("side", "top")}
    if any(metrics[view]["minimum_eigenvalue"] <= 0 for view in ("side", "top")):
        raise FloatingPointError(f"{run}: FP64 strict-SPD gate is not positive")
    return {
        "path": str(run),
        "args": args,
        "predictions": predictions,
        "metrics": metrics,
        "missing_files": missing,
    }


def _protocol_mismatches(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        field: {"left": left.get(field), "right": right.get(field)}
        for field in PROTOCOL_FIELDS
        if left.get(field) != right.get(field)
    }


def _paired_row(true: dict[str, Any], null: dict[str, Any], view: str) -> dict[str, Any]:
    true_prediction = true["predictions"][view]
    null_prediction = null["predictions"][view]
    validate_prediction_pair(true_prediction, null_prediction)
    delta = _frame_nll(null_prediction) - _frame_nll(true_prediction)
    return {
        "true_nll_mean": float(_frame_nll(true_prediction).mean()),
        "null_nll_mean": float(_frame_nll(null_prediction).mean()),
        "null_minus_true_mean": float(delta.mean()),
        "samples": int(delta.size),
    }


def _summarize_effects(values: np.ndarray) -> dict[str, float | bool]:
    if values.size < 2:
        raise ValueError("at least two topology effects are required")
    quartiles = np.quantile(values, [0.25, 0.75])
    return {
        "null_minus_true_mean": float(values.mean()),
        "null_minus_true_sample_std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "iqr": float(quartiles[1] - quartiles[0]),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "all_positive": bool(np.all(values > 0)),
    }


def audit(root: Path, manifest_path: Path, true_run: Path) -> dict[str, Any]:
    manifest = _json(manifest_path)
    if manifest.get("outcome_filtered"):
        raise ValueError("topology manifest is outcome-filtered")
    records = manifest.get("records", [])
    if not records:
        raise ValueError("topology manifest has no records")
    true = _load_run(
        true_run, topology_index=None, allow_missing_provenance=True
    )
    results = []
    for record in records:
        index = int(record["index"])
        run = root / f"topology_{index:02d}" / "seed_42" / "shuffled_graph_student_t"
        null = _load_run(run, topology_index=index)
        mismatches = _protocol_mismatches(true["args"], null["args"])
        if mismatches:
            raise ValueError(f"topology {index}: protocol mismatch {mismatches}")
        paired = {view: _paired_row(true, null, view) for view in ("side", "top")}
        results.append(
            {
                "index": index,
                "topology_seed": record["topology_seed"],
                "true_edge_overlap": edge_overlap(
                    record["reference_edges"], record["edges"]
                ),
                "degree_sequence": record["degree_sequence"],
                "metrics": null["metrics"],
                "paired_nll": paired,
                "run_dir": str(run),
            }
        )
    summary = {}
    for view in ("side", "top"):
        values = np.asarray(
            [row["paired_nll"][view]["null_minus_true_mean"] for row in results],
            dtype=np.float64,
        )
        summary[view] = _summarize_effects(values)
    return {
        "schema_version": 1,
        "kind": "itop_degree_matched_topology_null_audit",
        "manifest": str(manifest_path),
        "outcome_filtered": False,
        "selection": "Side train/validation only; Top evaluation only",
        "model_seed": 42,
        "split_seed": 42,
        "true_run": str(true_run),
        "true_run_missing_files": true["missing_files"],
        "topology_count": len(results),
        "per_topology": results,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--true-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.manifest, args.true_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
