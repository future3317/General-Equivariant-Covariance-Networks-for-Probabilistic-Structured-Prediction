"""Paired ITOP Full-t versus Graph-t proper-score and resource analysis."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.audit_itop_topology_pairing import (
    _subject_ids,
    cluster_bootstrap_mean_interval,
)
from scripts.itop_reproducibility import atomic_write_json


def paired_bootstrap_interval(
    differences: torch.Tensor, *, repetitions: int = 2000, seed: int = 42
) -> dict[str, float | int]:
    values = differences.detach().double().cpu().numpy().reshape(-1)
    if values.size < 2 or repetitions < 100 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires finite paired values and at least 100 repeats")
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = generator.integers(0, values.size, size=values.size)
        means[index] = values[sample].mean()
    lower, upper = np.quantile(means, (0.025, 0.975))
    return {
        "pairs": int(values.size),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean": float(values.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def aggregate_paired_differences(records: list[torch.Tensor]) -> torch.Tensor:
    if not records or len({tuple(record.shape) for record in records}) != 1:
        raise ValueError("seed-wise paired differences must share one sample shape")
    return torch.stack([record.detach().double().cpu() for record in records]).mean(0)


def subject_cluster_bootstrap_interval(
    differences: torch.Tensor,
    subject_ids: list[str] | np.ndarray,
    *,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    values = differences.detach().double().cpu().numpy().reshape(1, -1)
    interval = cluster_bootstrap_mean_interval(
        values,
        np.asarray(subject_ids),
        seed=seed,
        samples=repetitions,
    )
    return {
        "pairs": int(values.shape[1]),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean": float(values.mean()),
        "lower_95": interval[0],
        "upper_95": interval[1],
        "cluster_unit": "subject prefix from official compact label ID",
        "cluster_count": int(np.unique(subject_ids).size),
    }


def validate_paired_predictions(
    full: dict[str, torch.Tensor], graph: dict[str, torch.Tensor]
) -> None:
    for key in ("target", "frame_index"):
        if key not in full or key not in graph:
            raise KeyError(f"paired prediction artifact is missing {key}")
    if not torch.equal(full["frame_index"], graph["frame_index"]):
        raise ValueError("Full/Graph sample ordering differs")
    if not torch.equal(full["target"], graph["target"]):
        raise ValueError("Full/Graph targets differ")
    if "mean" in full and "mean" in graph and not torch.equal(full["mean"], graph["mean"]):
        raise ValueError("Full/Graph frozen means differ")


def pareto_frontier(
    rows: dict[str, dict[str, float]], *, x: str, y: str
) -> list[str]:
    frontier = []
    for name, row in rows.items():
        dominated = any(
            other[x] <= row[x]
            and other[y] <= row[y]
            and (other[x] < row[x] or other[y] < row[y])
            for other_name, other in rows.items()
            if other_name != name
        )
        if not dominated:
            frontier.append(name)
    return sorted(frontier)


def _load_predictions(run_dir: Path, view: str) -> dict[str, torch.Tensor]:
    prediction = torch.load(
        run_dir / f"predictions_{view}.pt", map_location="cpu", weights_only=True
    )
    required = ("mean", "target", "params", "frame_index")
    missing = [name for name in required if name not in prediction]
    if missing:
        raise KeyError(f"{run_dir}: predictions are missing {missing}")
    if prediction["mean"].shape != (4863, 45):
        raise ValueError(f"{run_dir}: expected 4,863 x 45 predictions")
    return prediction


def _per_sample_nll(run_dir: Path, prediction: dict[str, torch.Tensor]) -> torch.Tensor:
    from scripts.train_itop import _build_model

    training_args = Namespace(
        **json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    )
    model, _ = _build_model(training_args)
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, prediction["mean"].shape[0], 128):
            stop = start + 128
            log_prob, _ = model.distribution.log_prob(
                prediction["mean"][start:stop].float(),
                prediction["params"][start:stop].float(),
                prediction["target"][start:stop].float(),
                model.spd_map,
            )
            records.append(-log_prob.double().cpu())
    result = torch.cat(records)
    if result.shape != (4863,) or not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"{run_dir}: exact per-sample NLL audit failed")
    return result


def _parse_pair(value: str) -> tuple[int, Path, Path]:
    try:
        seed_text, full_text, graph_text = value.split("=", 2)
        return int(seed_text), Path(full_text), Path(graph_text)
    except ValueError as error:
        raise ValueError(f"invalid --pair specification: {value}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True, metavar="SEED=FULL=GRAPH")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()

    seed_records: dict[str, Any] = {}
    deltas_by_view: dict[str, list[torch.Tensor]] = {"side": [], "top": []}
    subjects_by_view: dict[str, np.ndarray] = {}
    for seed, full_dir, graph_dir in sorted(_parse_pair(value) for value in args.pair):
        views = {}
        for view in ("side", "top"):
            full_prediction = _load_predictions(full_dir, view)
            graph_prediction = _load_predictions(graph_dir, view)
            validate_paired_predictions(full_prediction, graph_prediction)
            full_nll = _per_sample_nll(full_dir, full_prediction)
            graph_nll = _per_sample_nll(graph_dir, graph_prediction)
            delta = graph_nll - full_nll
            deltas_by_view[view].append(delta)
            subjects_by_view[view] = _subject_ids(
                args.labels, full_prediction["frame_index"].numpy()
            )
            views[view] = {
                "full_mean_nll": float(full_nll.mean()),
                "graph_mean_nll": float(graph_nll.mean()),
                "graph_minus_full": paired_bootstrap_interval(
                    delta,
                    repetitions=args.bootstrap_repetitions,
                    seed=seed,
                ),
                "subject_cluster_bootstrap": subject_cluster_bootstrap_interval(
                    delta,
                    subjects_by_view[view],
                    repetitions=args.bootstrap_repetitions,
                    seed=seed,
                ),
            }
        seed_records[str(seed)] = views

    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))
    resource_rows = {}
    for family in ("full_student_t", "graph_student_t"):
        row = runtime["rows"][family]
        resource_rows[family] = {
            "coordinates": float(row["active_coordinates"]),
            "memory_mb": float(
                row["timings"]["forward_backward"]["peak_allocated_mb"]
            ),
            "forward_ms": float(row["timings"]["forward"]["median_ms"]),
            "nll_evaluation_ms": float(
                row["timings"]["nll_evaluation"]["median_ms"]
            ),
            "nll": float(
                np.mean(
                    [seed_records[str(seed)]["top"]["graph_mean_nll" if family.startswith("graph") else "full_mean_nll"] for seed in sorted(map(int, seed_records))]
                )
            ),
        }
    aggregate = {}
    for view, deltas in deltas_by_view.items():
        aggregate_delta = aggregate_paired_differences(deltas)
        aggregate[view] = paired_bootstrap_interval(
            aggregate_delta,
            repetitions=args.bootstrap_repetitions,
            seed=20260811,
        )
        aggregate[view]["subject_cluster_bootstrap"] = (
            subject_cluster_bootstrap_interval(
                aggregate_delta,
                subjects_by_view[view],
                repetitions=args.bootstrap_repetitions,
                seed=20260817,
            )
        )
    result = {
        "schema_version": 1,
        "comparison": "paired frozen-backbone Full-t versus Graph-t",
        "seeds": seed_records,
        "across_seed_mean_paired_difference": aggregate,
        "resources": resource_rows,
        "pareto": {
            "top_nll_vs_coordinates": pareto_frontier(
                resource_rows, x="coordinates", y="nll"
            ),
            "top_nll_vs_memory": pareto_frontier(
                resource_rows, x="memory_mb", y="nll"
            ),
        },
        "claim_boundary": (
            "Head-seed, shared-backbone family evidence; not independent-backbone "
            "robustness or calibrated OOD detection."
        ),
        "cluster_inference": (
            "Subject-cluster bootstrap uses the official compact label ID prefix; "
            "action-sequence metadata is not present in the compact artifact."
        ),
    }
    atomic_write_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
