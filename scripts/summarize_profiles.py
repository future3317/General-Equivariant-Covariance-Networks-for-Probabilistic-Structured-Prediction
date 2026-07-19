"""Aggregate profile_summary.json files into a comparison CSV.

Example:
    python scripts/summarize_profiles.py \
        --pattern "results/profiles/*/profile_summary.json" \
        --output results/profiles/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def flatten(summary: dict, name: str) -> dict:
    args = summary.get("args", {})
    comp = summary.get("component_timing_ms", {})
    steady = summary.get("steady_state", {})

    row = {
        "name": name,
        "batch_size": args.get("batch_size"),
        "num_workers": args.get("num_workers"),
        "pin_memory": args.get("pin_memory"),
        "persistent_workers": args.get("persistent_workers"),
        "prefetch_factor": args.get("prefetch_factor"),
        "atom_features": args.get("atom_features"),
        "lmax": args.get("lmax"),
        "hidden_dim": args.get("hidden_dim"),
        "num_layers": args.get("num_layers"),
        "num_parameters": summary.get("num_parameters"),
        "tp_weight_numel": summary.get("tp_weight_numel"),
        "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
        "data_wait_ms": comp.get("data_wait_ms"),
        "h2d_ms": comp.get("h2d_ms"),
        "backbone_ms": comp.get("backbone_ms"),
        "mean_head_ms": comp.get("mean_head_ms"),
        "covariance_head_ms": comp.get("covariance_head_ms"),
        "spd_loss_ms": comp.get("spd_loss_ms"),
        "backward_ms": comp.get("backward_ms"),
        "grad_clip_ms": comp.get("grad_clip_ms"),
        "optimizer_ms": comp.get("optimizer_ms"),
        "total_component_ms": sum(
            comp.get(k, 0.0)
            for k in [
                "data_wait_ms",
                "h2d_ms",
                "backbone_ms",
                "mean_head_ms",
                "covariance_head_ms",
                "spd_loss_ms",
                "backward_ms",
                "grad_clip_ms",
                "optimizer_ms",
            ]
        ),
        "graphs_per_second": steady.get("graphs_per_second"),
        "nodes_per_second": steady.get("nodes_per_second"),
        "edges_per_second": steady.get("edges_per_second"),
        "steady_total_time_ms": steady.get("total_time_ms"),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="results/profiles/*/profile_summary.json")
    parser.add_argument("--output", default="results/profiles/comparison.csv")
    args = parser.parse_args()

    paths = sorted(Path().glob(args.pattern))
    if not paths:
        print(f"No files matched {args.pattern}")
        return

    rows = []
    for p in paths:
        summary = load_summary(p)
        name = p.parent.name
        rows.append(flatten(summary, name))

    rows.sort(key=lambda r: r.get("edges_per_second") or 0.0, reverse=True)

    fieldnames = [
        "name",
        "edges_per_second",
        "graphs_per_second",
        "nodes_per_second",
        "total_component_ms",
        "data_wait_ms",
        "h2d_ms",
        "backbone_ms",
        "mean_head_ms",
        "covariance_head_ms",
        "spd_loss_ms",
        "backward_ms",
        "grad_clip_ms",
        "optimizer_ms",
        "peak_cuda_memory_mb",
        "tp_weight_numel",
        "num_parameters",
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "atom_features",
        "lmax",
        "hidden_dim",
        "num_layers",
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    for r in rows:
        print(
            f"{r['name']:40s}  {r['edges_per_second']:10.1f} edges/s  "
            f"{r['graphs_per_second']:6.2f} graphs/s  "
            f"{r['total_component_ms']:8.1f} ms/batch"
        )


if __name__ == "__main__":
    main()
