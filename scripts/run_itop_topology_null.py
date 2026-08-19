"""Run a preregistered ITOP topology-null manifest sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.audit_itop_topology_null import REQUIRED_FILES


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("outcome_filtered"):
        raise ValueError("topology manifest is outcome-filtered")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("topology manifest has no records")
    indices = [int(record["index"]) for record in records]
    if indices != list(range(len(records))):
        raise ValueError("topology manifest indices must be contiguous from zero")
    return records


def _command(args: argparse.Namespace, record: dict[str, Any], run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.train_itop",
        "--model",
        "shuffled_graph_student_t",
        "--phase",
        "frozen_head",
        "--backbone_checkpoint",
        str(args.backbone_checkpoint),
        "--feature_cache",
        str(args.feature_cache),
        "--topology_manifest",
        str(args.manifest),
        "--topology_index",
        str(record["index"]),
        "--data_dir",
        str(args.data_dir),
        "--run_dir",
        str(run_dir),
        "--student_t_dof",
        "5",
        "--hidden_dim",
        "64",
        "--lmax",
        "2",
        "--num_layers",
        "2",
        "--num_basis",
        "8",
        "--num_points",
        "512",
        "--num_neighbors",
        "16",
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--patience",
        str(args.patience),
        "--lr",
        "5e-4",
        "--weight_decay",
        "1e-5",
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        "2",
        "--seed",
        "42",
        "--split_seed",
        "42",
        "--backbone_precision",
        "bf16",
        "--tp_backend",
        "e3nn",
        "--cueq_method",
        "naive",
        "--device",
        "cuda:0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if "," in args.gpu:
        raise ValueError("--gpu must name exactly one physical GPU")
    if args.num_epochs < 1 or args.patience < 1:
        raise ValueError("--num-epochs and --patience must be positive")

    records = _load_manifest(args.manifest)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["PYTHONUNBUFFERED"] = "1"
    args.output_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        run_dir = (
            args.output_root
            / f"topology_{int(record['index']):02d}"
            / "seed_42"
            / "shuffled_graph_student_t"
        )
        missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
        if not missing:
            print(f"[skip complete] {run_dir}", flush=True)
            continue
        if run_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite incomplete topology run: {run_dir}"
            )
        command = _command(args, record, run_dir)
        print("[run]", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True, env=environment)
        missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"{run_dir}: missing required artifacts {missing}")


if __name__ == "__main__":
    main()
