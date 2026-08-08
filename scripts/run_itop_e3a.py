"""Run the minimal E3a independent end-to-end ITOP ensemble audit."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scripts.itop_reproducibility import (
    atomic_write_json,
    geometry_cache_provenance,
    source_provenance,
)

MEMBER_ARTIFACTS = (
    "best_model.pt",
    "last_state.pt",
    "history.json",
    "metrics.json",
    "predictions_side.pt",
    "predictions_top.pt",
    "args.json",
    "environment.json",
    "compilation.json",
    "feature_cache.json",
    "provenance.json",
    "train.log",
)


def _complete(root: Path) -> bool:
    return all((root / name).is_file() for name in MEMBER_ARTIFACTS)


def _member_command(args: argparse.Namespace, run_dir: Path, seed: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.train_itop",
        "--data_dir",
        str(args.data_dir),
        "--run_dir",
        str(run_dir),
        "--model",
        "full_student_t",
        "--phase",
        "end_to_end",
        "--student_t_dof",
        "5",
        "--representation_metric",
        "none",
        "--seed",
        str(seed),
        "--num_points",
        "512",
        "--num_neighbors",
        "16",
        "--hidden_dim",
        "64",
        "--lmax",
        "2",
        "--num_layers",
        "2",
        "--num_basis",
        "8",
        "--max_radius",
        "0.5",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--num_epochs",
        str(args.num_epochs),
        "--patience",
        "5",
        "--lr",
        "0.0005",
        "--weight_decay",
        "0.00001",
        "--backbone_precision",
        "bf16",
        "--tp_backend",
        args.tp_backend,
        "--cueq_method",
        args.cueq_method,
        "--device",
        "cuda:0",
    ]
    if (run_dir / "last_state.pt").is_file():
        command.append("--continue_run")
    elif run_dir.exists():
        raise FileExistsError(f"incomplete E3a member without resume state: {run_dir}")
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--study_dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--tp_backend", choices=("e3nn", "cueq"), default="e3nn")
    parser.add_argument("--cueq_method", choices=("naive", "fused_tp"), default="naive")
    parser.add_argument("--energy_samples", type=int, default=128)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if "," in args.gpu:
        raise ValueError("E3a uses exactly one physical GPU")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("E3a requires exactly three distinct independent seeds")
    if args.num_epochs < 1 or args.energy_samples < 16:
        raise ValueError("num_epochs must be positive and energy_samples at least 16")
    cache = geometry_cache_provenance(
        args.data_dir,
        num_points=512,
        num_neighbors=16,
        train_cache_sample_limit=None,
    )
    root = args.study_dir / "itop_e3a_full_student_t_n512"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["PYTHONUNBUFFERED"] = "1"
    manifest = {
        "schema_version": 1,
        "study": "E3a true end-to-end model/function uncertainty audit",
        "hypothesis": (
            "independently initialized full Student-t predictors generate useful "
            "function diversity beyond a frozen-head correction"
        ),
        "members": {
            "count": 3,
            "seeds": list(seeds),
            "independence": (
                "independent initialization and sampler seed; no input checkpoint, "
                "frozen feature cache, bootstrap, or shared mean path"
            ),
        },
        "model": "Full Student-t fixed nu=5, end-to-end, all parameters trainable",
        "selection": "Side validation NLL only; Side test and Top never select checkpoints",
        "controls": {
            "same_architecture_optimizer_schedule": True,
            "same_side_train_validation_split_policy": True,
            "same_geometry_cache_hash": cache["dataset_cache_hash"],
            "bootstrap_or_subsample": "not used in E3a",
        },
        "source": source_provenance(Path(__file__).resolve().parents[1]),
    }
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest, root / "study_manifest.json")
    for seed in seeds:
        run_dir = root / f"member_seed_{seed}"
        if _complete(run_dir):
            print(f"[skip complete] {run_dir}", flush=True)
            continue
        command = _member_command(args, run_dir, seed)
        print("[run]", subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, env=environment)
        if not args.dry_run and not _complete(run_dir):
            missing = [name for name in MEMBER_ARTIFACTS if not (run_dir / name).is_file()]
            raise RuntimeError(f"E3a member missing artifacts: {missing}")
    evaluation_dir = root / "ensemble_evaluation"
    if evaluation_dir.exists() and (evaluation_dir / "metrics.json").is_file():
        print(f"[skip complete] {evaluation_dir}", flush=True)
        return
    if evaluation_dir.exists():
        raise FileExistsError(f"incomplete E3a evaluation directory: {evaluation_dir}")
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_itop_probabilistic_ensemble",
        "--run_dirs",
        *(str(root / f"member_seed_{seed}") for seed in seeds),
        "--output_dir",
        str(evaluation_dir),
        "--device",
        "cuda:0",
        "--samples",
        str(args.energy_samples),
    ]
    print("[run]", subprocess.list2cmdline(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
