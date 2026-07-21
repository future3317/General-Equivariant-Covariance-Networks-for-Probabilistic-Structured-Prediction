"""Resumable single-GPU orchestration for the controlled ITOP study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable
import statistics


SEEDS = (42, 43, 44)
PROBABILISTIC_MODELS = (
    "independent_gaussian",
    "graph_gaussian",
    "graph_student_t",
)


def _complete(paths: Iterable[Path]) -> bool:
    return all(path.is_file() for path in paths)


def _run(
    command: list[str],
    *,
    outputs: tuple[Path, ...],
    environment: dict[str, str],
    dry_run: bool,
) -> None:
    if _complete(outputs):
        print(f"[skip complete] {outputs[0].parent}")
        return
    print("[run]", subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=environment)
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"stage completed without required artifacts: {missing}")


def _training_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    model: str,
    phase: str,
    seed: int,
    epochs: int,
    backbone_checkpoint: Path | None = None,
    feature_cache: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.train_itop",
        "--data_dir",
        str(args.data_dir),
        "--run_dir",
        str(run_dir),
        "--model",
        model,
        "--phase",
        phase,
        "--seed",
        str(seed),
        "--num_points",
        str(args.num_points),
        "--num_neighbors",
        "16",
        "--hidden_dim",
        "64",
        "--lmax",
        "2",
        "--num_layers",
        "2",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--num_epochs",
        str(epochs),
        "--patience",
        "5",
        "--backbone_precision",
        "bf16",
        "--tp_backend",
        "cueq",
        "--cueq_method",
        "fused_tp",
        "--device",
        "cuda:0",
    ]
    if backbone_checkpoint is not None:
        command.extend(("--backbone_checkpoint", str(backbone_checkpoint)))
    if feature_cache is not None:
        command.extend(("--feature_cache", str(feature_cache)))
    if resume_checkpoint is not None:
        command.extend(("--resume_checkpoint", str(resume_checkpoint)))
    if run_dir.is_dir() and not (run_dir / "last_state.pt").is_file():
        raise FileExistsError(
            f"incomplete run directory has no resumable state: {run_dir}"
        )
    if (run_dir / "last_state.pt").is_file():
        command.append("--continue_run")
    return command


def _best_graph_model(study_root: Path, seeds: tuple[int, ...]) -> tuple[str, dict]:
    candidates: dict[str, float] = {}
    for model in ("graph_gaussian", "graph_student_t"):
        seed_scores = []
        for seed in seeds:
            history_path = (
                study_root / f"seed_{seed}" / f"frozen_{model}" / "history.json"
            )
            history = json.loads(history_path.read_text(encoding="utf-8"))
            seed_scores.append(min(float(record["loss"]) for record in history))
        candidates[model] = statistics.fmean(seed_scores)
    selected = min(candidates, key=candidates.__getitem__)
    return selected, {"criterion": "mean_seed_validation_nll", "scores": candidates}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--study_dir", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("development", "final"), default="development"
    )
    parser.add_argument(
        "--gpu",
        required=True,
        help="one physical GPU index exposed to every child process",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if "," in args.gpu:
        raise ValueError("--gpu must name exactly one physical GPU")
    if args.profile == "development":
        args.num_points = 256
        seeds = SEEDS[:1]
        deterministic_epochs, frozen_epochs, joint_epochs = 30, 20, 10
    else:
        args.num_points = 512
        seeds = SEEDS
        deterministic_epochs, frozen_epochs, joint_epochs = 100, 60, 30

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["PYTHONUNBUFFERED"] = "1"
    study_root = args.study_dir / f"itop_{args.profile}_n{args.num_points}"
    if not args.dry_run:
        study_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "profile": args.profile,
            "single_gpu": True,
            "physical_gpu": args.gpu,
            "num_points": args.num_points,
            "seeds": list(seeds),
            "probabilistic_models": list(PROBABILISTIC_MODELS),
        }
        (study_root / "study_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    geometry_stages = (("side", "train"), ("side", "test"), ("top", "test"))
    for view, split in geometry_stages:
        cache = (
            args.data_dir
            / "cache"
            / f"{view}_{split}_n{args.num_points}_k16_centered_v1"
        )
        _run(
            [
                sys.executable,
                "-m",
                "scripts.precompute_itop_geometry",
                "--data_dir",
                str(args.data_dir),
                "--view",
                view,
                "--split",
                split,
                "--num_points",
                str(args.num_points),
                "--num_neighbors",
                "16",
            ],
            outputs=(cache / "metadata.json",),
            environment=environment,
            dry_run=args.dry_run,
        )

    deterministic_runs = []
    for seed in seeds:
        seed_root = study_root / f"seed_{seed}"
        deterministic = seed_root / "deterministic"
        deterministic_runs.append(deterministic)
        _run(
            _training_command(
                args,
                run_dir=deterministic,
                model="deterministic",
                phase="deterministic",
                seed=seed,
                epochs=deterministic_epochs,
            ),
            outputs=(
                deterministic / "best_model.pt",
                deterministic / "metrics.json",
                deterministic / "predictions_side.pt",
                deterministic / "predictions_top.pt",
            ),
            environment=environment,
            dry_run=args.dry_run,
        )
        checkpoint = deterministic / "best_model.pt"
        feature_cache = seed_root / "frozen_features"
        _run(
            [
                sys.executable,
                "-m",
                "scripts.precompute_itop_backbone_features",
                "--checkpoint",
                str(checkpoint),
                "--output_dir",
                str(feature_cache),
                "--data_dir",
                str(args.data_dir),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--device",
                "cuda:0",
            ],
            outputs=(
                feature_cache / "metadata.json",
                feature_cache / "side_train.pt",
                feature_cache / "side_test.pt",
                feature_cache / "top_test.pt",
            ),
            environment=environment,
            dry_run=args.dry_run,
        )
        for model in PROBABILISTIC_MODELS:
            frozen = seed_root / f"frozen_{model}"
            _run(
                _training_command(
                    args,
                    run_dir=frozen,
                    model=model,
                    phase="frozen_head",
                    seed=seed,
                    epochs=frozen_epochs,
                    backbone_checkpoint=checkpoint,
                    feature_cache=feature_cache,
                ),
                outputs=(frozen / "best_model.pt", frozen / "metrics.json"),
                environment=environment,
                dry_run=args.dry_run,
            )

    if args.dry_run:
        graph_model = "graph_gaussian_or_student_t_selected_by_validation_nll"
    else:
        graph_model, graph_selection = _best_graph_model(study_root, seeds)
        graph_selection["selected"] = graph_model
        (study_root / "graph_family_selection.json").write_text(
            json.dumps(graph_selection, indent=2) + "\n", encoding="utf-8"
        )

    for seed in seeds:
        seed_root = study_root / f"seed_{seed}"
        joint_models = ("independent_gaussian", graph_model)
        for model in joint_models:
            if model not in PROBABILISTIC_MODELS:
                print(
                    f"[deferred] select graph family after frozen-head runs for seed {seed}"
                )
                continue
            frozen = seed_root / f"frozen_{model}"
            joint = seed_root / f"joint_{model}"
            _run(
                _training_command(
                    args,
                    run_dir=joint,
                    model=model,
                    phase="joint_finetune",
                    seed=seed,
                    epochs=joint_epochs,
                    resume_checkpoint=frozen / "best_model.pt",
                ),
                outputs=(joint / "best_model.pt", joint / "metrics.json"),
                environment=environment,
                dry_run=args.dry_run,
            )

    if len(deterministic_runs) == 3:
        ensemble = study_root / "deterministic_ensemble"
        _run(
            [
                sys.executable,
                "-m",
                "scripts.evaluate_itop_ensemble",
                "--run_dirs",
                *(str(path) for path in deterministic_runs),
                "--output_dir",
                str(ensemble),
            ],
            outputs=(ensemble / "metrics.json",),
            environment=environment,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
