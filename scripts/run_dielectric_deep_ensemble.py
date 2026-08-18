"""Train independently initialized dielectric ensemble members.

This runner only orchestrates the existing dielectric training contract. It
does not fit weights, consume test data, or replace exact mixture evaluation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def member_command(args: argparse.Namespace, seed: int, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("train_dielectric.py")),
        "--data_dir",
        str(args.data_dir),
        "--save_dir",
        str(output),
        "--seed",
        str(seed),
        "--distribution",
        args.distribution,
        "--student_t_dof",
        str(args.student_t_dof),
        "--covariance_parameterization",
        args.covariance_parameterization,
        "--training_stage",
        args.training_stage,
        "--hidden_dim",
        str(args.hidden_dim),
        "--lmax",
        str(args.lmax),
        "--num_layers",
        str(args.num_layers),
        "--num_basis",
        str(args.num_basis),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--num_epochs",
        str(args.num_epochs),
        "--patience",
        str(args.patience),
        "--backbone_precision",
        args.backbone_precision,
        "--device",
        args.device,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--distribution", choices=("gaussian", "student_t"), default="student_t")
    parser.add_argument("--student_t_dof", type=float, default=5.0)
    parser.add_argument(
        "--covariance_parameterization",
        choices=("matrix_exp", "spectral_window", "centered_spectral_window"),
        default="centered_spectral_window",
    )
    parser.add_argument("--training_stage", choices=("mean", "covariance", "joint"), default="joint")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--backbone_precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain at least two distinct integers")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        output = args.output_root / f"seed_{seed}"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite ensemble member: {output}")
        command = member_command(args, seed, output)
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
