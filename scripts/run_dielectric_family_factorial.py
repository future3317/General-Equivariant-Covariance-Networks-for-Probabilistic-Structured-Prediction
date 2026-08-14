"""Run frozen-``H,mu`` dielectric operator-family by radial-law arms."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.frozen_distribution_features import frozen_distribution_loaders
from equivcompiler import (
    CenteredSpectralWindowCovariance,
    FeatureSpec,
    IsotypicBlockCovariance,
    LowRankCovariance,
    plan_readout,
)
from experiments.frozen_operator_arm import (
    FrozenOperatorArmSpec,
    evaluate_frozen_operator_arm,
    train_frozen_operator_arm,
)
from models.frozen_distribution_readout import FrozenMeanScatterElliptical
from scripts.itop_reproducibility import (
    atomic_write_json,
    sha256_file,
    source_provenance,
)
from spd_maps import RepresentationMetricMap

FAMILIES = ("isotropic", "block", "low_rank", "full")
DISTRIBUTIONS = ("gaussian", "student_t")
FAMILY_PARAMETER_COUNTS = {
    "isotropic": 1,
    "block": 2,
    "low_rank": 13,
    "full": 21,
}


def covariance_policy(family: str):
    """Return the exact typed policy predeclared for one factorial family."""

    policies = {
        "isotropic": LowRankCovariance(rank=0),
        "block": IsotypicBlockCovariance(),
        "low_rank": LowRankCovariance(rank=2),
        "full": CenteredSpectralWindowCovariance(-2.0, 2.0, -8.0, 8.0),
    }
    try:
        return policies[family]
    except KeyError as error:
        raise ValueError(f"unsupported factorial family: {family}") from error


def build_factorial_model(
    metadata: dict[str, Any],
    family: str,
    distribution: str,
    device: torch.device,
):
    """Compile one family/law composition against frozen feature semantics."""

    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"unsupported factorial distribution: {distribution}")
    plan = plan_readout(
        FeatureSpec.from_irreps(metadata["feature_irreps"], scope="global"),
        output=metadata["output_irreps"],
        covariance=covariance_policy(family),
        distribution=distribution,
        student_t_dof=float(metadata["student_t_dof"]),
        output_scope="global",
    )
    compilation = plan.compilation
    expected = FAMILY_PARAMETER_COUNTS[family]
    if compilation.covariance_parameter_count != expected:
        raise RuntimeError(
            f"{family} compiled {compilation.covariance_parameter_count} "
            f"coordinates, expected {expected}"
        )
    fidelity = compilation.as_dict()["execution_fidelity"]
    if (
        fidelity["exactness"] != "exact_for_active_family"
        or fidelity["checkpoint_mapping"] != "bijective"
    ):
        raise RuntimeError(f"factorial arm lacks exact execution fidelity: {fidelity}")
    spd_map = compilation.build_spd_map().to(device)
    metric = metadata["spd_map"].get("representation_metric", "none")
    if metric == "block_auto":
        values = [metadata["spd_map"]["metric_scalar"]] + [
            metadata["spd_map"]["metric_l2"]
        ] * (compilation.output_spec.dim - 1)
        spd_map = RepresentationMetricMap(
            spd_map, torch.tensor(values, dtype=torch.float32, device=device)
        ).to(device)
    elif metric != "none":
        raise ValueError(f"unsupported representation metric: {metric}")
    model = FrozenMeanScatterElliptical(
        metadata["feature_irreps"],
        compilation.operator_family.parameter_irreps,
        spd_map,
        distribution=distribution,
        student_t_dof=float(metadata["student_t_dof"]),
    ).to(device)
    return model, compilation


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _parse_csv(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(items).difference(allowed))
    if not items or unknown:
        raise ValueError(f"invalid {label}: {unknown or value}")
    return items


def _run_arm(
    args: argparse.Namespace,
    *,
    family: str,
    distribution: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    arm_dir = args.output_root / family / distribution / f"seed_{seed}"
    if arm_dir.exists():
        raise FileExistsError(f"refusing to overwrite factorial arm: {arm_dir}")
    _set_seed(seed)
    metadata, loaders = frozen_distribution_loaders(
        args.cache_dir,
        seed=seed,
        epoch=0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metadata = dict(metadata)
    model, compilation = build_factorial_model(
        metadata, family, distribution, device
    )

    def train_loader_for_epoch(epoch: int):
        _, epoch_loaders = frozen_distribution_loaders(
            args.cache_dir,
            seed=seed,
            epoch=epoch,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        return epoch_loaders["train"]

    training = train_frozen_operator_arm(
        model,
        train_loader_for_epoch=train_loader_for_epoch,
        validation_loader=loaders["val"],
        device=device,
        spec=FrozenOperatorArmSpec(
            run_dir=arm_dir,
            seed=seed,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
        ),
        checkpoint_metadata={
            "family": family,
            "distribution": distribution,
            "seed": seed,
        },
    )
    metrics, predictions = evaluate_frozen_operator_arm(
        model,
        loaders["test"],
        device=device,
        distribution=distribution,
        student_t_dof=float(metadata["student_t_dof"]),
        seed=seed,
    )
    prediction_path = arm_dir / "predictions_test.pt"
    _atomic_torch_save(predictions, prediction_path)
    diagnostics = {
        key: metrics.pop(key)
        for key in ("elliptical_falsification", "decision", "spectrum")
    }
    metrics.update(training)
    args_record = {
        "stage": args.stage,
        "family": family,
        "distribution": distribution,
        "seed": seed,
        "cache_dir": str(args.cache_dir),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "selection_split": "validation",
    }
    schema = {
        "distribution": model.schema(),
        "operator": compilation.operator_family.as_dict(),
        "compilation": compilation.as_dict(),
    }
    environment = {
        "source": source_provenance(Path(__file__).resolve().parents[1]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "pid": os.getpid(),
    }
    provenance = {
        "cache_metadata_sha256": sha256_file(args.cache_dir / "metadata.json"),
        "source_checkpoint": metadata["source_checkpoint"],
        "splits": metadata["splits"],
        "frozen_features": True,
        "frozen_mean": True,
    }
    atomic_write_json(args_record, arm_dir / "args.json")
    atomic_write_json(schema, arm_dir / "schema.json")
    atomic_write_json(environment, arm_dir / "environment.json")
    atomic_write_json(metrics, arm_dir / "metrics.json")
    atomic_write_json(diagnostics, arm_dir / "diagnostics.json")
    atomic_write_json(provenance, arm_dir / "provenance.json")
    names = (
        "args.json",
        "schema.json",
        "environment.json",
        "history.json",
        "best_model.pt",
        "last_model.pt",
        "predictions_test.pt",
        "metrics.json",
        "diagnostics.json",
        "provenance.json",
    )
    hashes = {name: sha256_file(arm_dir / name) for name in names}
    atomic_write_json(
        {
            "family": family,
            "distribution": distribution,
            "seed": seed,
            "artifact_sha256": hashes,
        },
        arm_dir / "manifest.json",
    )
    return {"arm_dir": str(arm_dir), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--distributions", default=",".join(DISTRIBUTIONS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--stage", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        parser.error("max_epochs and patience must be positive")
    if args.stage == "pilot" and args.max_epochs > 20:
        parser.error("pilot max_epochs cannot exceed 20")
    if args.stage == "formal" and args.max_epochs > 60:
        parser.error("formal max_epochs cannot exceed 60")
    families = _parse_csv(args.families, FAMILIES, "families")
    distributions = _parse_csv(
        args.distributions, DISTRIBUTIONS, "distributions"
    )
    seeds = tuple(int(value.strip()) for value in args.seeds.split(","))
    if args.stage == "formal" and seeds != (42, 43, 44):
        parser.error("formal factorial requires seeds 42,43,44")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    records = []
    for family in families:
        for distribution in distributions:
            for seed in seeds:
                records.append(
                    _run_arm(
                        args,
                        family=family,
                        distribution=distribution,
                        seed=seed,
                        device=device,
                    )
                )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
