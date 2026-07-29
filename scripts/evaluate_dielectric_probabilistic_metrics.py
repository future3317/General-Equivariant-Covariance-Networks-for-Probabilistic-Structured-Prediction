"""Compute distributional scores omitted from the training-time summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import torch

from data.dielectric_dataset import get_dielectric_irreps_loaders
from evaluation import (
    energy_score,
    isotropic_sliced_crps,
    risk_coverage_auc,
    sample_ensemble,
    variogram_score,
)
from evaluation.metrics import mahalanobis_distance_squared
from scripts.dielectric_runtime import (
    collect_dielectric_predictions,
    configure_inference_contract,
    inference_contract_from_args,
    inference_contract_hash,
    load_dielectric_checkpoint,
    load_dielectric_data_args,
    load_run_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="Monte-Carlo score seed")
    parser.add_argument(
        "--samples", type=int, default=128,
        help="Monte-Carlo samples for Energy Score and variogram score",
    )
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint_dir)
    model, _, _ = load_dielectric_checkpoint(checkpoint, args.device)
    train_args = load_dielectric_data_args(checkpoint)
    record = load_run_record(checkpoint)
    contract = record.get("inference_contract") or inference_contract_from_args(train_args, args.device)
    configure_inference_contract(contract)
    _, _, loader = get_dielectric_irreps_loaders(
        data_dir=train_args.data_dir,
        batch_size=train_args.batch_size,
        num_workers=getattr(train_args, "num_workers", 0),
        persistent_workers=getattr(train_args, "persistent_workers", False),
        pin_memory=getattr(train_args, "pin_memory", False),
        prefetch_factor=getattr(train_args, "prefetch_factor", None),
        lmax=train_args.lmax,
        storage=getattr(train_args, "dataset_storage", "files"),
        shard_cache_size=getattr(train_args, "shard_cache_size", 2),
    )
    pred = collect_dielectric_predictions(model, loader, args.device, inference_contract=contract)
    mu, target, scale = pred["mu_irreps"], pred["y_irreps"], pred["scale_irreps"]
    distribution = getattr(train_args, "distribution", "gaussian")
    dof = float(getattr(train_args, "student_t_dof", 5.0))
    error = (mu - target).norm(dim=-1)
    uncertainty = torch.diagonal(scale, dim1=-2, dim2=-1).sum(dim=-1)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    samples = sample_ensemble(
        mu.unsqueeze(0),
        scale.unsqueeze(0),
        num_samples=args.samples,
        distribution=distribution,
        student_t_dof=dof,
    )
    result = {
        "coordinate_space": "compiled_irreps",
        "distribution": distribution,
        "student_t_dof": dof,
        "monte_carlo_seed": args.seed,
        "monte_carlo_samples": args.samples,
        "inference_contract": contract,
        "inference_contract_hash": inference_contract_hash(contract),
        "energy_score": float(energy_score(mu, scale, target, num_samples=args.samples, distribution=distribution, student_t_dof=dof)),
        "isotropic_sliced_crps": float(isotropic_sliced_crps(mu, scale, target, num_directions=256, distribution=distribution, student_t_dof=dof)),
        "variogram_score": float(variogram_score(samples, target).item()),
        "risk_coverage_auc_log_irreps": float(risk_coverage_auc(uncertainty, error)),
        "mahalanobis2_mean": float(mahalanobis_distance_squared(target - mu, scale).mean()),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
