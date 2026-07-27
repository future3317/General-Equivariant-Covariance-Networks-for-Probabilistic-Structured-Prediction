"""Fit validation-only covariance temperature and report held-out metrics."""

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
from evaluation import calibration_error, empirical_coverage
from evaluation.temperature import (
    apply_block_temperature,
    apply_temperature,
    fit_block_temperatures,
    fit_temperature,
    scale_nll,
)
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
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    model, _, _ = load_dielectric_checkpoint(checkpoint_dir, args.device)
    train_args = load_dielectric_data_args(checkpoint_dir)
    record = load_run_record(checkpoint_dir)
    contract = record.get("inference_contract") or inference_contract_from_args(train_args, args.device)
    configure_inference_contract(contract)
    loaders = get_dielectric_irreps_loaders(
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
    val_loader, test_loader = loaders[1], loaders[2]
    val = collect_dielectric_predictions(model, val_loader, args.device, inference_contract=contract)
    test = collect_dielectric_predictions(model, test_loader, args.device, inference_contract=contract)
    distribution = getattr(train_args, "distribution", "gaussian")
    dof = getattr(train_args, "student_t_dof", 5.0)
    temperature = fit_temperature(
        val["mu_irreps"], val["y_irreps"], val["scale_irreps"],
        distribution=distribution, student_t_dof=dof,
    )
    calibrated_scale = apply_temperature(test["scale_irreps"], temperature)
    # ``0e + 2e`` has two isotypic blocks: one scalar coordinate and one
    # five-dimensional l=2 block.  Block temperatures preserve equivariance.
    block_ids = torch.tensor([0, 1, 1, 1, 1, 1], dtype=torch.long)
    block_temperatures = fit_block_temperatures(
        val["mu_irreps"],
        val["y_irreps"],
        val["scale_irreps"],
        block_ids,
        distribution=distribution,
        student_t_dof=dof,
    )
    block_calibrated_scale = apply_block_temperature(
        test["scale_irreps"], block_ids, block_temperatures
    )
    result = {
        "checkpoint_dir": str(checkpoint_dir),
        "fit_split": "validation",
        "eval_split": "test",
        "distribution": distribution,
        "student_t_dof": dof,
        "inference_contract": contract,
        "inference_contract_hash": inference_contract_hash(contract),
        "temperature": temperature,
        "validation_nll_before": float(scale_nll(
            val["mu_irreps"], val["y_irreps"], val["scale_irreps"],
            distribution=distribution, student_t_dof=dof,
        ).item()),
        "test_nll_before": float(scale_nll(
            test["mu_irreps"], test["y_irreps"], test["scale_irreps"],
            distribution=distribution, student_t_dof=dof,
        ).item()),
        "test_nll_after": float(scale_nll(
            test["mu_irreps"], test["y_irreps"], calibrated_scale,
            distribution=distribution, student_t_dof=dof,
        ).item()),
        "test_calibration_before": calibration_error(
            test["mu_irreps"], test["y_irreps"], test["scale_irreps"],
            reference=distribution, student_t_dof=dof,
        ),
        "test_calibration_after": calibration_error(
            test["mu_irreps"], test["y_irreps"], calibrated_scale,
            reference=distribution, student_t_dof=dof,
        ),
        "test_coverage_before": empirical_coverage(
            test["mu_irreps"], test["y_irreps"], test["scale_irreps"],
            reference=distribution, student_t_dof=dof,
        ),
        "test_coverage_after": empirical_coverage(
            test["mu_irreps"], test["y_irreps"], calibrated_scale,
            reference=distribution, student_t_dof=dof,
        ),
        "block_temperature": {
            "block_ids": block_ids.tolist(),
            "temperatures": block_temperatures,
            "test_nll_after": float(scale_nll(
                test["mu_irreps"], test["y_irreps"], block_calibrated_scale,
                distribution=distribution, student_t_dof=dof,
            ).item()),
            "test_calibration_after": calibration_error(
                test["mu_irreps"], test["y_irreps"], block_calibrated_scale,
                reference=distribution, student_t_dof=dof,
            ),
            "test_coverage_after": empirical_coverage(
                test["mu_irreps"], test["y_irreps"], block_calibrated_scale,
                reference=distribution, student_t_dof=dof,
            ),
        },
    }
    output = Path(args.output) if args.output else checkpoint_dir / "temperature_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
