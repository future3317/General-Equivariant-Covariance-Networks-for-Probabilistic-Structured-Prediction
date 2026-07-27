"""Evaluate a finite dielectric deep ensemble without mislabeling its density."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compatibility.torch_geometric import PyGDataLoader
from data.dielectric_dataset import DielectricIrrepsDataset
from data.paths import dataset_dir
from evaluation import (
    combine_ensemble_moments,
    empirical_coverage,
    ensemble_nll,
    sample_ensemble,
    variogram_score,
)
from scripts.dielectric_runtime import (
    collect_dielectric_predictions,
    configure_inference_contract,
    load_dielectric_checkpoint,
    load_dielectric_data_args,
    load_run_record,
)


def _sample_energy_score(samples: torch.Tensor, target: torch.Tensor) -> float:
    first = torch.linalg.vector_norm(samples - target.unsqueeze(0), dim=-1).mean(0)
    pairwise = torch.linalg.vector_norm(
        samples[:, None] - samples[None, :], dim=-1
    ).mean((0, 1))
    return float((first - 0.5 * pairwise).mean().item())


@torch.inference_mode()
def evaluate_ensemble(
    checkpoint_dirs: list[str | Path],
    *,
    data_dir: str | Path | None,
    split: str,
    device: str,
    batch_size: int,
    num_workers: int,
    samples: int,
) -> dict:
    if len(checkpoint_dirs) < 2:
        raise ValueError("ensemble evaluation requires at least two checkpoints")
    first_args = load_dielectric_data_args(checkpoint_dirs[0])
    data_root = dataset_dir(data_dir or getattr(first_args, "data_dir", None), "mp_dielectric")
    dataset = DielectricIrrepsDataset(
        data_root,
        split,
        lmax=int(getattr(first_args, "lmax", 2)),
        storage=getattr(first_args, "dataset_storage", "files"),
        shard_cache_size=int(getattr(first_args, "shard_cache_size", 2)),
    )
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions = []
    contracts = []
    for checkpoint_dir in checkpoint_dirs:
        model, _, _ = load_dielectric_checkpoint(checkpoint_dir, device)
        record = load_run_record(checkpoint_dir)
        contract = record.get("inference_contract")
        if contract is None:
            raise ValueError(f"missing inference contract in {checkpoint_dir}")
        contracts.append(record["inference_contract_hash"])
        configure_inference_contract(contract)
        predictions.append(
            collect_dielectric_predictions(
                model, loader, device, inference_contract=contract
            )
        )
    if len(set(contracts)) != 1:
        raise ValueError("ensemble members use different inference contracts")
    means = torch.stack([p["mu_irreps"] for p in predictions])
    scales = torch.stack([p["scale_irreps"] for p in predictions])
    target = predictions[0]["y_irreps"]
    moments = combine_ensemble_moments(means, scales, distribution="student_t")
    mixture_samples = sample_ensemble(
        means, scales, num_samples=samples, distribution="student_t", student_t_dof=5.0
    )
    output = {
        "kind": "finite_student_t_deep_ensemble",
        "density_semantics": "equally_weighted_member_mixture",
        "members": [str(Path(x)) for x in checkpoint_dirs],
        "split": split,
        "inference_contract_hash": contracts[0],
        "mixture_nll": float(
            ensemble_nll(means, scales, target, distribution="student_t", student_t_dof=5.0).item()
        ),
        "energy_score": _sample_energy_score(mixture_samples, target),
        "variogram_score": float(variogram_score(mixture_samples, target).item()),
        "moment_gaussian_coverage": empirical_coverage(
            moments["mean"], target, moments["total_covariance"], reference="gaussian"
        ),
        "aleatoric_trace": float(torch.diagonal(moments["aleatoric_covariance"], dim1=-2, dim2=-1).sum(-1).mean()),
        "epistemic_trace": float(torch.diagonal(moments["epistemic_covariance"], dim1=-2, dim2=-1).sum(-1).mean()),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dirs", nargs="+", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    result = evaluate_ensemble(
        args.checkpoint_dirs,
        data_dir=args.data_dir,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples=args.samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
