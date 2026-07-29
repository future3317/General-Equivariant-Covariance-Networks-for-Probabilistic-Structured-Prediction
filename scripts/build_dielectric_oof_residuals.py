"""Build out-of-fold dielectric residuals for covariance supervision.

Each checkpoint is evaluated only on the fold it did not see during mean
training.  The resulting residual cache is a training artifact, not a new
label space: the covariance stage consumes it as a fixed residual target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compatibility.torch_geometric import PyGDataLoader
from data.dielectric_dataset import DielectricIrrepsDataset
from data.oof import fold_assignments
from data.paths import dataset_dir
from scripts.dielectric_runtime import (
    configure_inference_contract,
    dataset_provenance,
    forward_dielectric,
    inference_contract_from_args,
    load_dielectric_checkpoint,
    load_dielectric_data_args,
    load_run_record,
    sha256_file,
)


@torch.inference_mode()
def build_oof_residuals(
    checkpoint_dirs: list[str | Path],
    *,
    data_dir: str | Path | None,
    output: str | Path,
    device: str,
    folds: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> dict:
    if len(checkpoint_dirs) != folds:
        raise ValueError("one mean checkpoint directory is required per fold")
    first_args = load_dielectric_data_args(checkpoint_dirs[0])
    data_root = dataset_dir(data_dir or getattr(first_args, "data_dir", None), "mp_dielectric")
    lmax = int(getattr(first_args, "lmax", 2))
    storage = getattr(first_args, "dataset_storage", "files")
    shard_cache_size = int(getattr(first_args, "shard_cache_size", 2))
    dataset = DielectricIrrepsDataset(
        data_root,
        "train",
        lmax=lmax,
        storage=storage,
        shard_cache_size=shard_cache_size,
    )
    assignments = fold_assignments(len(dataset), folds, seed)
    residuals = torch.empty(len(dataset), 6, dtype=torch.float64)
    filled = torch.zeros(len(dataset), dtype=torch.bool)
    checkpoint_hashes = []
    for fold, checkpoint_dir in enumerate(checkpoint_dirs):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_args = load_dielectric_data_args(checkpoint_dir)
        if getattr(checkpoint_args, "training_stage", None) != "mean":
            raise ValueError(f"OOF checkpoint {checkpoint_dir} is not a mean-stage run")
        if (
            getattr(checkpoint_args, "oof_folds", None) != folds
            or getattr(checkpoint_args, "oof_seed", None) != seed
            or getattr(checkpoint_args, "oof_holdout_fold", None) != fold
        ):
            raise ValueError(
                f"OOF checkpoint {checkpoint_dir} does not certify fold {fold} "
                f"under the requested {fold}-fold seed-{seed} split"
            )
        checkpoint_hashes.append(
            {"path": str(checkpoint_dir / "best_model.pt"), "sha256": sha256_file(checkpoint_dir / "best_model.pt")}
        )
        model, _, _ = load_dielectric_checkpoint(checkpoint_dir, device)
        record = load_run_record(checkpoint_dir)
        contract = record.get("inference_contract") or inference_contract_from_args(
            load_dielectric_data_args(checkpoint_dir), device
        )
        configure_inference_contract(contract)
        indices = torch.where(assignments == fold)[0].tolist()
        loader = PyGDataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        for batch in loader:
            batch = batch.to(device)
            if batch.edge_index is None or batch.edge_index.numel() == 0:
                continue
            result = forward_dielectric(model, batch, contract=contract)
            sample_id = batch.sample_id.detach().cpu().long().reshape(-1)
            residuals[sample_id] = (
                batch.y_irreps.detach().double().cpu() - result["mu"].detach().double().cpu()
            )
            filled[sample_id] = True
    if not bool(filled.all()):
        missing = torch.where(~filled)[0].tolist()
        raise RuntimeError(
            f"OOF residual construction skipped {len(missing)} train samples; "
            "refusing to write a partially initialized covariance target"
        )
    payload = {
        "version": 1,
        "split": "train",
        "folds": folds,
        "seed": seed,
        "residuals": residuals,
        "fold_assignments": assignments,
        "dataset": dataset_provenance(data_root),
        "checkpoint_hashes": checkpoint_hashes,
        "checkpoint_chain_sha256": hashlib.sha256(
            json.dumps(checkpoint_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "coordinate_space": "compiled_irreps",
        "source_note": "Every residual is predicted by a mean checkpoint trained without its sample.",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {k: v for k, v in payload.items() if k not in {"residuals", "fold_assignments"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dirs", nargs="+", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    metadata = build_oof_residuals(
        args.checkpoint_dirs,
        data_dir=args.data_dir,
        output=args.output,
        device=args.device,
        folds=args.folds,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
