"""Complete a trained elasticity run after a post-evaluation failure.

This path only reloads the recorded configuration and selected checkpoint.  It
does not train, change selection, or alter the saved training log.
"""

from __future__ import annotations

import argparse
import json
import re
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from data.elasticity_dataset import get_elasticity_irreps_loaders
from data.paths import dataset_dir
from data.representation_metrics import infer_representation_block_metric
from representations import rank4_elasticity_irreps
from scripts.evaluate_elasticity import evaluate_elasticity_predictions
from scripts.itop_reproducibility import sha256_file, source_provenance
from scripts.train_elasticity import build_elasticity_model, collect_predictions, validate
from spd_maps import RepresentationMetricMap


_EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+)/\d+: train_loss=(?P<train_loss>[-+0-9.eE]+), "
    r"train_nll=(?P<train_nll>[-+0-9.eE]+), val_loss=(?P<val_loss>[-+0-9.eE]+), "
    r"val_mae=(?P<val_mae>[-+0-9.eE]+)"
)


def _history_from_log(path: Path) -> list[dict[str, float | int]]:
    history = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _EPOCH_RE.search(line)
        if match is None:
            continue
        values = match.groupdict()
        history.append(
            {
                "epoch": int(values["epoch"]),
                "train_loss": float(values["train_loss"]),
                "train_nll": float(values["train_nll"]),
                "validation_criterion": float(values["val_loss"]),
                "val_loss": float(values["val_loss"]),
                "val_mae": float(values["val_mae"]),
            }
        )
    if not history:
        raise ValueError(f"no completed epochs found in {path}")
    return history


def complete_run(run_dir: Path, *, device: str | None = None) -> None:
    args = Namespace(**json.loads((run_dir / "args.json").read_text(encoding="utf-8")))
    args.data_dir = str(dataset_dir(args.data_dir, "mp_elastic"))
    if device is not None:
        args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, _, test_loader = get_elasticity_irreps_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        eval_subset=args.eval_subset,
        subset_seed=args.seed,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        lmax=args.lmax,
        num_basis=args.num_basis,
        normalization_mode=args.target_normalization,
    )
    train_dataset = train_loader.dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        train_dataset = train_dataset.dataset
    normalizer = train_dataset.target_normalizer

    model, schema = build_elasticity_model(args)
    model = model.to(args.device)
    if args.representation_metric == "block_auto":
        metric, metric_stats = infer_representation_block_metric(
            train_dataset.target_irreps, rank4_elasticity_irreps()
        )
        model.spd_map = RepresentationMetricMap(model.spd_map, metric).to(args.device)
        args.metric_stats = metric_stats
        args.metric = metric.tolist()

    checkpoint = run_dir / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    predictions = collect_predictions(
        model,
        test_loader,
        args.device,
        non_blocking=args.pin_memory and args.device.startswith("cuda"),
    )
    test_metrics = validate(
        model,
        test_loader,
        args.device,
        normalizer,
        non_blocking=args.pin_memory and args.device.startswith("cuda"),
    )
    evaluation = evaluate_elasticity_predictions(
        predictions,
        arm=args.arm,
        student_t_dof=args.student_t_dof,
        seed=args.seed,
    )
    history = _history_from_log(run_dir / "train.log")
    selected_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
    evaluation.update(
        {
            "mae_gpa": float(test_metrics["mae"]),
            "selected_epoch": selected_epoch,
            "postprocess_only_recovered": True,
        }
    )

    compact_predictions = {
        name: tensor for name, tensor in predictions.items() if name != "scale"
    }
    torch.save(compact_predictions, run_dir / "predictions.pt")
    data_files = sorted(Path(args.data_dir).glob("*.pkl"))
    environment = {
        "source": source_provenance(Path(__file__).resolve().parents[1]),
        "data_files": {path.name: sha256_file(path) for path in data_files},
        "split": {
            "seed": args.seed,
            "test_samples": len(test_loader.dataset),
        },
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "postprocess_only": True,
    }
    schema_record = {
        "schema_valid": bool(schema["family"]["certificates"]["valid"])
        and bool(schema["representation_reachability"]["active"]["reachable"]),
        "compiler": schema,
    }
    payloads = {
        "environment.json": environment,
        "schema.json": schema_record,
        "compilation.json": schema,
        "history.json": history,
        "metrics.json": evaluation,
    }
    for name, payload in payloads.items():
        (run_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    complete_run(args.run_dir, device=args.device)


if __name__ == "__main__":
    main()
