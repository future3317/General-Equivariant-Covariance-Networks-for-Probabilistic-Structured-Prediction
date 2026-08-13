"""Train a representation-compiled probabilistic elasticity model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from data.elasticity_normalization import ElasticityTargetNormalizer
from data.elasticity_dataset import get_elasticity_irreps_loaders
from data.paths import dataset_dir
from data.representation_metrics import infer_representation_block_metric
from equivcompiler import FeatureSpec, plan_readout
from models import DeterministicHead, EquivariantBackbone, StructuredProbabilisticPredictor
from representations import O3IrrepsSpec, rank4_elasticity_irreps
from scripts._common import (
    add_tensor_product_arguments,
    covariance_policy_from_cli,
    tensor_product_kwargs,
)
from spd_maps import RepresentationMetricMap


ELASTICITY_ARMS = ("deterministic", "low_rank_student_t", "full_student_t")


def _configure_arm(args: argparse.Namespace) -> None:
    """Apply an optional named study arm without changing legacy CLI behavior."""
    if args.arm is None:
        return
    if args.arm == "deterministic":
        args.objective = "deterministic"
        args.covariance = None
    elif args.arm == "low_rank_student_t":
        args.objective = "student_t"
        args.covariance = "low_rank"
    elif args.arm == "full_student_t":
        args.objective = "student_t"
        args.covariance = "full"
    else:  # pragma: no cover - argparse restricts this value
        raise ValueError(f"unsupported elasticity arm: {args.arm}")


def build_elasticity_model(args: argparse.Namespace):
    """Build a named arm while retaining the existing compiler path."""
    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
        atom_features=args.atom_features,
        **tensor_product_kwargs(args),
    )
    output = rank4_elasticity_irreps()
    if args.objective == "deterministic":
        output_spec = O3IrrepsSpec(output)
        return (
            StructuredProbabilisticPredictor(
                backbone=backbone,
                output_spec=output_spec,
                joint_head=DeterministicHead(backbone.irreps_out, output_spec, pool=True),
            ),
            {"kind": "deterministic_mean", "output_irreps": str(output_spec.irreps)},
        )

    plan = plan_readout(
        FeatureSpec.from_backbone(backbone),
        output=output,
        covariance=covariance_policy_from_cli(
            args.covariance,
            rank=args.rank,
            parameter_budget=args.parameter_budget,
        ),
        distribution=args.objective,
        student_t_dof=args.student_t_dof,
        output_scope="global",
    )
    return plan.bind(backbone), plan.compilation.as_dict()


def setup_logger(save_dir: str, experiment_name: str | None = None):
    if experiment_name is None:
        experiment_name = f"elasticity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"{experiment_name}.log")

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, experiment_name


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    warmup_mse_weight: float = 0.0,
    non_blocking: bool = False,
):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    num_samples = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device, non_blocking=non_blocking)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        result = model(batch, target=batch.y_irreps, return_scale=False)
        loss = result["loss"]

        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError("non-finite elasticity training loss")

        if warmup_mse_weight > 0.0:
            mse = torch.nn.functional.mse_loss(result["mu"], batch.y_irreps)
            loss = loss + warmup_mse_weight * mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = batch.y_irreps.shape[0]
        total_loss += loss.detach() * batch_size
        num_samples += batch_size

    return (total_loss / max(num_samples, 1)).item()


@torch.inference_mode()
def validate(
    model,
    dataloader,
    device,
    normalizer: ElasticityTargetNormalizer,
    non_blocking: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    num_loss_samples = 0
    num_mae_samples = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        batch = batch.to(device, non_blocking=non_blocking)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        result = model(batch, target=batch.y_irreps, return_scale=False)
        if not bool(torch.isfinite(result["loss"].detach()).all()):
            raise FloatingPointError("non-finite elasticity validation loss")
        batch_size = batch.y_irreps.shape[0]
        total_loss += result["loss"].item() * batch_size
        num_loss_samples += batch_size

        pred_21d = normalizer.inverse(result["mu"])
        target_21d = batch.y_physical_21d

        total_abs += torch.sum(torch.abs(pred_21d - target_21d)).item()
        num_mae_samples += batch_size * pred_21d.shape[-1]

    return {
        "loss": total_loss / max(num_loss_samples, 1),
        "mae": total_abs / max(num_mae_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ELASTICITY_ARMS, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--save_dir", default="checkpoints_elasticity")
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--covariance",
        choices=["auto", "full", "block", "low_rank"],
        default="auto",
    )
    parser.add_argument("--parameter_budget", type=int, default=192)
    parser.add_argument(
        "--objective", choices=["gaussian", "student_t"], default="gaussian"
    )
    parser.add_argument("--student_t_dof", type=float, default=5.0)
    parser.add_argument(
        "--representation_metric", choices=("none", "block_auto"), default="none",
        help="training-set RMS metric repeated over each O(3) isotypic block",
    )
    parser.add_argument(
        "--target_normalization",
        choices=("legacy_voigt", "representation_compatible"),
        default="legacy_voigt",
        help=(
            "target normalization; legacy_voigt preserves historical runs "
            "and representation_compatible preserves the O(3) target action"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--eval_subset", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument(
        "--atom_features", default="manual", choices=["manual", "learnable"]
    )
    add_tensor_product_arguments(parser)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    _configure_arm(args)
    args.data_dir = str(dataset_dir(args.data_dir, "mp_elastic"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger, _experiment_name = setup_logger(args.save_dir)
    logger.info("=" * 60)
    logger.info("Representation-compiled elasticity training")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    train_loader, val_loader, test_loader = get_elasticity_irreps_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        eval_subset=args.eval_subset,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        lmax=args.lmax,
        num_basis=args.num_basis,
        normalization_mode=args.target_normalization,
    )

    # Train stats for unnormalization during validation.
    if isinstance(train_loader.dataset, torch.utils.data.Subset):
        train_dataset = train_loader.dataset.dataset
    else:
        train_dataset = train_loader.dataset
    normalizer = train_dataset.target_normalizer

    model, compilation = build_elasticity_model(args)
    model = model.to(args.device)
    if args.representation_metric == "block_auto":
        if model.spd_map is None:
            raise ValueError("representation_metric=block_auto requires a probabilistic arm")
        # Convert normalized Cartesian targets once; the metric is inferred
        # from representation blocks and is independent of the dataset name.
        target_irreps = train_dataset.target_irreps
        metric, metric_stats = infer_representation_block_metric(
            target_irreps, rank4_elasticity_irreps()
        )
        args.metric_stats = metric_stats
        args.metric = metric.tolist()
        model.spd_map = RepresentationMetricMap(model.spd_map, metric).to(args.device)
        logger.info("Representation metric: %s", metric_stats)
    if args.compile_tp:
        model.backbone.compile_tensor_products(dynamic=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    if model.distribution is None:
        logger.info("Deterministic mean control: output=%s", compilation["output_irreps"])
    else:
        logger.info(
            "Compiled covariance: mode=%s, parameters=%d, canonical_depth=%d, active_depth=%d",
            compilation["family"]["kind"],
            compilation["family"]["parameter_count"],
            compilation["representation_reachability"]["canonical"]["depth"],
            compilation["representation_reachability"]["active"]["depth"],
        )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    non_blocking = args.pin_memory and args.device.startswith("cuda")
    for epoch in range(args.num_epochs):
        warmup_mse = (
            0.1
            if model.distribution is not None and epoch < args.warmup_epochs
            else 0.0
        )
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            warmup_mse,
            non_blocking=non_blocking,
        )
        val_metrics = validate(
            model, val_loader, args.device, normalizer, non_blocking=non_blocking
        )
        scheduler.step(val_metrics["loss"])

        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_mae={val_metrics['mae']:.4f}"
        )

        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            logger.info("  -> Saved best model")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(
        torch.load(
            os.path.join(args.save_dir, "best_model.pt"), map_location=args.device
        )
    )
    test_metrics = validate(
        model, test_loader, args.device, normalizer, non_blocking=non_blocking
    )
    logger.info(f"Test: loss={test_metrics['loss']:.4f}, mae={test_metrics['mae']:.4f}")

    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.save_dir, "compilation.json"), "w") as f:
        json.dump(compilation, f, indent=2)
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
