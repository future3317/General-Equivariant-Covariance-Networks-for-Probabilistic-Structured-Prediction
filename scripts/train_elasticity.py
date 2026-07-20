"""Train a representation-compiled probabilistic elasticity model."""

from __future__ import annotations

import argparse
import os
import json
import logging
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

from representations import (
    CompilerConfig,
    O3RepresentationCompiler,
    rank4_elasticity_irreps,
)
from models import EquivariantBackbone
from data.elasticity_dataset import get_elasticity_irreps_loaders
from data.paths import dataset_dir
from data.tensor_conversions import irreps_to_elasticity_21d
from scripts._common import add_tensor_product_arguments, tensor_product_kwargs


def setup_logger(save_dir: str, experiment_name: str | None = None):
    if experiment_name is None:
        experiment_name = f"elasticity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def unnormalize_21d(
    pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return pred_norm * std.to(pred_norm.device) + mean.to(pred_norm.device)


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
def validate(model, dataloader, device, mean_21d, std_21d, non_blocking: bool = False):
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
        batch_size = batch.y_irreps.shape[0]
        total_loss += result["loss"].item() * batch_size
        num_loss_samples += batch_size

        pred_21d_norm = irreps_to_elasticity_21d(result["mu"])
        pred_21d = unnormalize_21d(pred_21d_norm, mean_21d, std_21d)
        target_21d = unnormalize_21d(batch.y, mean_21d, std_21d)

        total_abs += torch.sum(torch.abs(pred_21d - target_21d)).item()
        num_mae_samples += batch_size * pred_21d.shape[-1]

    return {
        "loss": total_loss / max(num_loss_samples, 1),
        "mae": total_abs / max(num_mae_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--train_subset", type=int, default=None)
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
    args.data_dir = str(dataset_dir(args.data_dir, "mp_elastic"))

    logger, experiment_name = setup_logger(args.save_dir)
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
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        lmax=args.lmax,
        num_basis=args.num_basis,
    )

    # Train stats for unnormalization during validation.
    if isinstance(train_loader.dataset, torch.utils.data.Subset):
        train_dataset = train_loader.dataset.dataset
    else:
        train_dataset = train_loader.dataset
    mean_21d = torch.tensor(train_dataset.mean_21d, dtype=torch.float32)
    std_21d = torch.tensor(train_dataset.std_21d, dtype=torch.float32)

    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
        atom_features=args.atom_features,
        **tensor_product_kwargs(args),
    )
    compiler = O3RepresentationCompiler(
        rank4_elasticity_irreps(),
        CompilerConfig(
            covariance=args.covariance,
            output_scope="global",
            objective=args.objective,
            parameter_budget=args.parameter_budget,
            low_rank=args.rank,
            student_t_dof=args.student_t_dof,
        ),
    )
    compilation = compiler.compile(backbone.irreps_out)
    model = compilation.build_model(backbone).to(args.device)
    if args.compile_tp:
        model.backbone.compile_tensor_products(dynamic=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(
        "Compiled covariance: mode=%s, parameters=%d, canonical_depth=%d, active_depth=%d",
        compilation.covariance_mode,
        compilation.covariance_parameter_count,
        compilation.canonical_plan.depth,
        compilation.active_plan.depth,
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
        warmup_mse = 0.1 if epoch < args.warmup_epochs else 0.0
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            warmup_mse,
            non_blocking=non_blocking,
        )
        val_metrics = validate(
            model, val_loader, args.device, mean_21d, std_21d, non_blocking=non_blocking
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
        model, test_loader, args.device, mean_21d, std_21d, non_blocking=non_blocking
    )
    logger.info(f"Test: loss={test_metrics['loss']:.4f}, mae={test_metrics['mae']:.4f}")

    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.save_dir, "compilation.json"), "w") as f:
        json.dump(compilation.as_dict(), f, indent=2)
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
