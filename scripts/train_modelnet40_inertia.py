"""Train the equivariant covariance model on ModelNet40 inertia tensors."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

from representations import O3IrrepsSpec
from spd_maps import MatrixExponentialMap
from distributions import GaussianNLL
from models import (
    EquivariantBackbone,
    EquivariantMeanHead,
    O3QuadraticSymmetricOperatorHead,
    StructuredProbabilisticPredictor,
)
from data.modelnet40_inertia_dataset import get_modelnet40_inertia_loaders
from data.tensor_conversions import irreps_to_voigt


def setup_logger(save_dir: str, experiment_name: str | None = None):
    if experiment_name is None:
        experiment_name = f"modelnet40_inertia_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def physical_mae(pred_irreps: torch.Tensor, y_voigt_mean: torch.Tensor, y_voigt_std: torch.Tensor) -> torch.Tensor:
    """Mean absolute error in physical Voigt space."""
    pred_voigt_norm = irreps_to_voigt(pred_irreps)
    pred_voigt_phys = pred_voigt_norm * y_voigt_std + y_voigt_mean
    # We do not have a batched physical target attached; callers pass normalized target.
    return torch.mean(torch.abs(pred_voigt_phys))


def train_epoch(model, dataloader, optimizer, device, distribution, non_blocking: bool = False):
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = batch.y_irreps.shape[0]
        total_loss += loss.detach() * batch_size
        num_samples += batch_size

    return (total_loss / max(num_samples, 1)).item()


@torch.inference_mode()
def validate(model, dataloader, device, non_blocking: bool = False):
    model.eval()
    total_loss = 0.0
    total_phys_abs = 0.0
    total_irreps_abs = 0.0
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

        # Physical-space MAE: denormalize predicted Voigt.
        pred_voigt_norm = irreps_to_voigt(result["mu"])
        pred_voigt_phys = pred_voigt_norm * batch.y_voigt_std.to(device) + batch.y_voigt_mean.to(device)
        target_voigt_norm = irreps_to_voigt(batch.y_irreps)
        target_voigt_phys = target_voigt_norm * batch.y_voigt_std.to(device) + batch.y_voigt_mean.to(device)

        total_phys_abs += torch.mean(torch.abs(pred_voigt_phys - target_voigt_phys)).item() * batch_size
        total_irreps_abs += torch.mean(torch.abs(result["mu"] - batch.y_irreps)).item() * batch_size
        num_mae_samples += batch_size

    return {
        "loss": total_loss / max(num_loss_samples, 1),
        "phys_mae": total_phys_abs / max(num_mae_samples, 1),
        "irreps_mae": total_irreps_abs / max(num_mae_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default="data/modelnet40/cache/modelnet40_inertia_dataset.pkl")
    parser.add_argument("--target_type", default="inertia", choices=["inertia", "shape_covariance"])
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_neighbors", type=int, default=16)
    parser.add_argument("--max_radius", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"checkpoints_modelnet40_{args.target_type}"

    logger, experiment_name = setup_logger(args.save_dir)
    logger.info("=" * 60)
    logger.info(f"GECN ModelNet40 {args.target_type} training")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    train_loader, val_loader, test_loader = get_modelnet40_inertia_loaders(
        cache_path=args.cache_path,
        target_type=args.target_type,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_neighbors=args.num_neighbors,
        max_radius=args.max_radius,
        num_basis=args.num_basis,
        lmax=args.lmax,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    output_spec = O3IrrepsSpec("0e + 2e")
    # Use learnable embedding because all points share the same type.
    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
        atom_features="learnable",
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3QuadraticSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)

    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    non_blocking = args.pin_memory and args.device.startswith("cuda")
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, args.device, GaussianNLL(), non_blocking)
        val_metrics = validate(model, val_loader, args.device, non_blocking)
        scheduler.step(val_metrics["loss"])

        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_phys_mae={val_metrics['phys_mae']:.4f}, val_irreps_mae={val_metrics['irreps_mae']:.4f}"
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

    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=args.device))
    test_metrics = validate(model, test_loader, args.device, non_blocking=non_blocking)
    logger.info(f"Test: loss={test_metrics['loss']:.4f}, phys_mae={test_metrics['phys_mae']:.4f}, irreps_mae={test_metrics['irreps_mae']:.4f}")

    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
