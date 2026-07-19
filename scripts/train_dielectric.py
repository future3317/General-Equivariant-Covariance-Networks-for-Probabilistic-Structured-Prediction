"""Train the equivariant covariance model on dielectric tensor prediction."""

from __future__ import annotations

import argparse
import os
import json
import logging
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
    O3EquivariantSymmetricOperatorHead,
    StructuredProbabilisticPredictor,
)
from data.dielectric_dataset import get_dielectric_irreps_loaders
from data.tensor_conversions import irreps_to_km, irreps_to_matrix_exp_voigt
from voigt_utils import kelvin_mandel_to_voigt
from matrix_log_transform import matrix_exponential_transform


def setup_logger(save_dir: str, experiment_name: str | None = None):
    if experiment_name is None:
        experiment_name = f"dielectric_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def physical_mae(pred_irreps: torch.Tensor, target_km: torch.Tensor) -> torch.Tensor:
    """Mean absolute error in physical dielectric tensor space."""
    # pred_irreps: [B, 6] log-tensor in irrep space
    # target_km: [B, 6] log-tensor in KM space
    pred_voigt = irreps_to_matrix_exp_voigt(pred_irreps)
    target_voigt = matrix_exponential_transform(kelvin_mandel_to_voigt(target_km))
    return torch.mean(torch.abs(pred_voigt - target_voigt))


def log_mae(pred_irreps: torch.Tensor, target_km: torch.Tensor) -> torch.Tensor:
    """Mean absolute error in log-Kelvin-Mandel space."""
    pred_km = irreps_to_km(pred_irreps)
    return torch.mean(torch.abs(pred_km - target_km))


def train_epoch(model, dataloader, optimizer, device, distribution, warmup_mse_weight: float = 0.0):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        result = model(batch, target=batch.y_irreps)
        loss = result["loss"]

        if warmup_mse_weight > 0.0:
            mse = torch.nn.functional.mse_loss(result["mu"], batch.y_irreps)
            loss = loss + warmup_mse_weight * mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.inference_mode()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_phys_mae = 0.0
    total_log_mae = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        batch = batch.to(device)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        result = model(batch, target=batch.y_irreps)
        total_loss += result["loss"].item()
        total_phys_mae += physical_mae(result["mu"], batch.y_km).item()
        total_log_mae += log_mae(result["mu"], batch.y_km).item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "phys_mae": total_phys_mae / max(num_batches, 1),
        "log_mae": total_log_mae / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/mp_dielectric")
    parser.add_argument("--save_dir", default="checkpoints_dielectric")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger, experiment_name = setup_logger(args.save_dir)
    logger.info("=" * 60)
    logger.info("GECN dielectric training")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    train_loader, val_loader, test_loader = get_dielectric_irreps_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
    )

    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3EquivariantSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)

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

    for epoch in range(args.num_epochs):
        warmup_mse = 0.1 if epoch < args.warmup_epochs else 0.0
        train_loss = train_epoch(model, train_loader, optimizer, args.device, GaussianNLL(), warmup_mse)
        val_metrics = validate(model, val_loader, args.device)
        scheduler.step(val_metrics["loss"])

        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_phys_mae={val_metrics['phys_mae']:.4f}, val_log_mae={val_metrics['log_mae']:.4f}"
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

    # Test on best model.
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=args.device))
    test_metrics = validate(model, test_loader, args.device)
    logger.info(f"Test: loss={test_metrics['loss']:.4f}, phys_mae={test_metrics['phys_mae']:.4f}, log_mae={test_metrics['log_mae']:.4f}")

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
