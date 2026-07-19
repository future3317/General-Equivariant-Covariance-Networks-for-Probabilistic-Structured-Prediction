"""Train the gecn model on elasticity tensor prediction with low-rank covariance."""

from __future__ import annotations

import argparse
import os
import json
import logging
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

from gecn import (
    O3IrrepsSpec,
    LowRankPlusIsotropicMap,
    GaussianNLL,
    EquivariantBackbone,
    EquivariantMeanHead,
    O3EquivariantLowRankCovarianceHead,
    StructuredProbabilisticPredictor,
    rank4_elasticity_irreps,
)
from gecn.data.elasticity_dataset import get_elasticity_irreps_loaders
from gecn.data.tensor_conversions import irreps_to_elasticity_21d


def setup_logger(save_dir: str, experiment_name: str | None = None):
    if experiment_name is None:
        experiment_name = f"gecn_elasticity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def unnormalize_21d(pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return pred_norm * std.to(pred_norm.device) + mean.to(pred_norm.device)


def train_epoch(model, dataloader, optimizer, device, warmup_mse_weight: float = 0.0):
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
def validate(model, dataloader, device, mean_21d, std_21d):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        batch = batch.to(device)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        result = model(batch, target=batch.y_irreps)
        pred_21d_norm = irreps_to_elasticity_21d(result["mu"])
        pred_21d = unnormalize_21d(pred_21d_norm, mean_21d, std_21d)
        target_21d = unnormalize_21d(batch.y, mean_21d, std_21d)

        total_loss += result["loss"].item()
        total_mae += torch.mean(torch.abs(pred_21d - target_21d)).item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "mae": total_mae / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/mp_elastic")
    parser.add_argument("--save_dir", default="checkpoints_gecn_elasticity")
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger, experiment_name = setup_logger(args.save_dir)
    logger.info("=" * 60)
    logger.info("GECN elasticity training (low-rank covariance)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    train_loader, val_loader, test_loader = get_elasticity_irreps_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
    )

    # Train stats for unnormalization during validation.
    if isinstance(train_loader.dataset, torch.utils.data.Subset):
        train_dataset = train_loader.dataset.dataset
    else:
        train_dataset = train_loader.dataset
    mean_21d = torch.tensor(train_dataset.mean_21d, dtype=torch.float32)
    std_21d = torch.tensor(train_dataset.std_21d, dtype=torch.float32)

    output_spec = O3IrrepsSpec(rank4_elasticity_irreps())
    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3EquivariantLowRankCovarianceHead(
        backbone.irreps_out, output_spec, rank=args.rank, pool=True
    )

    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=LowRankPlusIsotropicMap(dim=output_spec.dim, rank=args.rank),
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
        train_loss = train_epoch(model, train_loader, optimizer, args.device, warmup_mse)
        val_metrics = validate(model, val_loader, args.device, mean_21d, std_21d)
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

    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=args.device))
    test_metrics = validate(model, test_loader, args.device, mean_21d, std_21d)
    logger.info(f"Test: loss={test_metrics['loss']:.4f}, mae={test_metrics['mae']:.4f}")

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
