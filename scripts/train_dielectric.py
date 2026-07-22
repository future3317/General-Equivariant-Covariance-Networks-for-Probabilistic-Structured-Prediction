"""Train the equivariant covariance model on dielectric tensor prediction."""

from __future__ import annotations

import argparse
import os
import json
import logging
import random
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

from equivcompiler import FeatureSpec, FullCovariance, plan_readout
from models import EquivariantBackbone
from data.dielectric_dataset import get_dielectric_irreps_loaders
from data.paths import dataset_dir
from data.tensor_conversions import irreps_to_km, irreps_to_matrix_exp_voigt
from voigt_utils import kelvin_mandel_to_voigt
from matrix_log_transform import matrix_exponential_transform
from scripts._common import add_tensor_product_arguments, tensor_product_kwargs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward(
    model,
    batch,
    *,
    target: torch.Tensor,
    use_bf16: bool,
):
    """Run BF16 only through the backbone; keep SPD/NLL in FP32."""
    if not use_bf16:
        return model(batch, target=target, return_scale=False)
    with torch.autocast(device_type=batch.pos.device.type, dtype=torch.bfloat16):
        node_features, graph_batch = model.backbone(batch)
    return model.forward_from_features(
        node_features.float(),
        graph_batch,
        target=target,
        return_scale=False,
    )


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


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    warmup_mse_weight: float = 0.0,
    non_blocking: bool = False,
    use_bf16: bool = False,
):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    num_samples = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device, non_blocking=non_blocking)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        result = _forward(
            model,
            batch,
            target=batch.y_irreps,
            use_bf16=use_bf16,
        )
        loss = result["loss"]

        if warmup_mse_weight > 0.0:
            mse = torch.nn.functional.mse_loss(result["mu"], batch.y_irreps)
            loss = loss + warmup_mse_weight * mse

        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError("non-finite dielectric training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
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
    non_blocking: bool = False,
    use_bf16: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_phys_abs = 0.0
    total_log_abs = 0.0
    num_loss_samples = 0
    num_mae_elements = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        batch = batch.to(device, non_blocking=non_blocking)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        result = _forward(
            model,
            batch,
            target=batch.y_irreps,
            use_bf16=use_bf16,
        )
        if not bool(torch.isfinite(result["loss"].detach()).all()):
            raise FloatingPointError("non-finite dielectric validation loss")
        batch_size = batch.y_irreps.shape[0]
        total_loss += result["loss"].item() * batch_size
        num_loss_samples += batch_size

        total_phys_abs += physical_mae(result["mu"], batch.y_km).item() * batch_size
        total_log_abs += log_mae(result["mu"], batch.y_km).item() * batch_size
        num_mae_elements += batch_size

    return {
        "loss": total_loss / max(num_loss_samples, 1),
        "phys_mae": total_phys_abs / max(num_mae_elements, 1),
        "log_mae": total_log_abs / max(num_mae_elements, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--dataset_storage", choices=["files", "shards"], default="files"
    )
    parser.add_argument("--shard_cache_size", type=int, default=2)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone_precision", choices=("bf16", "fp32"), default="bf16"
    )
    parser.add_argument("--allow_tf32", action="store_true")
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
    args.data_dir = str(dataset_dir(args.data_dir, "mp_dielectric"))
    if (
        args.backbone_precision == "bf16"
        and args.tp_backend == "cueq"
        and args.cueq_method == "fused_tp"
    ):
        raise ValueError(
            "cuEquivariance fused_tp does not provide the BF16 edge-feature "
            "kernel required by this backbone; use --backbone_precision fp32"
        )
    _set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.benchmark = True

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
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        lmax=args.lmax,
        storage=args.dataset_storage,
        shard_cache_size=args.shard_cache_size,
    )

    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
        atom_features=args.atom_features,
        **tensor_product_kwargs(args),
    )
    plan = plan_readout(
        FeatureSpec.from_backbone(backbone),
        output="0e + 2e",
        covariance=FullCovariance(),
        distribution="gaussian",
        output_scope="global",
    )
    compilation = plan.compilation
    model = plan.bind(backbone).to(device)
    if args.compile_tp:
        model.backbone.compile_tensor_products(dynamic=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info("Compiled lifting depth: %d", compilation.active_plan.depth)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    non_blocking = args.pin_memory and device.type == "cuda"
    use_bf16 = args.backbone_precision == "bf16" and device.type == "cuda"
    for epoch in range(args.num_epochs):
        warmup_mse = 0.1 if epoch < args.warmup_epochs else 0.0
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            warmup_mse,
            non_blocking=non_blocking,
            use_bf16=use_bf16,
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            non_blocking=non_blocking,
            use_bf16=use_bf16,
        )
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
    model.load_state_dict(
        torch.load(
            os.path.join(args.save_dir, "best_model.pt"), map_location=args.device
        )
    )
    test_metrics = validate(
        model,
        test_loader,
        device,
        non_blocking=non_blocking,
        use_bf16=use_bf16,
    )
    logger.info(
        f"Test: loss={test_metrics['loss']:.4f}, phys_mae={test_metrics['phys_mae']:.4f}, log_mae={test_metrics['log_mae']:.4f}"
    )

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
