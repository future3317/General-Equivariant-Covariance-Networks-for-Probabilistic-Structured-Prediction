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

from equivcompiler import (
    FeatureSpec,
    FullCovariance,
    SpectralWindowCovariance,
    plan_readout,
)
from models import EquivariantBackbone
from data.dielectric_dataset import get_dielectric_irreps_loaders
from data.representation_metrics import infer_rank2_block_metric
from data.paths import dataset_dir
from data.tensor_conversions import irreps_to_km, irreps_to_matrix_exp_voigt
from voigt_utils import kelvin_mandel_to_voigt
from matrix_log_transform import matrix_exponential_transform
from evaluation import (
    calibration_error,
    covariance_spectrum_diagnostics,
    empirical_coverage,
    mahalanobis_distance_squared,
    sharpness,
    whitened_residual_covariance,
)
from scripts._common import add_tensor_product_arguments, tensor_product_kwargs
from spd_maps import RepresentationMetricMap


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
    return_scale: bool = False,
):
    """Run BF16 only through the backbone; keep SPD/NLL in FP32."""
    if not use_bf16:
        return model(batch, target=target, return_scale=return_scale)
    with torch.autocast(device_type=batch.pos.device.type, dtype=torch.bfloat16):
        node_features, graph_batch = model.backbone(batch)
    return model.forward_from_features(
        node_features.float(),
        graph_batch,
        target=target,
        return_scale=return_scale,
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
    diagnostics: bool = False,
    log_variance_bounds: tuple[float, float] | None = None,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
):
    model.eval()
    total_loss = 0.0
    total_phys_abs = 0.0
    total_log_abs = 0.0
    num_loss_samples = 0
    num_mae_elements = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []

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

        if diagnostics:
            # The model is trained in FP32, but diagnostics of a declared
            # spectral interval must not mistake FP32 reconstruction error of
            # a high-condition-number matrix for a violation of the compiled
            # distribution.  Re-materialize the same frozen generator in
            # FP64; this changes neither mu nor the trained parameters.
            if model.spd_map is None:
                raise TypeError("dielectric diagnostics require a probabilistic SPD map")
            predictions.append(result["mu"].detach().double().cpu())
            targets.append(batch.y_irreps.detach().double().cpu())
            scales.append(
                model.spd_map(result["params"].detach().double()).cpu()
            )

    metrics = {
        "loss": total_loss / max(num_loss_samples, 1),
        "phys_mae": total_phys_abs / max(num_mae_elements, 1),
        "log_mae": total_log_abs / max(num_mae_elements, 1),
    }
    if not diagnostics:
        return metrics

    if not scales:
        raise RuntimeError("no dielectric batches were available for diagnostics")
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    scale = torch.cat(scales)
    maha2 = mahalanobis_distance_squared(target - prediction, scale)
    metrics["probabilistic_diagnostics"] = {
        "coordinate_space": "log_kelvin_mandel",
        "scale_materialization_dtype": "float64",
        "calibration": calibration_error(
            prediction,
            target,
            scale,
            reference=distribution,
            student_t_dof=student_t_dof,
        ),
        "ellipsoid_coverage": empirical_coverage(
            prediction,
            target,
            scale,
            reference=distribution,
            student_t_dof=student_t_dof,
        ),
        "sharpness": sharpness(scale),
        "spectrum": covariance_spectrum_diagnostics(
            scale, log_variance_bounds=log_variance_bounds
        ),
        "mahalanobis2_mean": float(maha2.mean().item()),
        "mahalanobis2_median": float(maha2.median().item()),
        "whitened_residual_covariance_trace": float(
            whitened_residual_covariance(prediction, target, scale).item()
        ),
    }
    return metrics


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
    parser.add_argument(
        "--covariance_parameterization",
        choices=("matrix_exp", "spectral_window"),
        default="spectral_window",
        help="SPD realization used identically for training, validation, and inference.",
    )
    parser.add_argument("--log_variance_min", type=float, default=-4.0)
    parser.add_argument("--log_variance_max", type=float, default=4.0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument(
        "--distribution",
        choices=("gaussian", "student_t"),
        default="gaussian",
        help="proper probabilistic objective",
    )
    parser.add_argument("--student_t_dof", type=float, default=5.0)
    parser.add_argument(
        "--representation_metric",
        choices=("none", "block_auto"),
        default="none",
        help="equivariant 0e/2e target metric for multi-scale outputs",
    )
    parser.add_argument("--rotation_augmentation", action="store_true")
    parser.add_argument("--rotation_probability", type=float, default=1.0)
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Evaluate best_model.pt and write validation/test diagnostics without training.",
    )
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
    if args.covariance_parameterization == "spectral_window" and not (
        args.log_variance_min < args.log_variance_max
    ):
        parser.error("--log_variance_min must be smaller than --log_variance_max")
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
        rotation_augmentation=args.rotation_augmentation,
        rotation_probability=args.rotation_probability,
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
    covariance = (
        FullCovariance()
        if args.covariance_parameterization == "matrix_exp"
        else SpectralWindowCovariance(
            args.log_variance_min,
            args.log_variance_max,
        )
    )
    plan = plan_readout(
        FeatureSpec.from_backbone(backbone),
        output="0e + 2e",
        covariance=covariance,
        distribution=args.distribution,
        student_t_dof=args.student_t_dof,
        output_scope="global",
    )
    compilation = plan.compilation
    model = plan.bind(backbone).to(device)
    if args.representation_metric == "block_auto":
        metric, metric_stats = infer_rank2_block_metric(train_loader.dataset)
        args.metric_scalar = metric_stats["metric_scalar"]
        args.metric_l2 = metric_stats["metric_l2"]
        args.metric_stats = metric_stats
        model.spd_map = RepresentationMetricMap(model.spd_map, metric).to(device)
        logger.info("Representation metric: %s", metric_stats)
    if args.compile_tp:
        model.backbone.compile_tensor_products(dynamic=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info("Compiled lifting depth: %d", compilation.active_plan.depth)

    non_blocking = args.pin_memory and device.type == "cuda"
    use_bf16 = args.backbone_precision == "bf16" and device.type == "cuda"

    def write_final_evaluations() -> tuple[dict, dict]:
        bounds = (
            (args.log_variance_min, args.log_variance_max)
            if args.covariance_parameterization == "spectral_window"
            else None
        )
        validation_metrics = validate(
            model,
            val_loader,
            device,
            non_blocking=non_blocking,
            use_bf16=use_bf16,
            diagnostics=True,
            log_variance_bounds=bounds,
            distribution=args.distribution,
            student_t_dof=args.student_t_dof,
        )
        test_metrics = validate(
            model,
            test_loader,
            device,
            non_blocking=non_blocking,
            use_bf16=use_bf16,
            diagnostics=True,
            log_variance_bounds=bounds,
            distribution=args.distribution,
            student_t_dof=args.student_t_dof,
        )
        with open(os.path.join(args.save_dir, "validation_metrics.json"), "w") as f:
            json.dump(validation_metrics, f, indent=2)
        with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        return validation_metrics, test_metrics

    if args.evaluate_only:
        checkpoint_path = os.path.join(args.save_dir, "best_model.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        validation_metrics, test_metrics = write_final_evaluations()
        logger.info(
            "Validation: loss=%.4f, phys_mae=%.4f, log_mae=%.4f",
            validation_metrics["loss"],
            validation_metrics["phys_mae"],
            validation_metrics["log_mae"],
        )
        logger.info(
            "Test: loss=%.4f, phys_mae=%.4f, log_mae=%.4f",
            test_metrics["loss"],
            test_metrics["phys_mae"],
            test_metrics["log_mae"],
        )
        return

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

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
    validation_metrics, test_metrics = write_final_evaluations()
    logger.info(
        f"Test: loss={test_metrics['loss']:.4f}, phys_mae={test_metrics['phys_mae']:.4f}, log_mae={test_metrics['log_mae']:.4f}"
    )

    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.save_dir, "compilation.json"), "w") as f:
        json.dump(compilation.as_dict(), f, indent=2)
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
