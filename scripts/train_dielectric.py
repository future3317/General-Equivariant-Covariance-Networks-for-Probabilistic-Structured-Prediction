"""Train the equivariant covariance model on dielectric tensor prediction."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch import optim
from tqdm import tqdm

from data.dielectric_dataset import get_dielectric_irreps_loaders
from data.paths import dataset_dir
from data.pseudo_covariance import validate_pseudo_cache
from data.representation_metrics import (
    infer_rank2_block_metric,
    transformed_spectral_bounds,
)
from data.tensor_conversions import irreps_to_km, irreps_to_matrix_exp_voigt
from evaluation import (
    calibration_error,
    covariance_spectrum_diagnostics,
    empirical_coverage,
    mahalanobis_distance_squared,
    sharpness,
    whitened_residual_covariance,
)
from matrix_log_transform import matrix_exponential_transform
from scripts._common import add_tensor_product_arguments
from scripts.dielectric_runtime import (
    DielectricRunSpec,
    build_dielectric_model,
    compilation_record_with_hash,
    configure_inference_contract,
    dataset_provenance,
    forward_dielectric,
    inference_contract_from_args,
    load_dielectric_checkpoint,
    load_run_record,
    load_run_spec,
    sha256_file,
    source_provenance,
    write_run_spec,
)
from voigt_utils import kelvin_mandel_to_voigt


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
    faithful: bool = False,
    covariance_residual: torch.Tensor | None = None,
    pseudo_sqrt_covariance: torch.Tensor | None = None,
):
    """Compatibility wrapper around the unique inference runtime."""
    contract = {
        "backbone_precision": "bf16" if use_bf16 else "fp32",
    }
    return forward_dielectric(
        model,
        batch,
        target=target,
        return_scale=return_scale,
        contract=contract,
        faithful=faithful,
        covariance_residual=covariance_residual,
        pseudo_sqrt_covariance=pseudo_sqrt_covariance,
    )


def setup_logger(save_dir: str, experiment_name: str | None = None):
    if experiment_name is None:
        experiment_name = f"dielectric_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
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
    training_stage: str,
    warmup_mse_weight: float = 0.0,
    non_blocking: bool = False,
    use_bf16: bool = False,
    faithful: bool = False,
    oof_residuals: torch.Tensor | None = None,
    pseudo_sqrt_covariances: torch.Tensor | None = None,
):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    num_samples = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device, non_blocking=non_blocking)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        covariance_residual = None
        pseudo_sqrt_covariance = None
        if oof_residuals is not None:
            if not hasattr(batch, "sample_id"):
                raise ValueError("OOF residual training requires dataset sample_id")
            sample_id = batch.sample_id.detach().cpu().long().reshape(-1)
            if sample_id.max().item() >= oof_residuals.shape[0]:
                raise ValueError("OOF residual cache does not cover this dataset split")
            covariance_residual = oof_residuals[sample_id].to(device)
        if pseudo_sqrt_covariances is not None:
            if not hasattr(batch, "sample_id"):
                raise ValueError("pseudo-covariance training requires dataset sample_id")
            sample_id = batch.sample_id.detach().cpu().long().reshape(-1)
            if sample_id.max().item() >= pseudo_sqrt_covariances.shape[0]:
                raise ValueError("pseudo-covariance cache does not cover this dataset split")
            pseudo_sqrt_covariance = pseudo_sqrt_covariances[sample_id].to(device)
        result = _forward(
            model,
            batch,
            target=batch.y_irreps,
            use_bf16=use_bf16,
            faithful=faithful,
            covariance_residual=covariance_residual,
            pseudo_sqrt_covariance=pseudo_sqrt_covariance,
        )
        mse = (
            torch.zeros((), device=device)
            if training_stage == "covariance_warmup"
            else torch.nn.functional.mse_loss(result["mu"], batch.y_irreps)
        )
        loss = (
            mse if training_stage == "mean" else
            result["wasserstein_loss"] if training_stage == "covariance_warmup" else
            result["loss"]
        )

        if warmup_mse_weight > 0.0:
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
    training_stage: str = "joint",
):
    model.eval()
    total_nll = 0.0
    total_mse = 0.0
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
        total_nll += result["loss"].item() * batch_size
        total_mse += torch.nn.functional.mse_loss(
            result["mu"], batch.y_irreps
        ).item() * batch_size
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
        "nll": total_nll / max(num_loss_samples, 1),
        "mean_mse": total_mse / max(num_loss_samples, 1),
        "phys_mae": total_phys_abs / max(num_mae_elements, 1),
        "log_mae": total_log_abs / max(num_mae_elements, 1),
    }
    metrics["loss"] = metrics["mean_mse"] if training_stage == "mean" else metrics["nll"]
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


def _checkpoint_directory(path: str) -> str:
    candidate = os.path.abspath(path)
    return os.path.dirname(candidate) if os.path.isfile(candidate) else candidate


def configure_training_stage(model, spec: DielectricRunSpec, args: argparse.Namespace) -> dict:
    """Load the declared predecessor and expose exactly the stage parameters."""
    stage = args.training_stage
    expected_predecessor = {"mean": None, "covariance_warmup": "mean", "covariance": ("mean", "covariance_warmup"), "joint": "covariance"}
    if stage not in expected_predecessor:
        raise ValueError(f"unknown training stage: {stage}")
    if stage == "mean" and args.init_checkpoint is not None:
        raise ValueError("mean stage must start without --init_checkpoint")
    if stage != "mean" and args.init_checkpoint is None:
        raise ValueError(f"{stage} stage requires --init_checkpoint")

    source_dir = None
    if args.init_checkpoint is not None:
        source_dir = _checkpoint_directory(args.init_checkpoint)
        source_record = load_run_record(source_dir)
        source_spec = DielectricRunSpec.from_dict(source_record["model"])
        if source_spec != spec:
            raise ValueError(
                "init checkpoint has a different model semantic contract; "
                "start a new stage with identical RunSpec fields"
            )
        allowed = expected_predecessor[stage]
        allowed = (allowed,) if isinstance(allowed, str) else allowed
        if source_record.get("training_stage") not in allowed:
            raise ValueError(
                f"{stage} stage requires a predecessor in {allowed}, "
                f"got {source_record.get('training_stage')!r}"
            )
        source_model, _, _ = load_dielectric_checkpoint(source_dir, args.device)
        model.load_state_dict(source_model.state_dict())

    for parameter in model.parameters():
        parameter.requires_grad_(stage not in {"covariance", "covariance_warmup"})
    if stage in {"covariance", "covariance_warmup"}:
        projection = getattr(model.joint_head, "covariance_projection", None)
        if projection is None:
            raise TypeError("compiled readout does not expose covariance_projection")
        for parameter in projection.parameters():
            parameter.requires_grad_(True)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"{stage} stage exposes no trainable parameters")
    return {
        "training_stage": stage,
        "init_checkpoint": source_dir,
        "frozen_parameter_names": [
            name for name, parameter in model.named_parameters() if not parameter.requires_grad
        ],
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
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
    parser.add_argument(
        "--covariance_parameterization",
        choices=("matrix_exp", "spectral_window", "centered_spectral_window"),
        default="spectral_window",
        help="SPD realization used identically for training, validation, and inference.",
    )
    parser.add_argument("--log_variance_min", type=float, default=-4.0)
    parser.add_argument("--log_variance_max", type=float, default=4.0)
    parser.add_argument("--shape_min", type=float, default=-2.0)
    parser.add_argument("--shape_max", type=float, default=2.0)
    parser.add_argument("--volume_min", type=float, default=-8.0)
    parser.add_argument("--volume_max", type=float, default=8.0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument(
        "--training_stage",
        choices=("mean", "covariance_warmup", "covariance", "joint"),
        default="joint",
        help="strict training state: deterministic mean, covariance fit, or joint fine-tuning",
    )
    parser.add_argument(
        "--faithful_joint",
        action="store_true",
        help="Use MSE-only mean/trunk gradients and detached covariance residuals during joint training.",
    )
    parser.add_argument(
        "--pseudo_covariance_cache",
        default=None,
        help="Train-only isotropic OOF residual-covariance cache used only by covariance_warmup.",
    )
    parser.add_argument(
        "--oof_residuals",
        default=None,
        help="Path to a build_dielectric_oof_residuals.py cache used as fixed covariance residual supervision.",
    )
    parser.add_argument("--init_checkpoint", default=None)
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
    parser.add_argument(
        "--metric_sample_limit", type=int, default=256,
        help="maximum training samples used to estimate the equivariant metric",
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
    parser.add_argument("--oof_folds", type=int, default=None, help="Number of deterministic OOF folds for a mean checkpoint.")
    parser.add_argument("--oof_holdout_fold", type=int, default=None, help="Fold excluded from this OOF mean checkpoint.")
    parser.add_argument("--oof_seed", type=int, default=0, help="Shared OOF assignment seed recorded in every fold run.")
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
    if args.evaluate_only:
        checkpoint_args_path = os.path.join(args.save_dir, "args.json")
        if os.path.isfile(checkpoint_args_path):
            # Reconstruct the exact training semantics before creating the
            # model.  Only the runtime device and requested output directory
            # are allowed to come from the evaluate-only invocation.
            with open(checkpoint_args_path, encoding="utf-8") as handle:
                saved_args = json.load(handle)
            runtime_device = args.device
            runtime_save_dir = args.save_dir
            merged_args = vars(args).copy()
            merged_args.update(saved_args)
            merged_args["device"] = runtime_device
            merged_args["save_dir"] = runtime_save_dir
            merged_args["evaluate_only"] = True
            args = argparse.Namespace(**merged_args)
    oof_residuals = None
    pseudo_sqrt_covariances = None
    pseudo_covariance_metadata = None
    if args.oof_residuals is not None:
        if args.training_stage == "mean":
            parser.error("--oof_residuals is only valid for covariance or joint stages")
        payload = torch.load(args.oof_residuals, map_location="cpu")
        if not isinstance(payload, dict) or "residuals" not in payload:
            parser.error("invalid OOF residual cache: expected a residuals tensor")
        oof_residuals = payload["residuals"].float().contiguous()
    if args.pseudo_covariance_cache is not None:
        if args.training_stage != "covariance_warmup":
            parser.error("--pseudo_covariance_cache is valid only for covariance_warmup")
        payload = torch.load(args.pseudo_covariance_cache, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            parser.error("invalid pseudo-covariance cache")
        try:
            validate_pseudo_cache(payload)
        except ValueError as error:
            parser.error(str(error))
        pseudo_sqrt_covariances = payload["sqrt_covariance"].float().contiguous()
        pseudo_covariance_metadata = {
            key: value for key, value in payload.items() if not isinstance(value, torch.Tensor)
        }
        pseudo_covariance_metadata["cache_path"] = str(args.pseudo_covariance_cache)
        pseudo_covariance_metadata["cache_sha256"] = sha256_file(args.pseudo_covariance_cache)
    elif args.training_stage == "covariance_warmup":
        parser.error("covariance_warmup requires --pseudo_covariance_cache")
    if args.covariance_parameterization == "spectral_window" and not (
        args.log_variance_min < args.log_variance_max
    ):
        parser.error("--log_variance_min must be smaller than --log_variance_max")
    if args.covariance_parameterization == "centered_spectral_window" and not (
        args.shape_min < args.shape_max and args.volume_min < args.volume_max
    ):
        parser.error("centered spectral bounds must be strictly increasing")
    if (args.oof_folds is None) != (args.oof_holdout_fold is None):
        parser.error("--oof_folds and --oof_holdout_fold must be supplied together")
    if args.oof_folds is not None:
        if args.training_stage != "mean":
            parser.error("OOF fold exclusion is valid only while training mean checkpoints")
        if args.oof_folds != 5:
            parser.error("dielectric OOF residual construction requires exactly five folds")
        if not 0 <= args.oof_holdout_fold < args.oof_folds:
            parser.error("--oof_holdout_fold must lie in [0, --oof_folds)")
    # A successor stage inherits every model-semantic field from its declared
    # predecessor.  This prevents a default CLI value from silently changing
    # the compiled representation, SPD map, or likelihood between stages.
    if not args.evaluate_only and args.training_stage != "mean":
        if args.init_checkpoint is None:
            parser.error(f"--training_stage {args.training_stage} requires --init_checkpoint")
        inherited_spec = load_run_spec(_checkpoint_directory(args.init_checkpoint))
        for field, value in inherited_spec.as_dict().items():
            setattr(args, field, value)
    # The released dielectric graph cache contains an eight-coordinate radial
    # basis.  Treat this as an explicit dataset contract rather than allowing
    # a later opaque matrix-multiplication failure in the message-passing MLP.
    if args.num_basis != 8:
        parser.error("the dielectric graph cache requires --num_basis 8")
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
    inference_contract = inference_contract_from_args(args, device)
    configure_inference_contract(inference_contract)

    logger, _experiment_name = setup_logger(args.save_dir)
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
        oof_folds=args.oof_folds,
        oof_holdout_fold=args.oof_holdout_fold,
        oof_seed=args.oof_seed,
    )

    if args.representation_metric == "block_auto" and not args.evaluate_only:
        _metric, metric_stats = infer_rank2_block_metric(
            train_loader.dataset, max_samples=args.metric_sample_limit
        )
        args.metric_scalar = metric_stats["metric_scalar"]
        args.metric_l2 = metric_stats["metric_l2"]
        args.metric_stats = metric_stats
        logger.info("Representation metric: %s", metric_stats)
    if args.evaluate_only:
        model, spec, compilation = load_dielectric_checkpoint(args.save_dir, device)
        record = load_run_record(args.save_dir)
        inference_contract = record.get("inference_contract") or inference_contract
        configure_inference_contract(inference_contract)
    else:
        spec = DielectricRunSpec.from_namespace(args)
        model, compilation = build_dielectric_model(spec, device)
    stage_record = None
    if not args.evaluate_only:
        stage_record = configure_training_stage(model, spec, args)
        stage_record["faithful_joint"] = bool(
            args.faithful_joint and args.training_stage == "joint"
        )
        if pseudo_covariance_metadata is not None:
            stage_record["pseudo_covariance"] = pseudo_covariance_metadata
        write_run_spec(
            args.save_dir,
            spec,
            compilation=compilation.as_dict(),
            training_stage=args.training_stage,
            init_checkpoint=stage_record["init_checkpoint"],
            inference_contract=inference_contract,
            provenance={
                "source": source_provenance(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "dataset": dataset_provenance(args.data_dir),
            },
        )
        with open(os.path.join(args.save_dir, "stage.json"), "w") as f:
            json.dump(stage_record, f, indent=2)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
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
        # ``RepresentationMetricMap`` materializes the physical scale as
        # ``D^{-1} S_tilde D^{-1}``.  The declared spectral window therefore
        # has to be expressed in physical coordinates before diagnostics;
        # comparing against the internal window would report false violations.
        if bounds is not None and args.representation_metric == "block_auto":
            metric = torch.tensor(
                [args.metric_scalar] + [args.metric_l2] * 5,
                dtype=torch.float64,
            )
            bounds = transformed_spectral_bounds(bounds, metric)
        validation_metrics = validate(
            model,
            val_loader,
            device,
            non_blocking=non_blocking,
            use_bf16=use_bf16,
            diagnostics=True,
            log_variance_bounds=bounds,
            distribution=spec.distribution,
            student_t_dof=spec.student_t_dof,
            training_stage=args.training_stage,
        )
        test_metrics = validate(
            model,
            test_loader,
            device,
            non_blocking=non_blocking,
            use_bf16=use_bf16,
            diagnostics=True,
            log_variance_bounds=bounds,
            distribution=spec.distribution,
            student_t_dof=spec.student_t_dof,
            training_stage=args.training_stage,
        )
        with open(os.path.join(args.save_dir, "validation_metrics.json"), "w") as f:
            json.dump(validation_metrics, f, indent=2)
        with open(os.path.join(args.save_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        return validation_metrics, test_metrics

    if args.evaluate_only:
        validation_metrics, test_metrics = write_final_evaluations()
        # ``args.json`` is the immutable training contract.  Never overwrite
        # it from an evaluate-only invocation: doing so would replace the
        # checkpoint's parameterization/distribution with CLI defaults and
        # make the next evaluation mathematically inconsistent.
        with open(os.path.join(args.save_dir, "compilation.json"), "w") as f:
            json.dump(compilation_record_with_hash(compilation.as_dict()), f, indent=2)
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
        warmup_mse = 0.1 if args.training_stage == "joint" and epoch < args.warmup_epochs else 0.0
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.training_stage,
            warmup_mse,
            non_blocking=non_blocking,
            use_bf16=use_bf16,
            faithful=args.faithful_joint and args.training_stage == "joint",
            oof_residuals=oof_residuals,
            pseudo_sqrt_covariances=pseudo_sqrt_covariances,
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            non_blocking=non_blocking,
            use_bf16=use_bf16,
            training_stage=args.training_stage,
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

    with open(os.path.join(args.save_dir, "compilation.json"), "w") as f:
        json.dump(compilation_record_with_hash(compilation.as_dict()), f, indent=2)
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
