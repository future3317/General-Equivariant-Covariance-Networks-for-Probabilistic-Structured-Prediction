"""Train a representation-compiled probabilistic elasticity model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from data.elasticity_dataset import get_elasticity_irreps_loaders
from data.elasticity_normalization import ElasticityTargetNormalizer
from data.paths import dataset_dir
from data.representation_metrics import infer_representation_block_metric
from equivcompiler import FeatureSpec, plan_readout
from models import (
    DeterministicHead,
    EquivariantBackbone,
    StructuredProbabilisticPredictor,
)
from representations import O3IrrepsSpec, rank4_elasticity_irreps
from scripts._common import (
    add_tensor_product_arguments,
    covariance_policy_from_cli,
    tensor_product_kwargs,
)
from scripts.evaluate_elasticity import evaluate_elasticity_predictions
from scripts.itop_reproducibility import sha256_file, source_provenance
from spd_maps import RepresentationMetricMap

ELASTICITY_ARMS = (
    "deterministic",
    "low_rank_student_t",
    "full_student_t",
)
ELASTICITY_CANDIDATE_ARMS = ("full_asinh_exp_student_t",)
ELASTICITY_ALL_ARMS = ELASTICITY_ARMS + ELASTICITY_CANDIDATE_ARMS


def _configure_arm(args: argparse.Namespace) -> None:
    """Apply a named arm while retaining the legacy trainer contract."""

    if args.arm is None:
        return
    configuration = elasticity_arm_configuration(args.arm)
    args.objective = str(configuration["objective"])
    args.covariance = configuration["covariance"]


def elasticity_arm_configuration(arm: str) -> dict[str, str | None]:
    """Map study-arm names to the smallest scientific intervention."""

    configurations = {
        "deterministic": {"objective": "deterministic", "covariance": None},
        "low_rank_student_t": {
            "objective": "student_t",
            "covariance": "low_rank",
        },
        "full_student_t": {"objective": "student_t", "covariance": "full"},
        "full_asinh_exp_student_t": {
            "objective": "student_t",
            "covariance": "asinh_exponential",
        },
    }
    try:
        return configurations[arm]
    except KeyError as error:
        raise ValueError(f"unsupported elasticity arm: {arm}") from error


def build_elasticity_model(args: argparse.Namespace):
    """Build one matched elasticity arm and return its machine-readable schema."""

    configuration = elasticity_arm_configuration(args.arm)
    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
        atom_features=args.atom_features,
        **tensor_product_kwargs(args),
    )
    output_spec = O3IrrepsSpec(rank4_elasticity_irreps())
    if configuration["objective"] == "deterministic":
        model = StructuredProbabilisticPredictor(
            backbone=backbone,
            output_spec=output_spec,
            joint_head=DeterministicHead(backbone.irreps_out, output_spec, pool=True),
        )
        return model, {
            "kind": "deterministic_mean",
            "output_irreps": str(output_spec.irreps),
            "output_dimension": output_spec.dim,
        }

    plan = plan_readout(
        FeatureSpec.from_backbone(backbone),
        output=rank4_elasticity_irreps(),
        covariance=covariance_policy_from_cli(
            str(configuration["covariance"]),
            rank=args.rank,
            parameter_budget=args.parameter_budget,
            shape_min=getattr(args, "shape_min", -2.0),
            shape_max=getattr(args, "shape_max", 2.0),
            volume_min=getattr(args, "volume_min", -8.0),
            volume_max=getattr(args, "volume_max", 8.0),
        ),
        distribution=str(configuration["objective"]),
        student_t_dof=args.student_t_dof,
        quadratic_oracle=getattr(args, "student_t_quadratic_oracle", "direct"),
        output_scope="global",
    )
    return plan.bind(backbone), plan.compilation.as_dict()


def setup_logger(save_dir: str, experiment_name: str | None = None):
    if experiment_name is None:
        experiment_name = f"elasticity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train.log")

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


def _operator_forward_diagnostics(model, params, residual) -> dict[str, float]:
    """Summarize generator and quadratic-form values during stability audits."""
    spd_map = model.spd_map
    while getattr(spd_map, "_transform_parameters", None) is None and hasattr(
        spd_map, "base"
    ):
        spd_map = spd_map.base
    transform = getattr(spd_map, "_transform_parameters", None)
    if transform is None:
        return {}
    with torch.no_grad():
        generator = transform(params.detach())
        generator = 0.5 * (generator + generator.transpose(-1, -2))
        eigenvalues = torch.linalg.eigvalsh(generator)
        log_quadratic = model.spd_map.log_precision_action(
            params.detach(), residual.detach()
        )
        return {
            "generator_lambda_min": float(eigenvalues[..., 0].min()),
            "generator_lambda_max": float(eigenvalues[..., -1].max()),
            "generator_frobenius_max": float(
                generator.square().sum(dim=(-2, -1)).sqrt().max()
            ),
            "residual_norm_max": float(residual.square().sum(dim=-1).sqrt().max()),
            "log_quadratic_min": float(log_quadratic.min()),
            "log_quadratic_max": float(log_quadratic.max()),
            "generator_trace_min": float(
                torch.diagonal(generator, dim1=-2, dim2=-1).sum(-1).min()
            ),
            "generator_trace_max": float(
                torch.diagonal(generator, dim1=-2, dim2=-1).sum(-1).max()
            ),
        }


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    warmup_mse_weight: float = 0.0,
    non_blocking: bool = False,
    record_operator_diagnostics: bool = False,
    return_stats: bool = False,
):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_nll = torch.tensor(0.0, device=device)
    num_samples = 0
    diagnostic_extrema: dict[str, float] = {}

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device, non_blocking=non_blocking)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        result = model(batch, target=batch.y_irreps, return_scale=False)
        nll = result["loss"]

        if record_operator_diagnostics and "params" in result:
            forward_diagnostics = _operator_forward_diagnostics(
                model,
                result["params"],
                batch.y_irreps - result["mu"].detach(),
            )
            for key, value in forward_diagnostics.items():
                if key.endswith("_min"):
                    diagnostic_extrema[key] = min(
                        diagnostic_extrema.get(key, float("inf")), value
                    )
                else:
                    diagnostic_extrema[key] = max(
                        diagnostic_extrema.get(key, float("-inf")), value
                    )

        if not bool(torch.isfinite(nll.detach()).all()):
            raise FloatingPointError(
                "non-finite elasticity training loss; "
                f"forward_diagnostics={forward_diagnostics if record_operator_diagnostics else {}}"
            )

        loss = nll
        if warmup_mse_weight > 0.0:
            mse = torch.nn.functional.mse_loss(result["mu"], batch.y_irreps)
            loss = loss + warmup_mse_weight * mse

        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError("non-finite elasticity training objective")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError(
                "non-finite elasticity parameter after optimizer step"
            )

        batch_size = batch.y_irreps.shape[0]
        total_loss += loss.detach() * batch_size
        total_nll += nll.detach() * batch_size
        num_samples += batch_size

    stats = {
        "loss": (total_loss / max(num_samples, 1)).item(),
        "nll": (total_nll / max(num_samples, 1)).item(),
    }
    if record_operator_diagnostics:
        stats.update(diagnostic_extrema)
    return stats if return_stats else stats["loss"]


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


@torch.inference_mode()
def collect_predictions(model, dataloader, device, *, non_blocking: bool = False):
    """Collect one compact test artifact and in-memory scatters for evaluation."""

    records: dict[str, list[torch.Tensor]] = {
        "sample_id": [],
        "mean": [],
        "target": [],
    }
    probabilistic = model.distribution is not None
    if probabilistic:
        records["params"] = []
        records["scale"] = []
    for batch in tqdm(dataloader, desc="Predictions", leave=False):
        batch = batch.to(device, non_blocking=non_blocking)
        output = model(batch, return_scale=probabilistic)
        records["sample_id"].append(batch.sample_id.detach().cpu().reshape(-1))
        records["mean"].append(output["mu"].detach().cpu())
        records["target"].append(batch.y_irreps.detach().cpu())
        if probabilistic:
            records["params"].append(output["params"].detach().cpu())
            records["scale"].append(output["scale"].detach().double().cpu())
    return {name: torch.cat(values, dim=0) for name, values in records.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ELASTICITY_ALL_ARMS, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--save_dir", default="checkpoints_elasticity")
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--covariance",
        choices=[
            "auto",
            "full",
            "block",
            "low_rank",
            "centered_spectral_window",
            "asinh_exponential",
        ],
        default="auto",
    )
    parser.add_argument("--parameter_budget", type=int, default=192)
    parser.add_argument("--shape_min", type=float, default=-2.0)
    parser.add_argument("--shape_max", type=float, default=2.0)
    parser.add_argument("--volume_min", type=float, default=-8.0)
    parser.add_argument("--volume_max", type=float, default=8.0)
    parser.add_argument(
        "--objective", choices=["gaussian", "student_t"], default="gaussian"
    )
    parser.add_argument("--student_t_dof", type=float, default=5.0)
    parser.add_argument(
        "--student_t_quadratic_oracle",
        choices=("direct", "shifted_log"),
        default="direct",
        help="Student-t quadratic-form lowering used by the matrix-exponential map",
    )
    parser.add_argument(
        "--representation_metric", choices=("none", "block_auto"), default="none",
        help="training-set RMS metric repeated over each O(3) isotypic block",
    )
    parser.add_argument(
        "--target_normalization",
        choices=(
            "legacy_voigt",
            "representation_compatible",
            "representation_compatible_multiplicity",
        ),
        default="legacy_voigt",
        help=(
            "target normalization; legacy_voigt preserves historical runs; "
            "representation_compatible preserves the O(3) target action; "
            "representation_compatible_multiplicity whitens repeated irreps"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--record_operator_diagnostics",
        action="store_true",
        help="record generator spectrum and log-quadratic diagnostics per epoch",
    )
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
    if args.arm is None:
        legacy = (args.covariance, args.objective)
        legacy_arms = {
            ("low_rank", "student_t"): "low_rank_student_t",
            ("full", "student_t"): "full_student_t",
        }
        if legacy not in legacy_arms:
            parser.error(
                "the audited trainer requires --arm; legacy Gaussian/Block commands "
                "are no longer accepted by this evidence runner"
            )
        args.arm = legacy_arms[legacy]
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
        subset_seed=args.seed,
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

    model, schema = build_elasticity_model(args)
    model = model.to(args.device)
    if args.representation_metric == "block_auto":
        if model.spd_map is None:
            raise ValueError("representation metric requires a probabilistic arm")
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
        logger.info("Deterministic mean control: output_dimension=%d", schema["output_dimension"])
    else:
        logger.info(
            "Compiled scatter: mode=%s, coordinates=%d, canonical_depth=%d, active_depth=%d",
            schema["family"]["kind"],
            schema["family"]["parameter_count"],
            schema["representation_reachability"]["canonical"]["depth"],
            schema["representation_reachability"]["active"]["depth"],
        )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    selected_epoch = 0
    patience_counter = 0
    history = []

    non_blocking = args.pin_memory and args.device.startswith("cuda")
    using_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if using_cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    for epoch in range(args.num_epochs):
        warmup_mse = (
            0.1
            if model.distribution is not None and epoch < args.warmup_epochs
            else 0.0
        )
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            warmup_mse,
            non_blocking=non_blocking,
            record_operator_diagnostics=args.record_operator_diagnostics,
            return_stats=True,
        )
        val_metrics = validate(
            model, val_loader, args.device, normalizer, non_blocking=non_blocking
        )
        scheduler.step(val_metrics["loss"])

        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs}: "
            f"train_loss={train_stats['loss']:.4f}, "
            f"train_nll={train_stats['nll']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_mae={val_metrics['mae']:.4f}"
            + (
                " | operator_diagnostics="
                + json.dumps(
                    {
                        key: value
                        for key, value in train_stats.items()
                        if key not in {"loss", "nll"}
                    },
                    sort_keys=True,
                )
                if args.record_operator_diagnostics
                else ""
            )
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_nll": train_stats["nll"],
                **{
                    key: value
                    for key, value in train_stats.items()
                    if key not in {"loss", "nll"}
                },
                "validation_criterion": val_metrics["loss"],
                **val_metrics,
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            selected_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            logger.info("  -> Saved best model")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if using_cuda:
        torch.cuda.synchronize(args.device)
    elapsed_seconds = time.perf_counter() - started

    model.load_state_dict(
        torch.load(
            os.path.join(args.save_dir, "best_model.pt"), map_location=args.device
        )
    )
    test_metrics = validate(
        model, test_loader, args.device, normalizer, non_blocking=non_blocking
    )
    logger.info(f"Test: loss={test_metrics['loss']:.4f}, mae={test_metrics['mae']:.4f}")
    predictions = collect_predictions(
        model, test_loader, args.device, non_blocking=non_blocking
    )
    evaluation = evaluate_elasticity_predictions(
        predictions,
        arm=args.arm,
        student_t_dof=args.student_t_dof,
        seed=args.seed,
    )
    evaluation["mae_gpa"] = float(test_metrics["mae"])
    evaluation["selected_epoch"] = selected_epoch
    evaluation["runtime"] = {
        "wall_seconds": elapsed_seconds,
        "examples_per_second": (
            len(train_loader.dataset) * len(history) / max(elapsed_seconds, 1e-12)
        ),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(args.device) / 1024**3 if using_cuda else 0.0
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(args.device) / 1024**3 if using_cuda else 0.0
        ),
    }

    run_dir = Path(args.save_dir)
    compact_predictions = {
        name: tensor
        for name, tensor in predictions.items()
        if name != "scale"
    }
    torch.save(compact_predictions, run_dir / "predictions.pt")
    schema_valid = (
        schema.get("kind") == "deterministic_mean"
        if model.distribution is None
        else bool(schema["family"]["certificates"]["valid"])
        and bool(schema["representation_reachability"]["active"]["reachable"])
        and schema["covariance_representation"]["highest_angular_momentum"] == 8
    )
    schema_record = {"schema_valid": schema_valid, "compiler": schema}
    data_files = sorted(Path(args.data_dir).glob("*.pkl"))
    environment = {
        "source": source_provenance(Path(__file__).resolve().parents[1]),
        "data_files": {path.name: sha256_file(path) for path in data_files},
        "split": {
            "seed": args.seed,
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
        },
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    payloads = {
        "args.json": vars(args),
        "environment.json": environment,
        "schema.json": schema_record,
        "history.json": history,
        "metrics.json": evaluation,
    }
    for name, payload in payloads.items():
        with (run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
