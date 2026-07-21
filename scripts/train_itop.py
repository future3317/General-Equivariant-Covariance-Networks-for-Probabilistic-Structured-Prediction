"""Single-GPU two-stage ITOP study for structured equivariant uncertainty."""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from data.itop_dataset import (
    ITOP_INDEPENDENT_GRAPH,
    ITOP_OUTPUT_GRAPH,
    get_itop_loaders,
    get_itop_test_loader,
)
from data.itop_features import get_itop_feature_loaders
from data.paths import dataset_dir
from equivcompiler import FeatureSpec, GraphPrecision, plan_readout
from evaluation import (
    binary_auroc,
    bone_length_error,
    calibration_absolute_error,
    joint_errors,
    joint_mahalanobis_squared,
    joint_residual_correlation,
    marginal_joint_covariances,
    per_joint_marginal_coverage,
    residual_correlation_by_graph_distance,
    risk_coverage_auc,
    visible_occluded_mpjpe,
)
from models import (
    BaselineProbabilisticPredictor,
    ControlledMeanOperatorHead,
    DeterministicHead,
    EquivariantBackbone,
)
from representations import O3IrrepsSpec
from scripts._common import add_tensor_product_arguments, tensor_product_kwargs


MODEL_KINDS = (
    "deterministic",
    "independent_gaussian",
    "graph_gaussian",
    "graph_student_t",
)
PHASES = ("deterministic", "frozen_head", "joint_finetune")


def _logger(run_dir: Path, *, continuing: bool) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=continuing)
    name = f"itop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (
        logging.FileHandler(run_dir / "train.log", mode="a" if continuing else "w"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def build_itop_backbone(args: argparse.Namespace) -> EquivariantBackbone:
    return EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        max_radius=args.max_radius,
        lmax=args.lmax,
        num_layers=args.num_layers,
        num_basis=args.num_basis,
        atom_feature_dim=32,
        atom_features="learnable",
        **tensor_product_kwargs(args),
    )


def _build_model(args: argparse.Namespace):
    backbone = build_itop_backbone(args)
    output = O3IrrepsSpec(ITOP_OUTPUT_GRAPH.output_irreps)
    if args.model == "deterministic":
        head = DeterministicHead(backbone.irreps_out, output, pool=True)
        model = BaselineProbabilisticPredictor(
            backbone,
            output,
            head,
            spd_map=None,
            distribution=None,
        )
        return model, None

    graph = (
        ITOP_INDEPENDENT_GRAPH
        if args.model == "independent_gaussian"
        else ITOP_OUTPUT_GRAPH
    )
    objective = "student_t" if args.model == "graph_student_t" else "gaussian"
    plan = plan_readout(
        FeatureSpec.from_backbone(backbone),
        output=ITOP_OUTPUT_GRAPH.output_irreps,
        covariance=GraphPrecision(graph),
        distribution=objective,
        student_t_dof=args.student_t_dof,
        output_scope="global",
    )
    model = plan.bind(backbone)
    model.joint_head = ControlledMeanOperatorHead(
        DeterministicHead(backbone.irreps_out, output, pool=True),
        model.joint_head,
    )
    return model, plan


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("ITOP checkpoint must contain a dictionary")
    return payload


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _configure_initialization(model, args: argparse.Namespace) -> bool:
    if args.phase == "deterministic":
        if args.model != "deterministic":
            raise ValueError("phase=deterministic requires model=deterministic")
        return False
    if args.model == "deterministic":
        raise ValueError("probabilistic phases require a probabilistic model")
    if args.phase == "frozen_head":
        if args.backbone_checkpoint is None:
            raise ValueError("frozen_head requires --backbone_checkpoint")
        payload = _load_checkpoint(args.backbone_checkpoint)
        model.backbone.load_state_dict(payload["backbone_state"], strict=True)
        model.joint_head.mean_head.load_state_dict(
            payload["mean_head_state"], strict=True
        )
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in model.joint_head.mean_head.parameters():
            parameter.requires_grad_(False)
        return True
    if args.resume_checkpoint is None:
        raise ValueError("joint_finetune requires --resume_checkpoint")
    payload = _load_checkpoint(args.resume_checkpoint)
    model.load_state_dict(payload["model_state"], strict=True)
    return False


def _forward(
    model,
    batch,
    *,
    target: torch.Tensor | None,
    return_scale: bool,
    use_bf16: bool,
):
    if isinstance(batch, dict):
        features = batch["features"].float()
        graph_batch = torch.arange(features.shape[0], device=features.device)
        return model.forward_from_features(
            features,
            graph_batch,
            target=target,
            return_scale=return_scale,
        )
    enabled = use_bf16 and batch.pos.device.type == "cuda"
    with torch.autocast(
        device_type=batch.pos.device.type, dtype=torch.bfloat16, enabled=enabled
    ):
        node_features, graph_batch = model.backbone(batch)
    return model.forward_from_features(
        node_features.float(),
        graph_batch,
        target=target,
        return_scale=return_scale,
    )


def _to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {
            name: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in batch.items()
        }
    return batch.to(device, non_blocking=True)


def _batch_field(batch, name: str) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch[name]
    return getattr(batch, name)


def train_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    *,
    frozen_backbone: bool,
    use_bf16: bool,
) -> float:
    model.train()
    if frozen_backbone:
        model.backbone.eval()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = _to_device(batch, device)
        target = _batch_field(batch, "target" if isinstance(batch, dict) else "y_pose")
        optimizer.zero_grad(set_to_none=True)
        result = _forward(
            model,
            batch,
            target=target,
            return_scale=False,
            use_bf16=use_bf16,
        )
        loss = result["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        batch_size = target.shape[0]
        total += float(loss.detach()) * batch_size
        count += batch_size
    return total / max(count, 1)


def _energy_and_bone_error(
    mean: torch.Tensor,
    scatter: torch.Tensor,
    target: torch.Tensor,
    *,
    samples: int,
    student_t_dof: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cholesky = torch.linalg.cholesky(scatter)
    noise = torch.randn(
        mean.shape[0], samples, mean.shape[-1], device=mean.device, dtype=mean.dtype
    )
    draws = torch.einsum("bij,bnj->bni", cholesky, noise)
    if student_t_dof is not None:
        chi2 = (
            torch.distributions.Chi2(student_t_dof)
            .sample((mean.shape[0], samples))
            .to(device=mean.device, dtype=mean.dtype)
        )
        draws = draws * torch.sqrt(student_t_dof / chi2).unsqueeze(-1)
    draws = mean[:, None, :] + draws
    fit = torch.linalg.vector_norm(draws - target[:, None, :], dim=-1).mean(-1)
    diversity = torch.linalg.vector_norm(
        draws[:, :, None, :] - draws[:, None, :, :], dim=-1
    ).mean(dim=(-2, -1))
    score = fit - 0.5 * diversity
    bone = bone_length_error(draws, target, ITOP_OUTPUT_GRAPH.edges)
    return score.mean(), bone


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device: torch.device,
    *,
    model_kind: str,
    student_t_dof: float,
    detailed: bool,
    use_bf16: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    total_loss = 0.0
    total_energy = 0.0
    total_bone = 0.0
    count = 0
    means: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    visibility: list[torch.Tensor] = []
    scatters: list[torch.Tensor] = []
    covariances: list[torch.Tensor] = []
    frame_mahalanobis: list[torch.Tensor] = []
    parameters: list[torch.Tensor] = []
    frame_indices: list[torch.Tensor] = []
    view_ids: list[torch.Tensor] = []
    is_student = model_kind == "graph_student_t"

    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = _to_device(batch, device)
        target_batch = _batch_field(
            batch, "target" if isinstance(batch, dict) else "y_pose"
        )
        visible_batch = _batch_field(batch, "visible_joints")
        result = _forward(
            model,
            batch,
            target=target_batch,
            return_scale=detailed and model_kind != "deterministic",
            use_bf16=use_bf16,
        )
        batch_size = target_batch.shape[0]
        total_loss += float(result["loss"]) * batch_size
        count += batch_size
        means.append(result["mu"].float().cpu())
        targets.append(target_batch.cpu())
        visibility.append(visible_batch.cpu())
        frame_indices.append(_batch_field(batch, "frame_index").cpu())
        view_ids.append(_batch_field(batch, "view_id").cpu())
        if detailed and model_kind != "deterministic":
            scatter = result["scale"].float()
            covariance = (
                (student_t_dof / (student_t_dof - 2.0)) * scatter
                if is_student
                else scatter
            )
            scatters.append(scatter.cpu())
            covariances.append(covariance.cpu())
            parameters.append(result["params"].float().cpu())
            residual = target_batch - result["mu"].float()
            frame_mahalanobis.append(
                model.spd_map.precision_action(result["params"].float(), residual).cpu()
            )
            energy, bone = _energy_and_bone_error(
                result["mu"].float(),
                scatter,
                target_batch,
                samples=32,
                student_t_dof=student_t_dof if is_student else None,
            )
            total_energy += float(energy) * batch_size
            total_bone += float(bone) * batch_size

    mean = torch.cat(means)
    target = torch.cat(targets)
    visible = torch.cat(visibility).bool()
    errors = joint_errors(mean, target)
    metrics: dict[str, Any] = {
        "loss": total_loss / max(count, 1),
        "mpjpe_cm": float(errors.mean().item() * 100.0),
    }
    for centimeters in (5, 10, 15):
        metrics[f"pck_{centimeters}cm"] = float(
            (errors <= centimeters / 100.0).float().mean().item()
        )
    metrics.update(
        {
            f"{key.removesuffix('_m')}_cm": value * 100.0
            for key, value in visible_occluded_mpjpe(mean, target, visible).items()
        }
    )
    correlation = joint_residual_correlation(mean, target)
    distance_correlation = residual_correlation_by_graph_distance(
        correlation, ITOP_OUTPUT_GRAPH.edges
    )
    metrics["adjacent_joint_residual_correlation"] = distance_correlation["1"]
    metrics["residual_correlation_by_skeleton_distance"] = distance_correlation

    artifact = {
        "mean": mean,
        "target": target,
        "visible_joints": visible,
        "joint_errors": errors,
        "residual_correlation": correlation,
        "frame_index": torch.cat(frame_indices),
        "view_id": torch.cat(view_ids),
    }
    if not detailed or model_kind == "deterministic":
        return metrics, artifact

    scatter = torch.cat(scatters)
    covariance = torch.cat(covariances)
    frame_maha2 = torch.cat(frame_mahalanobis)
    joint_maha2 = joint_mahalanobis_squared(mean, target, scatter)
    marginal = marginal_joint_covariances(covariance)
    frame_uncertainty = torch.linalg.slogdet(covariance).logabsdet
    joint_uncertainty = torch.diagonal(marginal, dim1=-2, dim2=-1).sum(-1)
    calibration_dof = student_t_dof if is_student else None
    coverage = per_joint_marginal_coverage(
        joint_maha2,
        student_t_dof=calibration_dof,
    )
    metrics.update(
        {
            "nll": metrics.pop("loss"),
            "energy_score_m": total_energy / max(count, 1),
            "mace": calibration_absolute_error(
                frame_maha2,
                45,
                student_t_dof=calibration_dof,
            ),
            "joint_mace": coverage["mace"],
            "per_joint_marginal_coverage": coverage,
            "frame_risk_coverage_auc_cm": float(
                risk_coverage_auc(frame_uncertainty, errors.mean(-1)).item() * 100.0
            ),
            "joint_risk_coverage_auc_cm": float(
                risk_coverage_auc(joint_uncertainty.flatten(), errors.flatten()).item()
                * 100.0
            ),
            "occluded_visible_variance_ratio": float(
                (
                    joint_uncertainty[~visible].mean()
                    / joint_uncertainty[visible].mean()
                ).item()
            ),
            "sample_bone_length_error_cm": total_bone / max(count, 1) * 100.0,
        }
    )
    artifact.update(
        {
            "params": torch.cat(parameters),
            "frame_uncertainty": frame_uncertainty,
            "joint_uncertainty": joint_uncertainty,
            "frame_mahalanobis2": frame_maha2,
            "joint_mahalanobis2": joint_maha2,
        }
    )
    return metrics, artifact


def _ood_metrics(
    side_metrics: dict[str, Any],
    side: dict[str, torch.Tensor],
    top_metrics: dict[str, Any],
    top: dict[str, torch.Tensor],
) -> dict[str, float]:
    if "frame_uncertainty" not in side or "frame_uncertainty" not in top:
        return {}
    scores = torch.cat((side["frame_uncertainty"], top["frame_uncertainty"]))
    labels = torch.cat(
        (
            torch.zeros_like(side["frame_uncertainty"], dtype=torch.long),
            torch.ones_like(top["frame_uncertainty"], dtype=torch.long),
        )
    )
    side_mean = side["frame_uncertainty"].mean()
    top_mean = top["frame_uncertainty"].mean()
    return {
        "side_to_top_mpjpe_cm": float(top_metrics["mpjpe_cm"]),
        "side_to_top_nll": float(top_metrics["nll"]),
        "side_top_uncertainty_auroc": binary_auroc(scores, labels),
        "ood_logdet_uncertainty_increase": float((top_mean - side_mean).item()),
        "ood_logdet_uncertainty_ratio": float(
            torch.exp((top_mean - side_mean) / 45.0).item()
        ),
        "cross_view_frame_risk_coverage_auc_cm": float(
            top_metrics["frame_risk_coverage_auc_cm"]
        ),
    }


def _checkpoint_payload(model, args, epoch: int, validation: dict[str, Any]) -> dict:
    mean_head = (
        model.joint_head
        if args.model == "deterministic"
        else model.joint_head.mean_head
    )
    return {
        "schema_version": 1,
        "model_kind": args.model,
        "phase": args.phase,
        "seed": args.seed,
        "epoch": epoch,
        "validation": validation,
        "model_state": model.state_dict(),
        "backbone_state": model.backbone.state_dict(),
        "mean_head_state": mean_head.state_dict(),
        "args": vars(args),
    }


def _last_state_payload(
    model,
    optimizer,
    scheduler,
    args,
    *,
    next_epoch: int,
    best: float,
    stale: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args),
        "next_epoch": next_epoch,
        "best": best,
        "stale": stale,
        "history": history,
    }


def _validate_resume_args(saved: dict[str, Any], current: argparse.Namespace) -> None:
    ignored = {"continue_run", "num_epochs"}
    mismatches = {
        name: {"saved": value, "current": vars(current).get(name)}
        for name, value in saved.items()
        if name not in ignored and vars(current).get(name) != value
    }
    if mismatches:
        raise ValueError(f"resume arguments do not match run contract: {mismatches}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--continue_run",
        action="store_true",
        help="resume an interrupted run from run_dir/last_state.pt",
    )
    parser.add_argument("--model", choices=MODEL_KINDS, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--backbone_checkpoint")
    parser.add_argument("--resume_checkpoint")
    parser.add_argument(
        "--feature_cache",
        help="pooled frozen-backbone cache; valid only for phase=frozen_head",
    )
    parser.add_argument("--student_t_dof", type=float, default=5.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--num_points", type=int, choices=(256, 512), default=512)
    parser.add_argument("--num_neighbors", type=int, default=16)
    parser.add_argument("--max_radius", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone_precision", choices=("bf16", "fp32"), default="bf16"
    )
    parser.add_argument("--no_cache", action="store_true")
    add_tensor_product_arguments(parser)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.model == "graph_student_t" and args.student_t_dof <= 2.0:
        raise ValueError(
            "ITOP variance metrics require Student-t degrees of freedom > 2"
        )
    args.data_dir = str(dataset_dir(args.data_dir, "ITOP"))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    _set_seed(args.seed)
    run_dir = Path(args.run_dir)
    logger = _logger(run_dir, continuing=args.continue_run)
    model, plan = _build_model(args)
    frozen_backbone = _configure_initialization(model, args)
    model = model.to(device)
    if args.compile_tp:
        model.backbone.compile_tensor_products(dynamic=True)
    use_bf16 = args.backbone_precision == "bf16"

    loader_kwargs = dict(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_neighbors=args.num_neighbors,
        max_radius=args.max_radius,
        num_basis=args.num_basis,
        lmax=args.lmax,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor,
        use_cache=not args.no_cache,
    )
    feature_cache_metadata = None
    if args.feature_cache is not None:
        if args.phase != "frozen_head":
            raise ValueError("--feature_cache is valid only for phase=frozen_head")
        (
            train_loader,
            validation_loader,
            side_loader,
            top_loader,
            feature_cache_metadata,
        ) = get_itop_feature_loaders(
            args.feature_cache,
            backbone_checkpoint=args.backbone_checkpoint,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        train_loader, validation_loader, side_loader = get_itop_loaders(
            train_view="side", test_view="side", seed=args.seed, **loader_kwargs
        )
        top_loader = get_itop_test_loader(view="top", **loader_kwargs)

    compilation = plan.compilation.as_dict() if plan is not None else None
    logger.info("args=%s", json.dumps(vars(args), sort_keys=True))
    logger.info("environment=%s", json.dumps(_environment(device), sort_keys=True))
    logger.info("compilation=%s", json.dumps(compilation, sort_keys=True))
    logger.info("feature_cache=%s", json.dumps(feature_cache_metadata, sort_keys=True))
    logger.info(
        "trainable_parameters=%d",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best = math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_path = run_dir / "best_model.pt"
    last_state_path = run_dir / "last_state.pt"
    if args.continue_run and last_state_path.is_file():
        state = _load_checkpoint(last_state_path)
        _validate_resume_args(state["args"], args)
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["next_epoch"])
        best = float(state["best"])
        stale = int(state["stale"])
        history = list(state["history"])
        logger.info("resume next_epoch=%d best=%.8g stale=%d", start_epoch, best, stale)
    elif args.continue_run:
        logger.info("no last_state.pt found; restarting stage before its first epoch")
    if stale >= args.patience:
        start_epoch = args.num_epochs + 1
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            frozen_backbone=frozen_backbone,
            use_bf16=use_bf16,
        )
        validation, _ = evaluate(
            model,
            validation_loader,
            device,
            model_kind=args.model,
            student_t_dof=args.student_t_dof,
            detailed=False,
            use_bf16=use_bf16,
        )
        criterion = (
            validation["mpjpe_cm"]
            if args.model == "deterministic"
            else validation["loss"]
        )
        scheduler.step(criterion)
        record = {"epoch": epoch, "train_loss": train_loss, **validation}
        history.append(record)
        logger.info("epoch=%d %s", epoch, json.dumps(record, sort_keys=True))
        if criterion < best:
            best = criterion
            stale = 0
            _save_checkpoint(
                _checkpoint_payload(model, args, epoch, validation), best_path
            )
        else:
            stale += 1
        _save_checkpoint(
            _last_state_payload(
                model,
                optimizer,
                scheduler,
                args,
                next_epoch=epoch + 1,
                best=best,
                stale=stale,
                history=history,
            ),
            last_state_path,
        )
        if stale >= args.patience:
            logger.info("early_stop epoch=%d stale=%d", epoch, stale)
            break

    payload = _load_checkpoint(best_path)
    model.load_state_dict(payload["model_state"], strict=True)
    side_metrics, side_artifact = evaluate(
        model,
        side_loader,
        device,
        model_kind=args.model,
        student_t_dof=args.student_t_dof,
        detailed=True,
        use_bf16=use_bf16,
    )
    top_metrics, top_artifact = evaluate(
        model,
        top_loader,
        device,
        model_kind=args.model,
        student_t_dof=args.student_t_dof,
        detailed=True,
        use_bf16=use_bf16,
    )
    ood = _ood_metrics(side_metrics, side_artifact, top_metrics, top_artifact)
    results = {"side": side_metrics, "top": top_metrics, "ood": ood}
    logger.info("results=%s", json.dumps(results, sort_keys=True))
    records = {
        "args.json": vars(args),
        "environment.json": _environment(device),
        "compilation.json": compilation,
        "feature_cache.json": feature_cache_metadata,
        "history.json": history,
        "metrics.json": results,
    }
    for filename, record in records.items():
        (run_dir / filename).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    torch.save(side_artifact, run_dir / "predictions_side.pt")
    torch.save(top_artifact, run_dir / "predictions_top.pt")


if __name__ == "__main__":
    main()
