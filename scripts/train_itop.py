"""Train and evaluate graph-structured equivariant uncertainty on ITOP."""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from data.itop_dataset import ITOP_OUTPUT_GRAPH, get_itop_loaders
from evaluation import (
    bone_length_error,
    calibration_absolute_error,
    energy_score,
    joint_errors,
    joint_mahalanobis_squared,
    marginal_joint_covariances,
    occlusion_uncertainty_ratio,
    risk_coverage_auc,
    visible_occluded_mpjpe,
)
from models import EquivariantBackbone
from representations import CompilerConfig, O3RepresentationCompiler


def _logger(save_dir: Path) -> logging.Logger:
    save_dir.mkdir(parents=True, exist_ok=True)
    name = f"itop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (
        logging.FileHandler(save_dir / f"{name}.log", mode="w"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _protocol_views(protocol: str) -> tuple[str, str]:
    return {
        "side": ("side", "side"),
        "top": ("top", "top"),
        "side_to_top": ("side", "top"),
        "top_to_side": ("top", "side"),
    }[protocol]


def train_epoch(model, loader, optimizer, device: str) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch, target=batch.y_pose)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = batch.y_pose.shape[0]
        total += result["loss"].detach().item() * batch_size
        count += batch_size
    return total / max(count, 1)


@torch.inference_mode()
def evaluate(model, loader, device: str, *, detailed: bool = False) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    means: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    visibility: list[torch.Tensor] = []
    covariances: list[torch.Tensor] = []
    frame_mahalanobis: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = batch.to(device)
        result = model(batch, target=batch.y_pose, return_scale=detailed)
        batch_size = batch.y_pose.shape[0]
        total_loss += result["loss"].item() * batch_size
        count += batch_size
        means.append(result["mu"].cpu())
        targets.append(batch.y_pose.cpu())
        visibility.append(batch.visible_joints.cpu())
        if detailed:
            covariance = result["scale"]
            covariances.append(covariance.cpu())
            residual = batch.y_pose - result["mu"]
            frame_mahalanobis.append(
                model.spd_map.precision_action(result["params"], residual).cpu()
            )

    mean = torch.cat(means)
    target = torch.cat(targets)
    visible = torch.cat(visibility).bool()
    errors = joint_errors(mean, target)
    metrics = {
        "nll": total_loss / max(count, 1),
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
    if not detailed:
        return metrics

    covariance = torch.cat(covariances)
    frame_maha2 = torch.cat(frame_mahalanobis)
    joint_maha2 = joint_mahalanobis_squared(mean, target, covariance)
    marginal = marginal_joint_covariances(covariance)
    frame_uncertainty = torch.linalg.slogdet(covariance).logabsdet
    joint_uncertainty = torch.diagonal(marginal, dim1=-2, dim2=-1).sum(-1)
    metrics.update(
        {
            "energy_score_m": float(energy_score(mean, covariance, target, 32).item()),
            "frame_calibration_ace": calibration_absolute_error(frame_maha2, 45),
            "joint_calibration_ace": calibration_absolute_error(joint_maha2, 3),
            "frame_risk_coverage_auc_cm": float(
                risk_coverage_auc(frame_uncertainty, errors.mean(-1)).item() * 100.0
            ),
            "joint_risk_coverage_auc_cm": float(
                risk_coverage_auc(joint_uncertainty.flatten(), errors.flatten()).item()
                * 100.0
            ),
            "occlusion_uncertainty_ratio": occlusion_uncertainty_ratio(
                covariance, visible
            ),
        }
    )
    cholesky = torch.linalg.cholesky(covariance)
    noise = torch.randn(mean.shape[0], 16, 45)
    samples = mean[:, None, :] + torch.einsum("bij,bnj->bni", cholesky, noise)
    metrics["sample_bone_length_error_cm"] = float(
        bone_length_error(samples, target, ITOP_OUTPUT_GRAPH.edges).item() * 100.0
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/itop")
    parser.add_argument(
        "--protocol",
        choices=["side", "top", "side_to_top", "top_to_side"],
        default="side",
    )
    parser.add_argument("--save_dir", default="checkpoints_itop")
    parser.add_argument("--covariance", choices=["auto", "graph", "full", "block", "low_rank"], default="auto")
    parser.add_argument("--objective", choices=["gaussian", "student_t"], default="gaussian")
    parser.add_argument("--student_t_dof", type=float, default=5.0)
    parser.add_argument("--parameter_budget", type=int, default=192)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_basis", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_neighbors", type=int, default=16)
    parser.add_argument("--max_radius", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--depth_noise_std", type=float, default=0.0)
    parser.add_argument("--point_dropout", type=float, default=0.0)
    parser.add_argument("--occlusion_fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    save_dir = Path(args.save_dir)
    logger = _logger(save_dir)
    train_view, test_view = _protocol_views(args.protocol)
    train_loader, validation_loader, test_loader = get_itop_loaders(
        args.data_dir,
        train_view=train_view,
        test_view=test_view,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_neighbors=args.num_neighbors,
        max_radius=args.max_radius,
        num_basis=args.num_basis,
        lmax=args.lmax,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        depth_noise_std=args.depth_noise_std,
        point_dropout=args.point_dropout,
        occlusion_fraction=args.occlusion_fraction,
    )

    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        num_basis=args.num_basis,
        atom_feature_dim=32,
        atom_features="learnable",
    )
    compiler = O3RepresentationCompiler.for_graph(
        ITOP_OUTPUT_GRAPH,
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
    logger.info("protocol=%s, train_view=%s, test_view=%s", args.protocol, train_view, test_view)
    logger.info("compilation=%s", json.dumps(compilation.as_dict()))
    logger.info("parameters=%d", sum(parameter.numel() for parameter in model.parameters()))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    best = math.inf
    stale = 0
    history: list[dict] = []
    for epoch in range(1, args.num_epochs + 1):
        train_nll = train_epoch(model, train_loader, optimizer, args.device)
        validation = evaluate(model, validation_loader, args.device)
        scheduler.step(validation["nll"])
        record = {"epoch": epoch, "train_nll": train_nll, **validation}
        history.append(record)
        logger.info("epoch=%d %s", epoch, json.dumps(record))
        if validation["nll"] < best:
            best = validation["nll"]
            stale = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(
        torch.load(save_dir / "best_model.pt", map_location=args.device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, args.device, detailed=True)
    logger.info("test=%s", json.dumps(test_metrics))
    for name, payload in (
        ("args.json", vars(args)),
        ("compilation.json", compilation.as_dict()),
        ("history.json", history),
        ("test_metrics.json", test_metrics),
    ):
        with (save_dir / name).open("w", encoding="utf-8") as target_file:
            json.dump(payload, target_file, indent=2)


if __name__ == "__main__":
    main()
