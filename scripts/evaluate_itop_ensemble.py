"""Evaluate a three-seed deterministic ITOP ensemble without extra training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.itop_dataset import ITOP_OUTPUT_GRAPH
from evaluation import (
    binary_auroc,
    bone_length_error,
    joint_errors,
    joint_residual_correlation,
    residual_correlation_by_graph_distance,
    risk_coverage_auc,
    visible_occluded_mpjpe,
)


def _load_members(
    run_dirs: tuple[Path, ...], view: str
) -> list[dict[str, torch.Tensor]]:
    records = [
        torch.load(
            run_dir / f"predictions_{view}.pt", map_location="cpu", weights_only=True
        )
        for run_dir in run_dirs
    ]
    reference = records[0]
    for record in records[1:]:
        if not torch.equal(record["frame_index"], reference["frame_index"]):
            raise ValueError("ensemble members use different frame ordering")
        if not torch.equal(record["view_id"], reference["view_id"]):
            raise ValueError("ensemble members use different views")
        torch.testing.assert_close(
            record["target"], reference["target"], rtol=0, atol=0
        )
        if not torch.equal(record["visible_joints"], reference["visible_joints"]):
            raise ValueError("ensemble members use different visibility records")
    return records


def _evaluate(records: list[dict[str, torch.Tensor]]) -> tuple[dict[str, Any], dict]:
    target = records[0]["target"]
    visible = records[0]["visible_joints"].bool()
    draws = torch.stack([record["mean"] for record in records], dim=1)
    mean = draws.mean(dim=1)
    errors = joint_errors(mean, target)
    centered = draws - mean[:, None, :]
    joint_uncertainty = (
        centered.reshape(len(mean), len(records), 15, 3).square().sum(-1).mean(1)
    )
    frame_uncertainty = joint_uncertainty.sum(-1)
    fit = torch.linalg.vector_norm(draws - target[:, None, :], dim=-1).mean(-1)
    diversity = torch.linalg.vector_norm(
        draws[:, :, None, :] - draws[:, None, :, :], dim=-1
    ).mean(dim=(-2, -1))
    correlation = joint_residual_correlation(mean, target)
    by_distance = residual_correlation_by_graph_distance(
        correlation, ITOP_OUTPUT_GRAPH.edges
    )
    metrics: dict[str, Any] = {
        "members": len(records),
        "mpjpe_cm": float(errors.mean().item() * 100.0),
        "energy_score_m": float((fit - 0.5 * diversity).mean().item()),
        "frame_risk_coverage_auc_cm": float(
            risk_coverage_auc(frame_uncertainty, errors.mean(-1)).item() * 100.0
        ),
        "joint_risk_coverage_auc_cm": float(
            risk_coverage_auc(joint_uncertainty.flatten(), errors.flatten()).item()
            * 100.0
        ),
        "occluded_visible_variance_ratio": float(
            (
                joint_uncertainty[~visible].mean() / joint_uncertainty[visible].mean()
            ).item()
        ),
        "sample_bone_length_error_cm": float(
            bone_length_error(draws, target, ITOP_OUTPUT_GRAPH.edges).item() * 100.0
        ),
        "adjacent_joint_residual_correlation": by_distance["1"],
        "residual_correlation_by_skeleton_distance": by_distance,
        "nll": None,
        "mace": None,
        "probability_note": "finite deterministic ensemble is not treated as a continuous density",
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
    artifact = {
        "mean": mean,
        "members": draws,
        "target": target,
        "visible_joints": visible,
        "frame_uncertainty": frame_uncertainty,
        "joint_uncertainty": joint_uncertainty,
        "joint_errors": errors,
        "frame_index": records[0]["frame_index"],
        "view_id": records[0]["view_id"],
    }
    return metrics, artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs=3, type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_dirs = tuple(args.run_dirs)
    side_metrics, side = _evaluate(_load_members(run_dirs, "side"))
    top_metrics, top = _evaluate(_load_members(run_dirs, "top"))
    scores = torch.cat((side["frame_uncertainty"], top["frame_uncertainty"]))
    labels = torch.cat(
        (
            torch.zeros_like(side["frame_uncertainty"], dtype=torch.long),
            torch.ones_like(top["frame_uncertainty"], dtype=torch.long),
        )
    )
    side_uncertainty = side["frame_uncertainty"].mean()
    top_uncertainty = top["frame_uncertainty"].mean()
    ood = {
        "side_to_top_mpjpe_cm": top_metrics["mpjpe_cm"],
        "side_top_uncertainty_auroc": binary_auroc(scores, labels),
        "ood_uncertainty_increase": float((top_uncertainty - side_uncertainty).item()),
        "ood_uncertainty_ratio": float((top_uncertainty / side_uncertainty).item()),
        "cross_view_frame_risk_coverage_auc_cm": top_metrics[
            "frame_risk_coverage_auc_cm"
        ],
    }
    result = {
        "kind": "three_seed_deterministic_ensemble",
        "member_run_dirs": [str(path) for path in run_dirs],
        "side": side_metrics,
        "top": top_metrics,
        "ood": ood,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(side, args.output_dir / "predictions_side.pt")
    torch.save(top, args.output_dir / "predictions_top.pt")


if __name__ == "__main__":
    main()
