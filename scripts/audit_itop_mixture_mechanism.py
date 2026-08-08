"""No-training mechanism audit for frozen ITOP finite-mixture artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from scipy.stats import spearmanr

from data.observation_descriptors import point_cloud_observation_descriptors
from evaluation import finite_mixture_log_prob
from scripts.itop_reproducibility import (
    atomic_write_json,
    sha256_file,
    source_provenance,
)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().cpu()
    return {
        "mean": float(values.mean()),
        "q10": float(torch.quantile(values, 0.10)),
        "median": float(torch.quantile(values, 0.50)),
        "q90": float(torch.quantile(values, 0.90)),
    }


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    value = float(
        spearmanr(
            left.detach().double().cpu().numpy(),
            right.detach().double().cpu().numpy(),
        ).statistic
    )
    return value if math.isfinite(value) else 0.0


def _prediction(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"invalid prediction artifact: {path}")
    return payload


def _audit_split(
    fixed: dict[str, torch.Tensor],
    mixture: dict[str, torch.Tensor],
    descriptors: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    def close(left: torch.Tensor, right: torch.Tensor) -> bool:
        return bool(torch.allclose(left, right, rtol=1e-6, atol=1e-7))

    required_equal = {
        "sample_id": torch.equal(fixed["sample_id"], mixture["sample_id"]),
        "mean": torch.equal(fixed["mean"], mixture["mean"]),
        "target": torch.equal(fixed["target"], mixture["target"]),
        "shared_scatter_component_0": torch.equal(
            fixed["scale"], mixture["component_scales"][0]
        ),
        "shared_scatter_component_1": torch.equal(
            fixed["scale"], mixture["component_scales"][1]
        ),
        "symmetric_component_mean": close(
            mixture["component_means"].mean(0), mixture["mean"]
        ),
        "fixed_half_weights": torch.equal(
            mixture["weights"], torch.full_like(mixture["weights"], 0.5)
        ),
        "geometry_sample_ids": torch.equal(
            mixture["sample_id"], descriptors["sample_id"]
        ),
    }
    if not all(required_equal.values()):
        raise ValueError(
            f"E2 mechanism alignment/leakage check failed: {required_equal}"
        )

    means = mixture["component_means"].to(device)
    scales = mixture["component_scales"].to(device)
    target = mixture["target"].to(device)
    weights = mixture["weights"].to(device)
    mixture_density = finite_mixture_log_prob(
        means,
        scales,
        target,
        distribution="student_t",
        student_t_dof=5.0,
        weights=weights,
    )
    fixed_density = finite_mixture_log_prob(
        fixed["mean"].unsqueeze(0).to(device),
        fixed["scale"].unsqueeze(0).to(device),
        target,
        distribution="student_t",
        student_t_dof=5.0,
    )
    responsibilities = mixture_density["responsibilities"].cpu()
    nll_gain = (mixture_density["log_prob"] - fixed_density["log_prob"]).cpu()

    delta = mixture["delta"].float()
    residual = mixture["target"].float() - mixture["mean"].float()
    delta_device = delta.to(device)
    chol = torch.linalg.cholesky(fixed["scale"].float().to(device))
    solved = torch.cholesky_solve(delta_device.unsqueeze(-1), chol).squeeze(-1)
    separation = (delta_device * solved).sum(-1).cpu()
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    denominator = (delta_norm * residual_norm).clamp_min(1e-12)
    signed_dot = (delta * residual).sum(-1)
    absolute_cosine = signed_dot.abs() / denominator
    assigned_sign = torch.where(responsibilities[0] >= responsibilities[1], 1.0, -1.0)
    assigned_cosine = assigned_sign * signed_dot / denominator
    entropy = -(responsibilities.clamp_min(1e-12).log() * responsibilities).sum(0)
    residual_severity = residual.reshape(-1, 15, 3).square().sum(-1).mean(-1)

    correlation_inputs = {
        "residual_severity": residual_severity,
        "residual_norm": residual_norm,
        "mode_axis_mahalanobis2": separation,
        "assignment_entropy": entropy,
        **{name: values for name, values in descriptors.items() if name != "sample_id"},
    }
    return {
        "alignment_and_leakage_checks": required_equal,
        "mode_axis_mahalanobis2": _quantiles(separation),
        "component_mean_mahalanobis2": _quantiles(4.0 * separation),
        "responsibilities": {
            "assignment_entropy": _quantiles(entropy),
            "normalized_entropy_mean": float(entropy.mean() / math.log(2.0)),
            "maximum_responsibility": _quantiles(responsibilities.max(0).values),
            "plus_assignment_fraction": float(
                (responsibilities[0] >= 0.5).float().mean()
            ),
        },
        "delta_residual_relation": {
            "delta_norm": _quantiles(delta_norm),
            "residual_norm": _quantiles(residual_norm),
            "delta_to_residual_norm_ratio": _quantiles(
                delta_norm / residual_norm.clamp_min(1e-12)
            ),
            "absolute_axis_cosine": _quantiles(absolute_cosine),
            "posterior_assigned_cosine": _quantiles(assigned_cosine),
        },
        "per_sample_nll_gain": {
            **_quantiles(nll_gain),
            "positive_fraction": float((nll_gain > 0).float().mean()),
        },
        "nll_gain_spearman": {
            name: _spearman(nll_gain, values)
            for name, values in correlation_inputs.items()
        },
        "separation_spearman": {
            name: _spearman(separation, values)
            for name, values in correlation_inputs.items()
            if name != "mode_axis_mahalanobis2"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_run", type=Path, required=True)
    parser.add_argument("--mixture_run", type=Path, action="append", required=True)
    parser.add_argument("--side_geometry", type=Path, required=True)
    parser.add_argument("--top_geometry", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite mechanism audit: {args.output_dir}"
        )
    device = torch.device(args.device)
    descriptors = {
        "test": point_cloud_observation_descriptors(args.side_geometry, view_id=0),
        "ood": point_cloud_observation_descriptors(args.top_geometry, view_id=1),
    }
    fixed_protocol = json.loads((args.fixed_run / "protocol.json").read_text())
    if fixed_protocol["selection"]["ood_used_for_selection"]:
        raise ValueError("fixed artifact used OOD for selection")
    record: dict[str, Any] = {
        "schema_version": 1,
        "study": "E2 no-training ITOP K=2 mechanism audit",
        "interpretation_boundary": (
            "mode-axis diagnostics distinguish density-mass placement from evidence "
            "of physical or true conditional multimodality"
        ),
        "fixed_run": {
            "path": str(args.fixed_run.resolve()),
            "checkpoint_sha256": sha256_file(args.fixed_run / "best_model.pt"),
        },
        "geometry": {
            "side": str(args.side_geometry.resolve()),
            "top": str(args.top_geometry.resolve()),
            "descriptor_inputs": [
                name
                for name in descriptors["test"]
                if name not in {"sample_id", "visible_fraction_diagnostic_only"}
            ],
            "visibility_role": "diagnostic_only_not_probe_input",
        },
        "runs": {},
        "source": source_provenance(Path(__file__).resolve().parents[1]),
    }
    for run in args.mixture_run:
        protocol = json.loads((run / "protocol.json").read_text())
        if protocol["selection"]["ood_used_for_selection"]:
            raise ValueError(f"mixture artifact used OOD for selection: {run}")
        name = f"seed_{protocol['seed']}"
        split_results = {}
        for split in ("test", "ood"):
            split_results[split] = _audit_split(
                _prediction(args.fixed_run / f"predictions_{split}.pt"),
                _prediction(run / f"predictions_{split}.pt"),
                descriptors[split],
                device=device,
            )
        record["runs"][name] = {
            "path": str(run.resolve()),
            "checkpoint_sha256": sha256_file(run / "best_model.pt"),
            "selection": protocol["selection"],
            "splits": split_results,
        }
    args.output_dir.mkdir(parents=True)
    atomic_write_json(record, args.output_dir / "mechanism_audit.json")
    print(json.dumps(record["runs"], indent=2))


if __name__ == "__main__":
    main()
