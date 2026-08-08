"""Evaluate independent ITOP Student-t members as an exact finite mixture."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch
from scipy.stats import spearmanr

from evaluation import (
    binary_auroc,
    energy_score_from_samples,
    finite_mixture_log_prob,
    mixture_projection_pit,
    risk_coverage_auc,
    sample_ensemble,
)
from evaluation.pose import joint_errors
from scripts.itop_reproducibility import (
    atomic_write_json,
    sha256_file,
    source_provenance,
)

STUDENT_T_DOF = 5.0


def _atomic_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sample_id(record: dict[str, torch.Tensor]) -> torch.Tensor:
    return record["view_id"].long() * (1 << 32) + record["frame_index"].long()


def _canonical_contract(record: dict[str, Any]) -> dict[str, Any]:
    contract = json.loads(json.dumps(record["training_contract"]))
    contract["randomness"].pop("seed", None)
    return contract


def _member_record(run_dir: Path) -> dict[str, Any]:
    required = (
        "best_model.pt",
        "last_state.pt",
        "metrics.json",
        "predictions_side.pt",
        "predictions_top.pt",
        "args.json",
        "environment.json",
        "compilation.json",
        "provenance.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete ensemble member {run_dir}: {missing}")
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    compilation = json.loads((run_dir / "compilation.json").read_text(encoding="utf-8"))
    if args["model"] != "full_student_t" or args["phase"] != "end_to_end":
        raise ValueError(f"{run_dir} is not an E3a Full Student-t member")
    if float(args["student_t_dof"]) != STUDENT_T_DOF:
        raise ValueError(f"{run_dir} does not use fixed nu={STUDENT_T_DOF:g}")
    if args.get("backbone_checkpoint") or args.get("resume_checkpoint"):
        raise ValueError(f"{run_dir} has an input checkpoint and is not independent")
    if args.get("feature_cache"):
        raise ValueError(f"{run_dir} uses a frozen feature cache and is not E3a")
    source = environment.get("source")
    if not isinstance(source, dict) or source.get("dirty"):
        raise ValueError(f"{run_dir} lacks clean-source provenance")
    freeze = provenance.get("freeze")
    if not isinstance(freeze, dict) or int(freeze.get("frozen_parameter_count", -1)) != 0:
        raise ValueError(f"{run_dir} did not train all parameters end-to-end")
    artifact_hashes = provenance.get("artifacts", {})
    for name in (
        "best_model.pt",
        "last_state.pt",
        "predictions_side.pt",
        "predictions_top.pt",
    ):
        if artifact_hashes.get(name) != sha256_file(run_dir / name):
            raise ValueError(f"{run_dir} provenance hash mismatch for {name}")
    return {
        "path": str(run_dir.resolve()),
        "seed": int(args["seed"]),
        "args": args,
        "environment": environment,
        "provenance": provenance,
        "compilation": compilation,
        "contract": _canonical_contract(environment),
        "checkpoint_sha256": artifact_hashes["best_model.pt"],
        "checkpoint_chain_sha256": {
            "best_model.pt": artifact_hashes["best_model.pt"],
            "last_state.pt": artifact_hashes["last_state.pt"],
        },
        "prediction_sha256": {
            "side": artifact_hashes["predictions_side.pt"],
            "top": artifact_hashes["predictions_top.pt"],
        },
    }


def _load_split(
    members: list[dict[str, Any]], split: str
) -> list[dict[str, torch.Tensor]]:
    records = [
        torch.load(
            Path(member["path"]) / f"predictions_{split}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for member in members
    ]
    required = {"mean", "target", "scale", "frame_index", "view_id"}
    if any(not required.issubset(record) for record in records):
        raise ValueError(f"member predictions lack Full Student-t fields for {split}")
    reference = records[0]
    for record in records[1:]:
        if not torch.equal(_sample_id(record), _sample_id(reference)):
            raise ValueError(f"member sample IDs differ for {split}")
        if not torch.equal(record["target"], reference["target"]):
            raise ValueError(f"member targets differ for {split}")
    if len(reference["mean"]) != 4863:
        raise ValueError(f"E3a requires 4,863 {split} samples")
    if not all(
        bool(torch.isfinite(record[key]).all())
        for record in records
        for key in ("mean", "target", "scale")
    ):
        raise FloatingPointError(f"non-finite member prediction for {split}")
    return records


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return 0.0
    value = float(
        spearmanr(
            left.detach().double().cpu().numpy(),
            right.detach().double().cpu().numpy(),
        ).statistic
    )
    return value if math.isfinite(value) else 0.0


def _energy(
    means: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int,
    samples: int,
) -> float:
    torch.manual_seed(seed)
    draws = sample_ensemble(
        means,
        scales,
        num_samples=samples,
        distribution="student_t",
        student_t_dof=STUDENT_T_DOF,
    )
    return float(energy_score_from_samples(draws, target).item())


def _subset_metrics(
    means: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    density = finite_mixture_log_prob(
        means,
        scales,
        target,
        distribution="student_t",
        student_t_dof=STUDENT_T_DOF,
    )
    ensemble_mean = means.mean(0)
    return {
        "members": int(means.shape[0]),
        "exact_mixture_nll": float(-density["log_prob"].mean()),
        "nll_semantics": "equal_weight_exact_finite_student_t_logsumexp",
        "ensemble_mean_mpjpe_cm": float(
            joint_errors(ensemble_mean, target).mean().item() * 100.0
        ),
        "energy_score_m": _energy(means, scales, target, seed=seed, samples=samples),
    }


def _spread_metrics(
    means: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    ensemble_mean = means.mean(0)
    centered = means - ensemble_mean.unsqueeze(0)
    between = torch.einsum("mni,mnj->nij", centered, centered) / means.shape[0]
    within = scales.mean(0) * (STUDENT_T_DOF / (STUDENT_T_DOF - 2.0))
    errors = joint_errors(ensemble_mean, target)
    frame_error = errors.mean(-1)
    diagonal = torch.diagonal(between, dim1=-2, dim2=-1)
    trace = diagonal.sum(-1)
    largest_eigenvalue = torch.linalg.eigvalsh(between).amax(-1)
    disagreement = torch.linalg.vector_norm(centered, dim=-1).mean(0)
    joint_spread = diagonal.reshape(-1, 15, 3).sum(-1)
    result = {
        "semantics": "between-member model/function spread diagnostic; not physical aleatoric covariance",
        "within_member_predictive_covariance": {
            "semantics": "mean Student-t covariance nu/(nu-2)*S across members",
            "mean_trace_m2": float(torch.diagonal(within, dim1=-2, dim2=-1).sum(-1).mean()),
        },
        "between_member_model_function_spread": {
            "semantics": "(1/M) sum_m (mu_m-mu_bar)(mu_m-mu_bar)^T",
            "mean_trace_m2": float(trace.mean()),
            "mean_max_eigenvalue_m2": float(largest_eigenvalue.mean()),
            "mean_member_disagreement_m": float(disagreement.mean()),
            "frame_error_spearman": {
                "trace": _spearman(trace, frame_error),
                "max_eigenvalue": _spearman(largest_eigenvalue, frame_error),
                "mean_disagreement": _spearman(disagreement, frame_error),
            },
            "frame_risk_coverage_auc_cm": {
                "trace": float(risk_coverage_auc(trace, frame_error).item() * 100.0),
                "max_eigenvalue": float(
                    risk_coverage_auc(largest_eigenvalue, frame_error).item() * 100.0
                ),
                "mean_disagreement": float(
                    risk_coverage_auc(disagreement, frame_error).item() * 100.0
                ),
            },
            "joint_error_spearman": _spearman(joint_spread.flatten(), errors.flatten()),
            "joint_risk_coverage_auc_cm": float(
                risk_coverage_auc(joint_spread.flatten(), errors.flatten()).item() * 100.0
            ),
        },
    }
    artifact = {
        "mean": ensemble_mean,
        "within_member_predictive_covariance": within,
        "between_member_model_function_spread": between,
        "frame_model_function_trace": trace,
        "frame_model_function_max_eigenvalue": largest_eigenvalue,
        "frame_member_mean_disagreement": disagreement,
        "joint_model_function_variance": joint_spread,
        "joint_errors": errors,
    }
    return result, artifact


def _evaluate_split(
    records: list[dict[str, torch.Tensor]],
    *,
    split: str,
    device: torch.device,
    samples: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    means = torch.stack([record["mean"] for record in records]).to(device)
    scales = torch.stack([record["scale"] for record in records]).to(device)
    target = records[0]["target"].to(device)
    individual = {
        f"member_{index}": _subset_metrics(
            means[index : index + 1],
            scales[index : index + 1],
            target,
            seed=10_000 + index,
            samples=samples,
        )
        for index in range(len(records))
    }
    full = _subset_metrics(
        means, scales, target, seed=20_000, samples=samples
    )
    full["mixture_projection_pit"] = mixture_projection_pit(
        means,
        scales,
        target,
        student_t_dof=STUDENT_T_DOF,
        seed=30_000,
    )
    spread, artifact = _spread_metrics(means, scales, target)
    subsets = {}
    for indices in itertools.combinations(range(len(records)), 2):
        name = "+".join(f"member_{index}" for index in indices)
        subsets[name] = _subset_metrics(
            means[list(indices)],
            scales[list(indices)],
            target,
            seed=40_000 + sum(indices),
            samples=samples,
        )
    artifact.update(
        {
            "sample_id": _sample_id(records[0]),
            "target": records[0]["target"],
            "component_means": means.cpu(),
            "component_scales": scales.cpu(),
        }
    )
    return {
        "split": split,
        "individual_members": individual,
        "three_member_ensemble": full,
        "two_member_subsets": subsets,
        "spread_diagnostics": spread,
    }, {name: value.cpu() for name, value in artifact.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs=3, type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite ensemble evaluation: {args.output_dir}")
    if args.samples < 16:
        raise ValueError("use at least 16 predictive samples for Energy Score")
    members = [_member_record(path) for path in args.run_dirs]
    if len({member["seed"] for member in members}) != len(members):
        raise ValueError("E3a members must use distinct initialization/sampler seeds")
    if len({json.dumps(member["contract"], sort_keys=True) for member in members}) != 1:
        raise ValueError("E3a member training contracts differ beyond randomness seed")
    if len({member["provenance"]["dataset_cache_hash"] for member in members}) != 1:
        raise ValueError("E3a members use different geometry-cache provenance")
    if len({json.dumps(member["compilation"], sort_keys=True) for member in members}) != 1:
        raise ValueError("E3a members use different compiler/operator schemas")
    device = torch.device(args.device)
    side, side_artifact = _evaluate_split(
        _load_split(members, "side"), split="Side IID", device=device, samples=args.samples
    )
    top, top_artifact = _evaluate_split(
        _load_split(members, "top"), split="Top cross-view OOD", device=device, samples=args.samples
    )
    side_trace = side_artifact["frame_model_function_trace"]
    top_trace = top_artifact["frame_model_function_trace"]
    ood_scores = torch.cat((side_trace, top_trace))
    ood_labels = torch.cat(
        (torch.zeros_like(side_trace, dtype=torch.long), torch.ones_like(top_trace, dtype=torch.long))
    )
    result = {
        "schema_version": 1,
        "kind": "E3a_independent_end_to_end_full_student_t_ensemble",
        "density_semantics": "equal_weight_exact_finite_student_t_logsumexp",
        "student_t_dof": STUDENT_T_DOF,
        "selection": "each member selected by Side validation NLL; Top never used for selection",
        "members": [
            {
                "path": member["path"],
                "seed": member["seed"],
                "checkpoint_sha256": member["checkpoint_sha256"],
                "checkpoint_chain_sha256": member["checkpoint_chain_sha256"],
                "prediction_sha256": member["prediction_sha256"],
                "source": member["environment"]["source"],
                "selected_epoch": torch.load(
                    Path(member["path"]) / "best_model.pt", map_location="cpu", weights_only=True
                )["epoch"],
            }
            for member in members
        ],
        "shared_contract": members[0]["contract"],
        "dataset_cache_hash": members[0]["provenance"]["dataset_cache_hash"],
        "source": source_provenance(Path(__file__).resolve().parents[1]),
        "side": side,
        "top": top,
        "cross_view_model_function_spread": {
            "side_mean_trace_m2": float(side_trace.mean()),
            "top_mean_trace_m2": float(top_trace.mean()),
            "top_to_side_trace_ratio": float(top_trace.mean() / side_trace.mean().clamp_min(1e-12)),
            "side_to_top_auroc": binary_auroc(ood_scores, ood_labels),
            "interpretation": "shift diagnostic only; not a calibration claim",
        },
    }
    args.output_dir.mkdir(parents=True)
    side_path = args.output_dir / "predictions_side.pt"
    top_path = args.output_dir / "predictions_top.pt"
    _atomic_save(side_artifact, side_path)
    _atomic_save(top_artifact, top_path)
    result["artifacts"] = {
        "predictions_side.pt": sha256_file(side_path),
        "predictions_top.pt": sha256_file(top_path),
    }
    atomic_write_json(result, args.output_dir / "metrics.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
