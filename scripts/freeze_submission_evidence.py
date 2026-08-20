"""Freeze matched dielectric headline metrics and submission provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import f

from distributions.student_t import student_t_log_prob_from_statistics
from evaluation.metrics import mahalanobis_distance_squared

SEEDS = (42, 43, 44)
LAW_ARMS = ("fixed", "global_nu", "conditional_scale", "conditional_nu")
TRAINED_ARMS = LAW_ARMS[1:]
HEADLINE_FIELDS = (
    "nll",
    "energy_score",
    "coverage50",
    "coverage90",
    "coverage95",
    "mace",
    "whitened_second_moment_defect",
    "radial_ks",
    "radius_direction_max_abs_spearman",
    "radius_direction_pvalue",
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _require_files(run: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run}: missing {missing}")


def _test_payload(diagnostics: dict[str, Any]) -> dict[str, Any]:
    payload = diagnostics.get("test", diagnostics)
    if not isinstance(payload, dict):
        raise TypeError("test diagnostics must be an object")
    return payload


def _nu_from_prediction(prediction: dict[str, Any], protocol: dict[str, Any]) -> torch.Tensor:
    if "nu" in prediction:
        return torch.as_tensor(prediction["nu"], dtype=torch.float64)
    schema = protocol.get("distribution_schema", {})
    value = schema.get("degrees_of_freedom", protocol.get("student_t_dof", 5.0))
    if isinstance(value, (int, float)):
        return torch.tensor(float(value), dtype=torch.float64)
    raise ValueError("prediction/protocol lacks Student-t degrees of freedom")


def _prediction_metrics(
    prediction: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, float]:
    mean = torch.as_tensor(prediction.get("mean", prediction.get("mu")), dtype=torch.float64)
    target = torch.as_tensor(prediction["target"], dtype=torch.float64)
    scale = torch.as_tensor(prediction["scale"], dtype=torch.float64)
    if mean.ndim != 2 or target.shape != mean.shape or scale.shape != (mean.shape[0], mean.shape[1], mean.shape[1]):
        raise ValueError("prediction tensors do not have compatible mean/target/scale shapes")
    if not all(bool(torch.isfinite(value).all()) for value in (mean, target, scale)):
        raise FloatingPointError("prediction tensors are non-finite")
    nu = _nu_from_prediction(prediction, protocol)
    if bool((nu <= 2.0).any()):
        raise ValueError("finite-covariance headline metrics require nu > 2")
    q = mahalanobis_distance_squared(target - mean, scale)
    logdet = torch.linalg.slogdet(scale)[1]
    nll = float((-student_t_log_prob_from_statistics(logdet, q, mean.shape[-1], nu)).mean())
    nu_np = nu.detach().cpu().numpy()
    if nu_np.ndim == 0:
        nu_np = np.full(mean.shape[0], float(nu_np))
    elif nu_np.shape != (mean.shape[0],):
        raise ValueError("conditional nu must be scalar or one value per test sample")
    q_np = q.detach().cpu().numpy()
    levels = (0.50, 0.90, 0.95)
    coverage = {
        f"coverage{int(level * 100)}": float(
            np.mean(q_np < mean.shape[-1] * f.ppf(level, mean.shape[-1], nu_np))
        )
        for level in levels
    }
    observed = np.asarray(
        [
            np.mean(q_np < mean.shape[-1] * f.ppf(level, mean.shape[-1], nu_np))
            for level in np.linspace(0.1, 0.9, 9)
        ],
        dtype=np.float64,
    )
    mace = float(np.mean(np.abs(observed - np.linspace(0.1, 0.9, 9))))
    return {"nll": nll, **coverage, "mace": mace}


def _stored_energy(payload: dict[str, Any], metrics: dict[str, Any]) -> float:
    value = payload.get("energy_score", metrics.get("energy_score"))
    if value is None:
        raise KeyError("missing stored Energy Score")
    return float(value)


def _stored_elliptical(payload: dict[str, Any]) -> dict[str, float]:
    elliptical = payload.get("elliptical_falsification", payload.get("elliptical", {}))
    if not isinstance(elliptical, dict):
        raise TypeError("elliptical diagnostics must be an object")
    radial = elliptical.get("radial_pit", {})
    dependence = elliptical.get("radius_direction_dependence", {})
    required = {
        "whitened_second_moment_defect": elliptical.get("whitened_second_moment_defect"),
        "radial_ks": radial.get("ks"),
        "radius_direction_max_abs_spearman": dependence.get("max_abs_spearman"),
        "radius_direction_pvalue": dependence.get("max_statistic_permutation_pvalue"),
    }
    if any(value is None for value in required.values()):
        raise KeyError("incomplete law-correct elliptical diagnostics")
    return {key: float(value) for key, value in required.items()}


def summarize_law_arm(run_dir: Path, *, arm: str, seed: int) -> dict[str, Any]:
    """Read one arm and recompute FP64 NLL/calibration from saved predictions."""

    _require_files(run_dir, ("diagnostics.json", "environment.json", "predictions_test.pt"))
    diagnostics = _json(run_dir / "diagnostics.json")
    protocol = _json(run_dir / "protocol.json") if (run_dir / "protocol.json").is_file() else {}
    metrics = _json(run_dir / "metrics.json") if (run_dir / "metrics.json").is_file() else {}
    provenance = _json(run_dir / "provenance.json") if (run_dir / "provenance.json").is_file() else {}
    prediction = torch.load(run_dir / "predictions_test.pt", map_location="cpu", weights_only=True)
    if not isinstance(prediction, dict):
        raise TypeError(f"prediction artifact is not a dictionary: {run_dir}")
    record = _prediction_metrics(prediction, protocol)
    payload = _test_payload(diagnostics)
    record["energy_score"] = _stored_energy(payload, metrics)
    record.update(_stored_elliptical(payload))
    frozen = protocol.get("frozen", {})
    checkpoint = frozen.get("operator_checkpoint")
    if arm == "fixed" and checkpoint is None:
        checkpoint = str((run_dir / "best_model.pt").resolve())
    if arm != "fixed" and not checkpoint:
        raise ValueError(f"{run_dir}: trained law arm lacks frozen operator checkpoint")
    environment = _json(run_dir / "environment.json")
    source = environment.get("source", {})
    cache_id = frozen.get("cache_metadata_sha256") or provenance.get("cache_metadata_sha256")
    selected_epoch = protocol.get("selection", {}).get("selected_epoch", metrics.get("selected_epoch"))
    best_validation_nll = protocol.get("selection", {}).get(
        "best_validation_nll", metrics.get("best_validation_nll")
    )
    return {
        "arm": arm,
        "seed": seed,
        "run_dir": str(run_dir),
        "operator_checkpoint": str(checkpoint),
        "cache_metadata_sha256": cache_id,
        "source_commit": source.get("commit"),
        "selected_epoch": selected_epoch,
        "best_validation_nll": best_validation_nll,
        "test": record,
        "artifact_manifest": str((run_dir / "manifest.json").resolve()) if (run_dir / "manifest.json").is_file() else None,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"seeds": rows}
    for field in HEADLINE_FIELDS:
        values = np.asarray([row["test"][field] for row in rows], dtype=np.float64)
        summary[field] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
    return summary


def build_headline_table(formal_root: Path, matched_root: Path) -> dict[str, Any]:
    """Build the four-arm table and enforce per-seed checkpoint identity."""

    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in LAW_ARMS}
    for seed in SEEDS:
        fixed = summarize_law_arm(formal_root / f"seed_{seed}", arm="fixed", seed=seed)
        rows["fixed"].append(fixed)
        fixed_checkpoint = str((formal_root / f"seed_{seed}" / "best_model.pt").resolve())
        for arm in TRAINED_ARMS:
            record = summarize_law_arm(
                matched_root / arm / f"seed_{seed}", arm=arm, seed=seed
            )
            if Path(record["operator_checkpoint"]).resolve() != Path(fixed_checkpoint).resolve():
                raise ValueError(f"seed {seed}: operator checkpoint mismatch for {arm}")
            rows[arm].append(record)
    return {
        "schema_version": 1,
        "kind": "dielectric_matched_four_law_headline",
        "seeds": list(SEEDS),
        "arms": {arm: _aggregate(values) for arm, values in rows.items()},
    }


def build_submission_manifest(
    headline: dict[str, Any],
    *,
    canonical_commit: str,
    evaluator_commit: str,
    topology_audit: str | None = None,
    elasticity_root: str | None = None,
) -> dict[str, Any]:
    """Create a compact submission-level index without copying large artifacts."""

    evidence: dict[str, Any] = {"dielectric_matched_four_law": headline}
    if topology_audit:
        evidence["itop_prufer_topology_null"] = {
            "audit": topology_audit,
            "status": "formal_descriptive_topology_draws",
        }
    if elasticity_root:
        evidence["elasticity"] = {
            "root": elasticity_root,
            "status": "representation_compatible_full_image_chart_separate_from_legacy_voigt",
        }
    return {
        "schema_version": 1,
        "kind": "tpami_submission_evidence_manifest",
        "canonical_commit": canonical_commit,
        "evaluator_commit": evaluator_commit,
        "split_contract": {
            "dielectric": "train/validation/test = 4236/485/281; validation-only selection",
            "itop": "Side train/validation; Top evaluation-only",
            "elasticity": "predefined train/validation/test; validation-only selection",
        },
        "claim_classes": {
            "formal": ["dielectric_matched_four_law", "dielectric_factorial", "itop_prufer_topology_null"],
            "negative_diagnostic": ["orientation", "shared_mean_mixture", "global_nu_single_seed"],
            "legacy": ["elasticity_legacy_voigt"],
        },
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-commit", required=True)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--topology-audit")
    parser.add_argument("--elasticity-root")
    args = parser.parse_args()
    headline = build_headline_table(args.formal_root, args.matched_root)
    manifest = build_submission_manifest(
        headline,
        canonical_commit=args.canonical_commit,
        evaluator_commit=args.evaluator_commit,
        topology_audit=args.topology_audit,
        elasticity_root=args.elasticity_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "headline_metrics.json").write_text(
        json.dumps(headline, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "submission_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"headline": str(args.output_dir / "headline_metrics.json"), "manifest": str(args.output_dir / "submission_evidence_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
