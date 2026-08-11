"""Verify and aggregate frozen dielectric family-factorial artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from evaluation import empirical_coverage
from scripts.itop_reproducibility import atomic_write_json, sha256_file
from scripts.run_dielectric_family_factorial import (
    DISTRIBUTIONS,
    FAMILIES,
    FAMILY_PARAMETER_COUNTS,
)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _audit_arm(root: Path, family: str, distribution: str, seed: int) -> dict:
    arm = root / family / distribution / f"seed_{seed}"
    manifest_path = arm / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing factorial manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["family"], manifest["distribution"], manifest["seed"]) != (
        family,
        distribution,
        seed,
    ):
        raise ValueError(f"arm identity mismatch: {arm}")
    for name, expected in manifest["artifact_sha256"].items():
        path = arm / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact hash mismatch: {path}")
    args = json.loads((arm / "args.json").read_text())
    schema = json.loads((arm / "schema.json").read_text())
    metrics = json.loads((arm / "metrics.json").read_text())
    diagnostics = json.loads((arm / "diagnostics.json").read_text())
    provenance = json.loads((arm / "provenance.json").read_text())
    if args["selection_split"] != "validation":
        raise ValueError(f"non-validation selection: {arm}")
    if args["family"] != family or args["distribution"] != distribution:
        raise ValueError(f"args identity mismatch: {arm}")
    if schema["distribution"]["objective"] != (
        f"exact_{distribution}_log_likelihood"
    ):
        raise ValueError(f"distribution schema mismatch: {arm}")
    if schema["operator"]["parameter_count"] != FAMILY_PARAMETER_COUNTS[family]:
        raise ValueError(f"operator family schema mismatch: {arm}")
    fidelity = schema["compilation"]["execution_fidelity"]
    if fidelity["exactness"] != "exact_for_active_family":
        raise ValueError(f"non-exact executor schema: {arm}")
    if not _finite_tree(metrics):
        raise FloatingPointError(f"non-finite metrics: {arm}")
    prediction = torch.load(
        arm / "predictions_test.pt", map_location="cpu", weights_only=True
    )
    required = {"mean", "target", "sample_id", "params", "scale"}
    if not required.issubset(prediction):
        raise ValueError(f"incomplete predictions: {arm}")
    if not all(bool(torch.isfinite(prediction[key]).all()) for key in required):
        raise FloatingPointError(f"non-finite predictions: {arm}")
    minimum = float(torch.linalg.eigvalsh(prediction["scale"].double()).min())
    if minimum <= 0.0:
        raise FloatingPointError(f"non-SPD predictions: {arm}")
    coverage = empirical_coverage(
        prediction["mean"],
        prediction["target"],
        prediction["scale"],
        levels=[0.9, 0.95],
        reference=distribution,
        student_t_dof=5.0,
    )
    elliptical = diagnostics["elliptical_falsification"]
    diagnostic_summary = {
        "radial_ks": elliptical["radial_pit"]["ks"],
        "projection_median_ks": elliptical["projection_pit"]["median_ks"],
        "direction_second_moment_defect": elliptical["direction_sphericality"][
            "second_moment_defect"
        ],
        "radius_direction_pvalue": elliptical["radius_direction_dependence"][
            "max_statistic_permutation_pvalue"
        ],
        "whitened_second_moment_defect": elliptical[
            "whitened_second_moment_defect"
        ],
    }
    return {
        "family": family,
        "distribution": distribution,
        "seed": seed,
        "metrics": metrics,
        "coverage": coverage,
        "diagnostic_summary": diagnostic_summary,
        "provenance": provenance,
        "sample_count": int(prediction["target"].shape[0]),
        "sample_id_sha256": _tensor_sha256(prediction["sample_id"]),
        "target_sha256": _tensor_sha256(prediction["target"]),
        "mean_sha256": _tensor_sha256(prediction["mean"]),
        "minimum_scale_eigenvalue_fp64": minimum,
    }


def _audit_fixed_control(path: Path, expected_cache_hash: str) -> dict[str, Any]:
    """Verify the zero-training cached Full-t control from the existing E1 path."""

    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        artifact = path / name
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"fixed-control artifact hash mismatch: {artifact}")
    protocol = json.loads((path / "protocol.json").read_text())
    diagnostics = json.loads((path / "diagnostics.json").read_text())
    if protocol["variant"] != "fixed":
        raise ValueError("pilot baseline control must use the fixed E1 variant")
    if protocol["selection"]["selected_epoch"] != 0:
        raise ValueError("fixed baseline control must be zero-training")
    if protocol["frozen"]["cache_metadata_sha256"] != expected_cache_hash:
        raise ValueError("fixed baseline uses a different frozen cache")
    test_nll = float(diagnostics["test"]["nll"])
    if not math.isfinite(test_nll):
        raise FloatingPointError("fixed baseline test NLL is non-finite")
    return {
        "path": str(path),
        "test_nll": test_nll,
        "reference_nll": -2.6247,
        "absolute_error": abs(test_nll + 2.6247),
    }


def aggregate_factorial(
    root: Path,
    *,
    stage: str,
    seeds: tuple[int, ...],
    baseline_control: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Audit all expected arms before computing any family/law comparison."""

    expected_seeds = (42,) if stage in {"smoke", "pilot"} else (42, 43, 44)
    if seeds != expected_seeds:
        raise ValueError(f"{stage} requires seeds {expected_seeds}")
    rows = [
        _audit_arm(root, family, distribution, seed)
        for family in FAMILIES
        for distribution in DISTRIBUTIONS
        for seed in seeds
    ]
    common_fields = (
        "sample_count",
        "sample_id_sha256",
        "target_sha256",
        "mean_sha256",
    )
    common = {
        field: len({row[field] for row in rows}) == 1 for field in common_fields
    }
    cache_hashes = {
        row["provenance"]["cache_metadata_sha256"] for row in rows
    }
    common["cache_metadata_sha256"] = len(cache_hashes) == 1
    if not all(common.values()):
        raise ValueError(f"factorial arms do not share frozen artifacts: {common}")
    if stage in {"pilot", "formal"} and rows[0]["sample_count"] != 281:
        raise ValueError(
            f"formal dielectric test split must contain 281 samples, got "
            f"{rows[0]['sample_count']}"
        )
    result: dict[str, Any] = {
        "stage": stage,
        "seeds": list(seeds),
        "rows": rows,
        "common_frozen_artifacts": common,
    }
    if stage == "pilot":
        if baseline_control is None:
            raise ValueError("pilot audit requires a zero-training baseline control")
        fixed = _audit_fixed_control(baseline_control, next(iter(cache_hashes)))
        reference = fixed["absolute_error"] <= 1e-4
        result["fixed_cache_control"] = fixed
        result["operational_gate"] = {
            "all_arms_present": len(rows) == 8,
            "finite_spd": all(
                row["minimum_scale_eigenvalue_fp64"] > 0.0 for row in rows
            ),
            "common_frozen_artifacts": all(common.values()),
            "validation_only_selection": True,
            "hashes_verified": True,
            "fixed_cache_reference": reference,
            "overall": reference,
        }
    if stage == "formal":
        aggregate = []
        for family in FAMILIES:
            for distribution in DISTRIBUTIONS:
                selected = [
                    row
                    for row in rows
                    if row["family"] == family
                    and row["distribution"] == distribution
                ]
                nlls = [row["metrics"]["nll"] for row in selected]
                energies = [row["metrics"]["energy_score"] for row in selected]
                aggregate.append(
                    {
                        "family": family,
                        "distribution": distribution,
                        "nll_mean": sum(nlls) / len(nlls),
                        "nll_std": torch.tensor(nlls).std(unbiased=True).item(),
                        "energy_score_mean": sum(energies) / len(energies),
                        "energy_score_std": torch.tensor(energies)
                        .std(unbiased=True)
                        .item(),
                    }
                )
        result["aggregate"] = aggregate
        result["paired_nll_deltas"] = {
            family: [
                next(
                    row["metrics"]["nll"]
                    for row in rows
                    if row["family"] == family
                    and row["distribution"] == "student_t"
                    and row["seed"] == seed
                )
                - next(
                    row["metrics"]["nll"]
                    for row in rows
                    if row["family"] == family
                    and row["distribution"] == "gaussian"
                    and row["seed"] == seed
                )
                for seed in seeds
            ]
            for family in FAMILIES
        }
    if output is not None:
        atomic_write_json(result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--baseline_control", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(","))
    result = aggregate_factorial(
        args.root,
        stage=args.stage,
        seeds=seeds,
        baseline_control=args.baseline_control,
        output=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
