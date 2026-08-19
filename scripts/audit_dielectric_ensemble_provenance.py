"""Audit provenance and density semantics for a completed dielectric ensemble.

This is a metadata-only gate.  It does not recompute predictions or alter a run;
it checks that the evaluator's exact-mixture result is traceable to matching,
clean, compiled members selected on validation and evaluated on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_ensemble(root: str | Path, seeds: Iterable[int]) -> dict[str, Any]:
    root = Path(root)
    expected_seeds = tuple(int(seed) for seed in seeds)
    failures: list[str] = []
    members: list[dict[str, Any]] = []

    for seed in expected_seeds:
        member_dir = root / f"seed_{seed}"
        run_spec_path = member_dir / "run_spec.json"
        if not run_spec_path.is_file():
            failures.append(f"missing_run_spec_seed_{seed}")
            continue
        record = _load(run_spec_path)
        provenance = record.get("provenance", {})
        source = provenance.get("source", {})
        dataset = provenance.get("dataset", {})
        members.append(
            {
                "seed": seed,
                "path": str(member_dir),
                "source_commit": source.get("commit"),
                "source_dirty": source.get("dirty"),
                "dataset_hash": dataset.get("dataset_hash"),
                "inference_contract_hash": record.get("inference_contract_hash"),
                "training_stage": record.get("training_stage"),
                "compiled_group": record.get("compilation", {}).get("group"),
                "distribution": record.get("model", {}).get("distribution"),
                "student_t_dof": record.get("model", {}).get("student_t_dof"),
                "covariance_parameterization": record.get("model", {}).get(
                    "covariance_parameterization"
                ),
                "has_mean_stage": (member_dir / "mean").is_dir(),
                "has_covariance_stage": (member_dir / "covariance").is_dir(),
            }
        )

    if len(members) != len(expected_seeds):
        failures.append("member_count")
    if members:
        for field in (
            "source_commit",
            "dataset_hash",
            "inference_contract_hash",
            "training_stage",
            "distribution",
            "student_t_dof",
            "covariance_parameterization",
            "compiled_group",
        ):
            if len({member[field] for member in members}) != 1:
                failures.append(field)
        if any(member["source_dirty"] is not False for member in members):
            failures.append("dirty_source")
        if any(
            not member["has_mean_stage"] or not member["has_covariance_stage"]
            for member in members
        ):
            failures.append("staged_training")
        if any(member["training_stage"] != "joint" for member in members):
            failures.append("joint_training")
        if any(member["compiled_group"] != "O(3)" for member in members):
            failures.append("compiled_group")
        if any(member["distribution"] != "student_t" for member in members):
            failures.append("student_t_distribution")
        if any(member["student_t_dof"] != 5.0 for member in members):
            failures.append("fixed_student_t_dof")

    metrics_path = root / "ensemble_3member_metrics.json"
    metrics: dict[str, Any] = {}
    if not metrics_path.is_file():
        failures.append("missing_ensemble_metrics")
    else:
        metrics = _load(metrics_path)
        if metrics.get("density_semantics") != "equally_weighted_member_mixture":
            failures.append("density_semantics")
        if metrics.get("fit_split") != "validation":
            failures.append("fit_split")
        if metrics.get("eval_split") != "test":
            failures.append("eval_split")
        if members and metrics.get("inference_contract_hash") != members[0]["inference_contract_hash"]:
            failures.append("metric_contract_hash")

    failures = sorted(set(failures))
    return {
        "root": str(root),
        "expected_seeds": list(expected_seeds),
        "member_count": len(members),
        "members": members,
        "compiled_group": (
            members[0]["compiled_group"] if members else None
        ),
        "density_semantics": metrics.get("density_semantics"),
        "fit_split": metrics.get("fit_split"),
        "eval_split": metrics.get("eval_split"),
        "failures": failures,
        "eligible": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    report = audit_ensemble(args.root, seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
