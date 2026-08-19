"""Audit representation-compatible full-image elasticity artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


REQUIRED_FILES = (
    "args.json",
    "environment.json",
    "compilation.json",
    "history.json",
    "metrics.json",
    "predictions.pt",
    "best_model.pt",
    "train.log",
)


def _finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def audit_run(run_dir: Path, *, seed: int) -> dict[str, Any]:
    failures: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        return {
            "seed": seed,
            "run_dir": str(run_dir),
            "eligible": False,
            "missing": missing,
            "failures": [f"missing artifacts: {', '.join(missing)}"],
        }

    payloads = {
        name: json.loads((run_dir / name).read_text(encoding="utf-8"))
        for name in REQUIRED_FILES
        if name.endswith(".json")
    }
    args = payloads["args.json"]
    compilation = payloads["compilation.json"]
    metrics = payloads["metrics.json"]
    history = payloads["history.json"]

    if args.get("seed") != seed:
        failures.append(f"args seed is {args.get('seed')!r}, expected {seed}")
    if args.get("arm") != "full_asinh_exp_student_t":
        failures.append(f"unexpected arm: {args.get('arm')!r}")
    if args.get("target_normalization") != "representation_compatible":
        failures.append("target normalization is not representation-compatible")
    if args.get("objective") != "student_t" or float(args.get("student_t_dof", 0)) != 5.0:
        failures.append("Student-t objective contract is not fixed nu=5")

    family = compilation.get("family", {})
    reachability = compilation.get("representation_reachability", {})
    if family.get("kind") != "asinh_exponential":
        failures.append(f"unexpected compiler family: {family.get('kind')!r}")
    active = reachability.get("active", {})
    canonical = reachability.get("canonical", {})
    if not active.get("reachable") or active.get("depth") != 3:
        failures.append("active full target is not recorded as reachable at depth 3")
    if not canonical.get("reachable") or canonical.get("depth") != 3:
        failures.append("canonical full target is not recorded as reachable at depth 3")
    if "8e" not in str(active.get("target_irreps", "")):
        failures.append("active target does not record the ell=8 obligation")

    selected_epochs = [int(row["epoch"]) for row in history]
    if not selected_epochs or min(history, key=lambda row: row["val_loss"])["epoch"] not in selected_epochs:
        failures.append("history does not contain validation-based checkpoint selection")
    if any("test" in str(row).lower() for row in history):
        failures.append("training history contains a test-derived selection field")

    if not _finite(metrics):
        failures.append("metrics contain non-finite values")
    if not bool(metrics.get("finite")):
        failures.append("metrics finite gate is false")
    fp64 = metrics.get("fp64_scatter", {})
    if not bool(fp64.get("strict_spd")) or float(fp64.get("minimum_eigenvalue", 0)) <= 0:
        failures.append("FP64 strict-SPD gate is false")

    predictions = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=True)
    if not _finite(predictions):
        failures.append("predictions contain non-finite values")

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "eligible": not failures,
        "missing": [],
        "failures": failures,
        "nll": metrics.get("nll"),
        "energy_score": metrics.get("energy_score"),
        "mae_gpa": metrics.get("mae_gpa"),
        "minimum_eigenvalue": fp64.get("minimum_eigenvalue"),
        "selected_epoch": min(history, key=lambda row: row["val_loss"])["epoch"],
        "active_depth": active.get("depth"),
        "active_target_irreps": active.get("target_irreps"),
    }


def audit_campaign(root: Path, seeds: tuple[int, ...] = (42, 43, 44)) -> dict[str, Any]:
    runs = tuple(
        audit_run(root / f"seed_{seed}" / "full_asinh_exp_student_t_reprnorm", seed=seed)
        for seed in seeds
    )
    return {
        "kind": "representation_compatible_elasticity_asinh_audit",
        "seeds": list(seeds),
        "eligible": all(run["eligible"] for run in runs),
        "runs": list(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_campaign(args.root)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")
    if not report["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
