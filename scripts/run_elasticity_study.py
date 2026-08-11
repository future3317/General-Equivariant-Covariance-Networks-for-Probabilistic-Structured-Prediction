"""Run matched, sequential elasticity arms under an explicit phase gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.itop_reproducibility import atomic_write_json
from scripts.train_elasticity import ELASTICITY_ARMS

REQUIRED_ARM_ARTIFACTS = (
    "args.json",
    "environment.json",
    "schema.json",
    "history.json",
    "metrics.json",
    "predictions.pt",
    "best_model.pt",
    "train.log",
)


def study_arm_arguments(arm: str) -> list[str]:
    if arm not in ELASTICITY_ARMS:
        raise ValueError(f"unsupported elasticity arm: {arm}")
    return ["--arm", arm]


def assign_devices(
    jobs: list[tuple[int, str]], devices: list[str]
) -> list[tuple[int, str, str]]:
    if not devices:
        raise ValueError("at least one device is required")
    return [
        (seed, arm, devices[index % len(devices)])
        for index, (seed, arm) in enumerate(jobs)
    ]


def pilot_gate(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered fast-pilot expansion gate."""

    if set(records) != set(ELASTICITY_ARMS):
        return {"passed": False, "reasons": ["pilot does not contain all three arms"]}
    reasons: list[str] = []
    for arm in ELASTICITY_ARMS:
        record = records[arm]
        if not record.get("required_artifacts_complete", False):
            reasons.append(f"{arm}: required artifacts incomplete")
        if not record.get("finite", False):
            reasons.append(f"{arm}: finite gate failed")
        if not record.get("schema_valid", False):
            reasons.append(f"{arm}: schema gate failed")
        if float(record.get("best_validation_criterion", float("inf"))) >= float(
            record.get("first_validation_criterion", float("inf"))
        ):
            reasons.append(f"{arm}: validation criterion did not improve")
        if float(record.get("peak_allocated_gib", float("inf"))) > 20.0:
            reasons.append(f"{arm}: peak allocated memory exceeded 20 GiB")
    return {"passed": not reasons, "reasons": reasons}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_gate_record(run_dir: Path) -> dict[str, Any]:
    complete = all((run_dir / name).is_file() for name in REQUIRED_ARM_ARTIFACTS)
    if not complete:
        return {"required_artifacts_complete": False}
    history = _read(run_dir / "history.json")
    metrics = _read(run_dir / "metrics.json")
    schema = _read(run_dir / "schema.json")
    criteria = [float(row["validation_criterion"]) for row in history]
    return {
        "required_artifacts_complete": True,
        "finite": bool(metrics.get("finite", False)),
        "schema_valid": bool(schema.get("schema_valid", False)),
        "first_validation_criterion": criteria[0],
        "best_validation_criterion": min(criteria),
        "peak_allocated_gib": float(metrics["runtime"]["peak_allocated_gib"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "formal"), required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    seeds = args.seeds or ([42] if args.stage == "pilot" else [42, 43, 44])
    if args.stage == "pilot" and seeds != [42]:
        parser.error("pilot stage is fixed to seed 42")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, arm) for seed in seeds for arm in ELASTICITY_ARMS]
    scheduled = assign_devices(jobs, args.devices)
    commands: list[tuple[Path, list[str]]] = []
    for seed, arm, device in scheduled:
        run_dir = args.output_dir / f"seed_{seed}" / arm
        command = [
                sys.executable,
                "-m",
                "scripts.train_elasticity",
                *study_arm_arguments(arm),
                "--save_dir",
                str(run_dir),
                "--seed",
                str(seed),
                "--device",
                device,
                "--num_workers",
                str(args.num_workers),
                "--pin_memory",
                "--persistent_workers",
                "--prefetch_factor",
                "2",
                "--batch_size",
                "16",
                "--rank",
                "2",
        ]
        if args.data_dir is not None:
            command.extend(("--data_dir", str(args.data_dir)))
        if args.stage == "pilot":
            command.extend(
                (
                        "--train_subset",
                        "1024",
                        "--eval_subset",
                        "256",
                        "--num_epochs",
                        "6",
                        "--patience",
                        "2",
                )
            )
        else:
            command.extend(("--num_epochs", "30", "--patience", "5"))
        commands.append((run_dir, command))

    for start in range(0, len(commands), len(args.devices)):
        wave = commands[start : start + len(args.devices)]
        processes = [subprocess.Popen(command) for _, command in wave]
        for (run_dir, command), process in zip(wave, processes, strict=True):
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

    records: dict[str, dict[str, Any]] = {}
    if args.stage == "pilot":
        for arm in ELASTICITY_ARMS:
            records[arm] = _arm_gate_record(args.output_dir / "seed_42" / arm)

    manifest: dict[str, Any] = {
        "stage": args.stage,
        "seeds": seeds,
        "arms": list(ELASTICITY_ARMS),
        "devices": args.devices,
    }
    if args.stage == "pilot":
        manifest["records"] = records
        manifest["gate"] = pilot_gate(records)
    atomic_write_json(manifest, args.output_dir / "study_manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
