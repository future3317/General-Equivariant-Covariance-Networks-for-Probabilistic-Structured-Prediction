"""Certify paired frozen-head splits across ITOP uncertainty families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from data.itop_dataset import itop_train_validation_indices
from scripts.itop_reproducibility import atomic_write_json

FAMILIES = ("full_student_t", "low_rank_student_t", "graph_student_t")
SEEDS = (42, 43, 44)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _parse_run(value: str) -> tuple[str, int, Path]:
    try:
        family, seed_text, path_text = value.split("=", 2)
        seed = int(seed_text)
    except ValueError as error:
        raise ValueError(f"invalid --run specification: {value}") from error
    if family not in FAMILIES or seed not in SEEDS:
        raise ValueError(f"unsupported family/seed in --run: {value}")
    path = Path(path_text)
    if not path.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {path}")
    return family, seed, path


def _split_hash(length: int, seed: int) -> tuple[str, int, int]:
    train, validation = itop_train_validation_indices(length, seed=seed)
    encoded = json.dumps(
        {"train": train, "validation": validation}, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(train), len(validation)


def _canonical_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record(family: str, seed: int, run: Path) -> dict[str, Any]:
    args = _read(run / "args.json")
    if args.get("model") != family or args.get("seed") != seed:
        raise ValueError(f"{run}: args.json does not identify {family}, seed {seed}")
    if args.get("phase") != "frozen_head":
        raise ValueError(f"{run}: paired audit requires frozen_head")
    cache_manifest = _read(run / "feature_cache.json")
    counts = cache_manifest.get("counts", {})
    train_count = int(counts.get("side_train", 0))
    if train_count < 2:
        raise ValueError(f"{run}: feature cache does not record side-train count")
    split_digest, train_size, validation_size = _split_hash(train_count, seed)
    environment = _read(run / "environment.json")
    feature_record = environment.get("feature_cache", {})
    return {
        "family": family,
        "seed": seed,
        "run_dir": str(run),
        "feature_cache_manifest_sha256": _canonical_json_hash(cache_manifest),
        "feature_cache_hash": feature_record.get("feature_cache_hash"),
        "backbone_checkpoint_sha256": cache_manifest["backbone_checkpoint_sha256"],
        "canonical_split_sha256": split_digest,
        "side_train_samples": train_count,
        "train_samples": train_size,
        "validation_samples": validation_size,
        "side_test_samples": int(counts.get("side_test", 0)),
        "top_test_samples": int(counts.get("top_test", 0)),
        "contract": {
            name: args.get(name)
            for name in (
                "feature_cache",
                "backbone_checkpoint",
                "batch_size",
                "num_points",
                "num_neighbors",
                "backbone_precision",
                "tp_backend",
                "cueq_method",
                "compile_tp",
                "student_t_dof",
            )
        },
    }


def _paired_seed(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    if {record["family"] for record in records} != set(FAMILIES):
        raise ValueError(f"seed {seed}: expected exactly {FAMILIES}")
    fields = (
        "feature_cache_manifest_sha256",
        "backbone_checkpoint_sha256",
        "canonical_split_sha256",
        "side_train_samples",
        "train_samples",
        "validation_samples",
        "side_test_samples",
        "top_test_samples",
        "contract",
    )
    reference = records[0]
    mismatches = {
        field: {
            record["family"]: record[field]
            for record in records
            if record[field] != reference[field]
        }
        for field in fields
    }
    mismatches = {field: values for field, values in mismatches.items() if values}
    if mismatches:
        raise ValueError(f"seed {seed}: families are not paired: {mismatches}")
    return {
        "seed": seed,
        "families": list(FAMILIES),
        "paired": True,
        **{field: reference[field] for field in fields if field != "contract"},
        "sampler": f"RandomSampler seed={seed}*1000003+epoch",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="FAMILY=SEED=RUN_DIR",
        help="repeat once for every family/seed cell",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parsed = [_parse_run(value) for value in args.run]
    if len(parsed) != len(FAMILIES) * len(SEEDS):
        parser.error("the audit requires all three families for seeds 42, 43, and 44")
    records = [_record(family, seed, run) for family, seed, run in parsed]
    paired = []
    for seed in SEEDS:
        paired.append(_paired_seed([record for record in records if record["seed"] == seed], seed))
    result = {
        "schema_version": 1,
        "kind": "itop_frozen_head_family_pairing_audit",
        "families": list(FAMILIES),
        "seeds": list(SEEDS),
        "selection": "validation-only NLL within each frozen-head run",
        "paired_by_seed": paired,
        "conclusion": (
            "Within each seed, Full-t, LR-t, and Graph-t share an identical "
            "frozen backbone, feature-cache manifest, canonical 90/10 side-train "
            "membership, sampler law, and side/top tests. Seeds vary this common "
            "split and the head initialization jointly."
        ),
    }
    atomic_write_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
