"""Attach complete source/data/checkpoint provenance to an existing run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dielectric_runtime import (
    compilation_record_with_hash,
    checkpoint_chain_provenance,
    dataset_provenance,
    inference_contract_from_args,
    inference_contract_hash,
    load_dielectric_data_args,
    load_run_record,
    source_provenance,
)


def attach(run_dir: Path, repo_root: Path, data_dir: Path | None = None) -> None:
    stages = [run_dir / name for name in ("mean", "covariance", "joint")]
    existing = [stage for stage in stages if (stage / "run_spec.json").is_file()]
    if not existing:
        raise FileNotFoundError(f"no dielectric stages found below {run_dir}")
    checkpoint_chain = checkpoint_chain_provenance(existing)
    source = source_provenance(repo_root)
    for stage in existing:
        args = load_dielectric_data_args(stage)
        record = load_run_record(stage)
        contract = record.get("inference_contract") or inference_contract_from_args(args, args.device)
        provenance = {
            "source": source,
            "dataset": dataset_provenance(data_dir or Path(args.data_dir)),
            "checkpoint_chain": checkpoint_chain,
        }
        record["inference_contract"] = contract
        record["inference_contract_hash"] = inference_contract_hash(contract)
        record["provenance"] = provenance
        compilation = compilation_record_with_hash(record.get("compilation", {}))
        record["compilation"] = compilation
        (stage / "run_spec.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        (stage / "compilation.json").write_text(json.dumps(compilation, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--repo_root", required=True, type=Path)
    parser.add_argument("--data_dir", type=Path)
    args = parser.parse_args()
    attach(args.run_dir, args.repo_root, args.data_dir)


if __name__ == "__main__":
    main()
