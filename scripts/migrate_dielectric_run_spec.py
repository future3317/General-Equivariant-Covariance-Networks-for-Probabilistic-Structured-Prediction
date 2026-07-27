"""Explicitly attach the strict model contract to a legacy dielectric run.

This migration is intentionally one-way and does not make runtime loading fall
back to ``args.json``.  Legacy runs remain marked ``legacy_pre_unified`` and
cannot be used as predecessors for the three-stage protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.dielectric_runtime import DielectricRunSpec, write_run_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    if (checkpoint_dir / "run_spec.json").exists():
        raise FileExistsError(f"run already has a strict contract: {checkpoint_dir}")
    args_path = checkpoint_dir / "args.json"
    compilation_path = checkpoint_dir / "compilation.json"
    if not args_path.is_file() or not compilation_path.is_file():
        raise FileNotFoundError("migration requires both args.json and compilation.json")
    values = json.loads(args_path.read_text())
    spec = DielectricRunSpec.from_dict(values)
    compilation = json.loads(compilation_path.read_text())
    write_run_spec(
        checkpoint_dir,
        spec,
        compilation=compilation,
        training_stage="legacy_pre_unified",
        init_checkpoint=None,
    )
    print(f"wrote {checkpoint_dir / 'run_spec.json'}")


if __name__ == "__main__":
    main()
