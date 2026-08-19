"""Prepare aligned label-free ITOP observation descriptors for frozen heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.frozen_distribution_features import validate_observation_descriptors
from data.itop_dataset import itop_train_validation_indices
from data.observation_descriptors import (
    O3_INVARIANT_DESCRIPTOR_NAMES,
    o3_invariant_descriptor_names,
    observation_descriptor_semantics,
    point_cloud_observation_descriptors,
)
from scripts.itop_reproducibility import (
    atomic_write_json,
    sha256_file,
    source_provenance,
)


def _save(payload: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _descriptor_payload(
    cache: Path,
    *,
    view_id: int,
    depth_path: Path | None,
    indices: torch.Tensor | None,
    names: tuple[str, ...] | None,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    raw = point_cloud_observation_descriptors(
        cache,
        view_id=view_id,
        depth_path=depth_path,
    )
    available = tuple(
        name
        for name in raw
        if name not in {"sample_id", "visible_fraction_diagnostic_only"}
    )
    selected_names = (
        tuple(name for name in available if name in O3_INVARIANT_DESCRIPTOR_NAMES)
        if names is None
        else names
    )
    if not set(selected_names).issubset(set(available)):
        missing = sorted(set(selected_names) - set(available))
        raise ValueError(f"requested descriptors are unavailable: {missing}")
    o3_invariant_descriptor_names(selected_names)
    sample_ids = raw["sample_id"]
    descriptor_columns = {
        name: raw[name].float() for name in selected_names
    }
    validate_observation_descriptors(
        descriptor_columns, count=int(sample_ids.shape[0])
    )
    descriptors = torch.stack(
        [descriptor_columns[name] for name in selected_names], dim=-1
    )
    if indices is not None:
        sample_ids = sample_ids.index_select(0, indices)
        descriptors = descriptors.index_select(0, indices)
    return {"sample_id": sample_ids, "descriptors": descriptors}, selected_names


def _split_record(payload: dict[str, torch.Tensor], path: Path) -> dict:
    return {
        "count": int(payload["sample_id"].shape[0]),
        "sha256": sha256_file(path),
        "sample_id_sha256": hashlib.sha256(
            payload["sample_id"].contiguous().numpy().tobytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side_train_cache", type=Path, required=True)
    parser.add_argument("--side_test_cache", type=Path, required=True)
    parser.add_argument("--top_test_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--side_depth_path", type=Path)
    parser.add_argument("--top_depth_path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--descriptor_names", nargs="+")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite descriptor cache: {args.output_dir}")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")

    side_count = int(np.load(args.side_train_cache / "points.npy", mmap_mode="r").shape[0])
    train_indices, validation_indices = itop_train_validation_indices(side_count, seed=args.seed)
    requested = tuple(args.descriptor_names) if args.descriptor_names else None
    sources = (
        ("train", args.side_train_cache, 0, args.side_depth_path, torch.tensor(train_indices)),
        ("val", args.side_train_cache, 0, args.side_depth_path, torch.tensor(validation_indices)),
        ("test", args.side_test_cache, 0, args.side_depth_path, None),
        ("ood", args.top_test_cache, 1, args.top_depth_path, None),
    )
    args.output_dir.mkdir(parents=True)
    records = {}
    names: tuple[str, ...] | None = requested
    for split, cache, view_id, depth_path, indices in sources:
        payload, names = _descriptor_payload(
            cache,
            view_id=view_id,
            depth_path=depth_path,
            indices=indices,
            names=names,
        )
        path = args.output_dir / f"{split}.pt"
        _save(payload, path)
        records[split] = _split_record(payload, path)
    metadata = {
        "schema_version": 1,
        "kind": "itop_observation_descriptors",
        "descriptor_names": list(names or ()),
        "descriptor_semantics": {
            name: observation_descriptor_semantics(name) for name in names or ()
        },
        "selection": {
            "seed": args.seed,
            "validation": "side_train held-out indices",
            "ood": "top test only; never used for selection",
        },
        "splits": records,
        "source": source_provenance(Path(__file__).resolve().parents[1]),
    }
    atomic_write_json(metadata, args.output_dir / "metadata.json")
    print(json.dumps({"output_dir": str(args.output_dir), "splits": records}, indent=2))


if __name__ == "__main__":
    main()
