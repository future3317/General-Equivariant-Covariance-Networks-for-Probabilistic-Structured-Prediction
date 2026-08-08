"""Validated task-neutral caches for frozen-H,mu distribution experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from scripts.itop_reproducibility import sha256_file

REQUIRED_FIELDS = ("features", "mean", "params", "target", "sample_id")


class FrozenDistributionDataset(Dataset):
    """One immutable split with aligned typed features and frozen predictions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"invalid frozen-distribution cache: {self.path}")
        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"cache lacks required fields {missing}: {self.path}")
        count = int(payload["features"].shape[0])
        if any(int(payload[field].shape[0]) != count for field in REQUIRED_FIELDS):
            raise ValueError(f"cache fields have inconsistent lengths: {self.path}")
        if torch.unique(payload["sample_id"]).numel() != count:
            raise ValueError(f"sample IDs are not unique within split: {self.path}")
        for field in ("features", "mean", "params", "target"):
            if not bool(torch.isfinite(payload[field]).all()):
                raise ValueError(f"non-finite {field} in {self.path}")
        self.payload = payload

    def __len__(self) -> int:
        return int(self.payload["features"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {field: self.payload[field][index] for field in REQUIRED_FIELDS}


def load_frozen_distribution_cache(
    cache_dir: str | Path,
) -> tuple[dict, dict[str, FrozenDistributionDataset]]:
    root = Path(cache_dir)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported frozen-distribution cache schema")
    if not {"train", "val"}.issubset(metadata.get("splits", {})):
        raise ValueError("cache must declare train and validation splits")
    datasets = {}
    for split, record in metadata["splits"].items():
        path = root / f"{split}.pt"
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"split hash mismatch: {path}")
        dataset = FrozenDistributionDataset(path)
        if len(dataset) != int(record["count"]):
            raise ValueError(f"split count mismatch: {path}")
        datasets[split] = dataset
    return metadata, datasets


def frozen_distribution_loaders(
    cache_dir: str | Path,
    *,
    seed: int,
    epoch: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[dict, dict[str, DataLoader]]:
    metadata, datasets = load_frozen_distribution_cache(cache_dir)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    loaders = {
        split: DataLoader(dataset, shuffle=False, **common)
        for split, dataset in datasets.items()
        if split != "train"
    }
    loaders["train"] = DataLoader(
        datasets["train"],
        sampler=RandomSampler(datasets["train"], generator=generator),
        **common,
    )
    return metadata, loaders
