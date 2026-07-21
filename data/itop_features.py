"""Frozen-backbone feature caches for controlled ITOP head comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from data.itop_dataset import itop_train_validation_indices


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ITOPFeatureDataset(Dataset):
    """In-memory pooled equivariant features with aligned pose labels."""

    def __init__(self, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {
            "features",
            "target",
            "visible_joints",
            "frame_index",
            "view_id",
        }
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f"invalid ITOP feature cache: {path}")
        count = payload["features"].shape[0]
        if any(payload[name].shape[0] != count for name in required):
            raise ValueError("ITOP feature cache fields have inconsistent lengths")
        self.payload = payload

    def __len__(self) -> int:
        return self.payload["features"].shape[0]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        return {name: value[item] for name, value in self.payload.items()}


def get_itop_feature_loaders(
    cache_dir: str | Path,
    *,
    backbone_checkpoint: str | Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    val_fraction: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, dict]:
    """Load seed-split train/validation and aligned side/top test features."""
    root = Path(cache_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    actual_hash = sha256_file(backbone_checkpoint)
    if metadata["backbone_checkpoint_sha256"] != actual_hash:
        raise ValueError(
            "feature cache was produced by a different backbone checkpoint"
        )
    train_full = ITOPFeatureDataset(root / "side_train.pt")
    side_test = ITOPFeatureDataset(root / "side_test.pt")
    top_test = ITOPFeatureDataset(root / "top_test.pt")
    train_indices, validation_indices = itop_train_validation_indices(
        len(train_full), seed=seed, val_fraction=val_fraction
    )
    validation = Subset(train_full, validation_indices)
    train = Subset(train_full, train_indices)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_sampler = RandomSampler(
        train,
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(
            train,
            sampler=train_sampler,
            generator=torch.Generator().manual_seed(seed + 1),
            **kwargs,
        ),
        DataLoader(validation, shuffle=False, **kwargs),
        DataLoader(side_test, shuffle=False, **kwargs),
        DataLoader(top_test, shuffle=False, **kwargs),
        metadata,
    )
