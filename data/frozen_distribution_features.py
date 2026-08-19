"""Validated task-neutral caches for frozen-H,mu distribution experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from compatibility.e3nn import o3
from data.observation_descriptors import (
    o3_invariant_descriptor_names,
    observation_descriptor_semantics,
)
from scripts.itop_reproducibility import sha256_file

REQUIRED_FIELDS = ("features", "mean", "params", "target", "sample_id")
_FORBIDDEN_OBSERVATION_DESCRIPTOR_TOKENS = (
    "visible",
    "target",
    "label",
    "ground_truth",
)


def validate_observation_descriptors(
    descriptors: dict[str, torch.Tensor], *, count: int
) -> tuple[str, ...]:
    """Validate label-free invariant descriptor tensors before model use.

    Descriptor names are intentionally checked here, at the cache boundary,
    so an observation-aware uncertainty head cannot silently consume visibility
    or target-derived fields.  The function returns insertion order, which is
    the serialized feature contract used by the optional runner path.
    """
    if count < 1:
        raise ValueError("descriptor count must be positive")
    if not descriptors:
        raise ValueError("at least one observation descriptor is required")
    names: list[str] = []
    for name, values in descriptors.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("observation descriptor names must be non-empty strings")
        normalized = name.lower()
        if any(token in normalized for token in _FORBIDDEN_OBSERVATION_DESCRIPTOR_TOKENS):
            raise ValueError(
                f"observation descriptor {name!r} is label-derived or diagnostic-only"
            )
        if not isinstance(values, torch.Tensor) or values.ndim != 1:
            raise ValueError(f"observation descriptor {name!r} must have shape (N,)")
        if values.shape[0] != count:
            raise ValueError(
                f"observation descriptor {name!r} has count {values.shape[0]}, expected {count}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"observation descriptor {name!r} contains non-finite values")
        names.append(name)
    return tuple(names)


def invariant_irrep_summary(
    features: torch.Tensor, irreps: str | o3.Irreps
) -> torch.Tensor:
    """Summarize typed features into legal O(3)-invariant scalar probes."""
    irreps = o3.Irreps(irreps)
    if features.shape[-1] != irreps.dim:
        raise ValueError("feature dimension does not match declared irreps")
    summaries = []
    for (multiplicity, irrep), feature_slice in zip(irreps, irreps.slices()):
        block = features[..., feature_slice].reshape(
            *features.shape[:-1], multiplicity, irrep.dim
        )
        if irrep.l == 0 and irrep.p == 1:
            summaries.append(block.squeeze(-1))
        else:
            summaries.append(torch.linalg.vector_norm(block, dim=-1))
    return torch.cat(summaries, dim=-1)


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
        self._observation_descriptors: torch.Tensor | None = None

    def attach_observation_descriptors(
        self,
        sample_ids: torch.Tensor,
        descriptors: torch.Tensor,
        names: list[str] | tuple[str, ...],
    ) -> None:
        """Attach an aligned, optional label-free descriptor matrix in memory."""
        if sample_ids.ndim != 1 or descriptors.ndim != 2:
            raise ValueError("descriptor sample IDs/matrix must have shapes (N,) and (N,M)")
        if sample_ids.shape[0] != descriptors.shape[0]:
            raise ValueError("descriptor sample IDs and values have inconsistent counts")
        if len(names) != descriptors.shape[1]:
            raise ValueError("descriptor names do not match descriptor width")
        descriptor_columns = {
            name: descriptors[:, index]
            for index, name in enumerate(names)
        }
        validate_observation_descriptors(
            descriptor_columns, count=int(sample_ids.shape[0])
        )
        o3_invariant_descriptor_names(tuple(names))
        if torch.unique(sample_ids).numel() != sample_ids.numel():
            raise ValueError("descriptor sample IDs must be unique")
        if torch.unique(self.payload["sample_id"]).numel() != len(self):
            raise ValueError("frozen cache sample IDs must be unique")
        descriptor_by_id = {
            int(identifier): index for index, identifier in enumerate(sample_ids.tolist())
        }
        try:
            indices = torch.tensor(
                [descriptor_by_id[int(identifier)] for identifier in self.payload["sample_id"].tolist()],
                dtype=torch.long,
            )
        except KeyError as error:
            raise ValueError("descriptor sample IDs do not cover the frozen cache") from error
        aligned = descriptors.index_select(0, indices).float().contiguous()
        if not bool(torch.isfinite(aligned).all()):
            raise ValueError("aligned observation descriptors are non-finite")
        self._observation_descriptors = aligned

    def __len__(self) -> int:
        return int(self.payload["features"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {field: self.payload[field][index] for field in REQUIRED_FIELDS}
        if self._observation_descriptors is not None:
            result["observation_descriptors"] = self._observation_descriptors[index]
        return result


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
    observation_descriptor_dir: str | Path | None = None,
) -> tuple[dict, dict[str, DataLoader]]:
    metadata, datasets = load_frozen_distribution_cache(cache_dir)
    if observation_descriptor_dir is not None:
        descriptor_root = Path(observation_descriptor_dir)
        descriptor_metadata = json.loads(
            (descriptor_root / "metadata.json").read_text(encoding="utf-8")
        )
        names = tuple(descriptor_metadata["descriptor_names"])
        for split, dataset in datasets.items():
            descriptor_path = descriptor_root / f"{split}.pt"
            if not descriptor_path.is_file():
                raise ValueError(f"missing observation descriptor split: {descriptor_path}")
            descriptor_payload = torch.load(
                descriptor_path, map_location="cpu", weights_only=True
            )
            dataset.attach_observation_descriptors(
                descriptor_payload["sample_id"],
                descriptor_payload["descriptors"],
                names,
            )
        metadata = dict(metadata)
        metadata["observation_descriptors"] = {
            "directory": str(descriptor_root.resolve()),
            "names": list(names),
            "semantics": {
                name: observation_descriptor_semantics(name) for name in names
            },
            "metadata_sha256": sha256_file(descriptor_root / "metadata.json"),
        }
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
