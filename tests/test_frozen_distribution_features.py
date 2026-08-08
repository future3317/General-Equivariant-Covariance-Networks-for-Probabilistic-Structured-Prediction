import json

import pytest
import torch

from data.frozen_distribution_features import load_frozen_distribution_cache
from scripts.itop_reproducibility import sha256_file


def _write_cache(root):
    splits = {}
    for split, count in (("train", 5), ("val", 3), ("test", 4)):
        payload = {
            "features": torch.randn(count, 6),
            "mean": torch.randn(count, 3),
            "params": torch.randn(count, 3, 3),
            "target": torch.randn(count, 3),
            "sample_id": torch.arange(count),
        }
        path = root / f"{split}.pt"
        torch.save(payload, path)
        splits[split] = {"count": count, "sha256": sha256_file(path)}
    (root / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "splits": splits}), encoding="utf-8"
    )


def test_frozen_distribution_cache_validates_counts_and_hashes(tmp_path):
    _write_cache(tmp_path)
    _, datasets = load_frozen_distribution_cache(tmp_path)
    assert {name: len(dataset) for name, dataset in datasets.items()} == {
        "train": 5,
        "val": 3,
        "test": 4,
    }


def test_frozen_distribution_cache_rejects_changed_artifact(tmp_path):
    _write_cache(tmp_path)
    torch.save({"broken": torch.tensor(1)}, tmp_path / "val.pt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_distribution_cache(tmp_path)
