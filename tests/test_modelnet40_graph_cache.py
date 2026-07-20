"""Tests for exact, compact ModelNet40 k-NN precomputation."""

from __future__ import annotations

import joblib
import numpy as np
import pytest
import torch

import data.modelnet40_inertia_dataset as modelnet40_dataset
from data.modelnet40_inertia_dataset import ModelNet40InertiaDataset
from data.point_cloud_graph import knn_graph
from scripts.precompute_modelnet40_graphs import write_graph_cache


def _write_source_cache(path, *, num_points: int = 8) -> None:
    generator = np.random.default_rng(42)
    train_points = generator.normal(size=(4, num_points, 3)).astype(np.float32)
    test_points = generator.normal(size=(2, num_points, 3)).astype(np.float32)
    payload = {
        "train": {
            "points": train_points,
            "inertia": generator.normal(size=(4, 6)).astype(np.float32),
            "labels": np.arange(4, dtype=np.int64),
        },
        "test": {
            "points": test_points,
            "inertia": generator.normal(size=(2, 6)).astype(np.float32),
            "labels": np.arange(2, dtype=np.int64),
        },
        "stats": {"std": np.ones(6, dtype=np.float32)},
    }
    joblib.dump(payload, path)


def test_cached_neighbors_match_online_knn(tmp_path, monkeypatch):
    source = tmp_path / "modelnet.pkl"
    graph_cache = tmp_path / "modelnet.knn_n8_k2.pt"
    _write_source_cache(source)
    write_graph_cache(source, graph_cache, num_points=8, num_neighbors=2)

    dataset = ModelNet40InertiaDataset(
        cache_path=source,
        graph_cache_path=graph_cache,
        split="train",
        num_points=8,
        num_neighbors=2,
    )
    expected = knn_graph(torch.from_numpy(dataset.points[0]).float(), k=2)

    def fail_if_recomputed(*_args, **_kwargs):
        pytest.fail("knn_graph must not run when a graph cache is configured")

    monkeypatch.setattr(modelnet40_dataset, "knn_graph", fail_if_recomputed)
    actual = dataset[0].edge_index
    torch.testing.assert_close(actual, expected)


def test_explicit_missing_graph_cache_is_an_error(tmp_path):
    source = tmp_path / "modelnet.pkl"
    _write_source_cache(source)
    with pytest.raises(FileNotFoundError, match="precomputed graph cache"):
        ModelNet40InertiaDataset(
            cache_path=source,
            graph_cache_path=tmp_path / "missing.pt",
            num_points=8,
            num_neighbors=2,
        )
