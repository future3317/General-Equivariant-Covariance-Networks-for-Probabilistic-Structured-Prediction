"""Tests for cache-friendly dielectric graph shards."""

import json
from unittest import mock

import torch

from compatibility.torch_geometric import Data
from data.dielectric_dataset import DielectricIrrepsDataset
from dielectric_data_loader import DielectricDataset, ShardBatchSampler
from scripts.shard_dielectric_graphs import shard_split


def _write_graph_files(root, count=7):
    graph_dir = root / "train_graphs_full"
    graph_dir.mkdir()
    metadata = {
        "log_mean": 0.0,
        "log_std": 1.0,
        "component_mean": [0.0] * 6,
        "component_std": [1.0] * 6,
    }
    (graph_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for index in range(count):
        graph = Data(
            node_features=torch.full((3, 2), float(index)),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_sh=torch.randn(2, 9),
            edge_rbf=torch.randn(2, 4),
            edge_weights=torch.ones(2),
            y=torch.arange(6, dtype=torch.float32) + index,
            pre_idx=torch.tensor(index),
        )
        torch.save(graph, graph_dir / f"{index}.pt")


def test_sharded_dataset_matches_per_file_storage(tmp_path):
    _write_graph_files(tmp_path)
    shard_split(tmp_path, "train", shard_size=3)
    per_file = DielectricDataset(tmp_path, "train", storage="files")
    sharded = DielectricDataset(tmp_path, "train", storage="shards")
    assert len(per_file) == len(sharded) == 7
    assert sharded.shard_ranges == ((0, 3), (3, 6), (6, 7))
    for index in range(len(per_file)):
        expected = per_file[index]
        actual = sharded[index]
        torch.testing.assert_close(actual.node_features, expected.node_features)
        torch.testing.assert_close(actual.edge_sh, expected.edge_sh)
        torch.testing.assert_close(actual.y, expected.y)


def test_shard_cache_loads_neighboring_samples_once(tmp_path):
    _write_graph_files(tmp_path)
    shard_split(tmp_path, "train", shard_size=4)
    dataset = DielectricDataset(tmp_path, "train", storage="shards", shard_cache_size=1)
    real_load = torch.load
    with mock.patch(
        "dielectric_data_loader.torch.load", wraps=real_load
    ) as mocked_load:
        dataset[0]
        dataset[1]
        dataset[3]
        assert mocked_load.call_count == 1
        dataset[4]
        assert mocked_load.call_count == 2


def test_shard_batch_sampler_covers_each_sample_once():
    torch.manual_seed(3)
    sampler = ShardBatchSampler(
        ((0, 3), (3, 7), (7, 10)),
        batch_size=4,
        shuffle=True,
        drop_last=False,
    )
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    assert len(batches) == len(sampler) == 3
    assert sorted(flattened) == list(range(10))
    assert all(1 <= len(batch) <= 4 for batch in batches)


def test_irrep_wrapper_produces_identical_targets_from_shards(tmp_path):
    _write_graph_files(tmp_path)
    shard_split(tmp_path, "train", shard_size=3)
    per_file = DielectricIrrepsDataset(tmp_path, "train", lmax=2, storage="files")
    sharded = DielectricIrrepsDataset(tmp_path, "train", lmax=2, storage="shards")
    for index in range(len(per_file)):
        expected = per_file[index]
        actual = sharded[index]
        torch.testing.assert_close(actual.edge_sh, expected.edge_sh)
        torch.testing.assert_close(actual.y_km, expected.y_km)
        torch.testing.assert_close(actual.y_irreps, expected.y_irreps)
