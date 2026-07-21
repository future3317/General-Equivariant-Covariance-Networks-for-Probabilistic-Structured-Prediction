"""Tests for the lightweight ITOP depth-map pipeline."""

import h5py
import json
import numpy as np
import pytest
import torch

from data.itop_dataset import (
    ITOPCachedDataset,
    ITOPDepthDataset,
    ITOP_OUTPUT_GRAPH,
    compact_itop_labels,
    depth_to_point_cloud,
    itop_train_validation_indices,
)
from data.itop_features import get_itop_feature_loaders
from equivcompiler import FeatureSpec, GraphPrecision, plan_readout
from models import ControlledMeanOperatorHead, DeterministicHead, EquivariantBackbone
from representations import O3IrrepsSpec
from scripts.precompute_itop_geometry import write_itop_geometry_cache


def _write_fixture(tmp_path):
    depth_path = tmp_path / "depth.h5"
    labels_path = tmp_path / "labels.h5"
    depth = np.zeros((2, 240, 320), dtype=np.float16)
    depth[:, 120, 160] = 2.0
    depth[:, 120, 161] = 2.0
    depth[:, 121, 160] = 2.0
    depth[:, 119, 160] = 2.0
    with h5py.File(depth_path, "w") as target:
        target["data"] = depth
        target["id"] = np.array([b"00_00000", b"00_00001"])

    joints = np.zeros((2, 15, 3), dtype=np.float16)
    joints[..., 2] = 2.0
    with h5py.File(labels_path, "w") as target:
        target["id"] = np.array([b"00_00000", b"00_00001"])
        target["is_valid"] = np.array([1, 0], dtype=np.uint8)
        target["visible_joints"] = np.ones((2, 15), dtype=np.int16)
        target["image_coordinates"] = np.zeros((2, 15, 2), dtype=np.int16)
        target["real_world_coordinates"] = joints
        target["segmentation"] = np.zeros((2, 240, 320), dtype=np.int8)
    return depth_path, labels_path


def test_depth_to_point_cloud_uses_documented_itop_intrinsics():
    depth = np.zeros((240, 320), dtype=np.float32)
    depth[120, 160] = 2.0
    depth[120, 161] = 2.0
    points = depth_to_point_cloud(depth)
    assert points.shape == (2, 3)
    assert np.allclose(points[0], [0.0, 0.0, 2.0])
    assert np.allclose(points[1], [0.007, 0.0, 2.0])


def test_label_compaction_omits_segmentation(tmp_path):
    _, labels_path = _write_fixture(tmp_path)
    compact_path = compact_itop_labels(labels_path, tmp_path / "labels_compact.npz")
    with np.load(compact_path) as compact:
        assert "real_world_coordinates" in compact
        assert "visible_joints" in compact
        assert "segmentation" not in compact


def test_itop_dataset_centers_observable_cloud_without_label_leakage(tmp_path):
    depth_path, labels_path = _write_fixture(tmp_path)
    compact_path = compact_itop_labels(labels_path, tmp_path / "labels_compact.npz")
    dataset = ITOPDepthDataset(
        depth_path,
        compact_path,
        view="side",
        num_points=8,
        num_neighbors=2,
        training=False,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.pos.shape == (8, 3)
    assert sample.y_pose.shape == (1, 45)
    assert sample.visible_joints.shape == (1, 15)
    assert sample.edge_index.shape == (2, 16)
    assert torch.allclose(sample.pos.mean(0), torch.zeros(3), atol=5e-3)
    absolute_joints = sample.y_pose.reshape(15, 3) + sample.centroid
    assert torch.allclose(absolute_joints[:, 2], torch.full((15,), 2.0))


def test_itop_skeleton_has_tree_complexity():
    assert ITOP_OUTPUT_GRAPH.num_nodes == 15
    assert ITOP_OUTPUT_GRAPH.num_edges == 14
    assert ITOP_OUTPUT_GRAPH.num_potentials * 6 == 174


def test_itop_sample_runs_through_compiled_graph_precision_model(tmp_path):
    depth_path, labels_path = _write_fixture(tmp_path)
    dataset = ITOPDepthDataset(
        depth_path,
        labels_path,
        view="side",
        num_points=8,
        num_neighbors=2,
        num_basis=4,
        training=False,
    )
    sample = dataset[0]
    sample.batch = torch.zeros(sample.pos.shape[0], dtype=torch.long)
    backbone = EquivariantBackbone(
        hidden_dim=4,
        lmax=2,
        num_layers=1,
        num_basis=4,
        atom_feature_dim=4,
        atom_features="learnable",
    )
    plan = plan_readout(
        FeatureSpec.from_backbone(backbone),
        output=ITOP_OUTPUT_GRAPH.output_irreps,
        covariance=GraphPrecision(ITOP_OUTPUT_GRAPH),
        output_scope="global",
    )
    model = plan.bind(backbone)
    result = model(sample, target=sample.y_pose, return_precision=True)
    assert result["mu"].shape == (1, 45)
    assert result["params"].shape == (1, 174)
    assert result["precision"].shape == (1, 45, 45)
    assert torch.isfinite(result["loss"])


def test_controlled_itop_readout_shares_mean_and_compiles_operator_parameters():
    backbone = EquivariantBackbone(
        hidden_dim=4,
        lmax=2,
        num_layers=1,
        num_basis=4,
        atom_feature_dim=4,
        atom_features="learnable",
    )
    output = O3IrrepsSpec(ITOP_OUTPUT_GRAPH.output_irreps)
    plan = plan_readout(
        FeatureSpec.from_backbone(backbone),
        output=output.irreps,
        covariance=GraphPrecision(ITOP_OUTPUT_GRAPH),
        output_scope="global",
    )
    direct_mean = DeterministicHead(backbone.irreps_out, output, pool=True)
    controlled = ControlledMeanOperatorHead(
        direct_mean,
        plan.compilation.build_head(),
    )
    features = torch.randn(7, backbone.irreps_out.dim)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    mean, parameters = controlled(features, batch)
    torch.testing.assert_close(mean, direct_mean(features, batch))
    assert parameters.shape == (2, 174)
    assert not any(
        parameter.requires_grad
        for parameter in controlled.operator_head.mean_projection.parameters()
    )


def test_precomputed_itop_geometry_is_exact_and_gpu_featurizable(tmp_path):
    root = tmp_path / "ITOP"
    root.mkdir()
    depth_path, labels_path = _write_fixture(tmp_path)
    expected_depth = root / "ITOP_side_train_depth_map.h5"
    expected_labels = root / "ITOP_side_train_labels.h5"
    depth_path.replace(expected_depth)
    labels_path.replace(expected_labels)
    cache = write_itop_geometry_cache(
        root,
        view="side",
        split="train",
        num_points=8,
        num_neighbors=2,
    )
    dataset = ITOPCachedDataset(
        cache,
        view="side",
        num_points=8,
        num_neighbors=2,
    )
    sample = dataset[0]
    assert sample.edge_index.shape == (2, 16)
    assert not hasattr(sample, "edge_sh")
    sample.batch = torch.zeros(8, dtype=torch.long)
    backbone = EquivariantBackbone(
        hidden_dim=4,
        max_radius=0.5,
        lmax=2,
        num_layers=1,
        num_basis=4,
        atom_feature_dim=4,
        atom_features="learnable",
    )
    node_features, batch = backbone(sample)
    assert node_features.shape[0] == 8
    assert batch.shape == (8,)


def _write_feature_cache(root, checkpoint, *, checkpoint_hash):
    from data.itop_features import sha256_file

    root.mkdir()
    fields = {
        "features": torch.randn(20, 4),
        "target": torch.randn(20, 45),
        "visible_joints": torch.ones(20, 15, dtype=torch.bool),
        "frame_index": torch.arange(20),
        "view_id": torch.zeros(20, dtype=torch.long),
    }
    torch.save(fields, root / "side_train.pt")
    torch.save(
        {name: value[:4] for name, value in fields.items()}, root / "side_test.pt"
    )
    top = {name: value[:3] for name, value in fields.items()}
    top["view_id"] = torch.ones(3, dtype=torch.long)
    torch.save(top, root / "top_test.pt")
    metadata = {
        "backbone_checkpoint_sha256": (
            sha256_file(checkpoint) if checkpoint_hash == "actual" else checkpoint_hash
        )
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_feature_cache_reuses_canonical_seed_split(tmp_path):
    checkpoint = tmp_path / "backbone.pt"
    torch.save({"state": 1}, checkpoint)
    cache = tmp_path / "features"
    _write_feature_cache(cache, checkpoint, checkpoint_hash="actual")
    train, validation, side, top, _ = get_itop_feature_loaders(
        cache,
        backbone_checkpoint=checkpoint,
        seed=17,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    expected_train, expected_validation = itop_train_validation_indices(20, seed=17)
    assert train.dataset.indices == expected_train
    assert validation.dataset.indices == expected_validation
    assert len(side.dataset) == 4
    assert len(top.dataset) == 3


def test_feature_cache_rejects_wrong_backbone_checkpoint(tmp_path):
    checkpoint = tmp_path / "backbone.pt"
    torch.save({"state": 1}, checkpoint)
    cache = tmp_path / "features"
    _write_feature_cache(cache, checkpoint, checkpoint_hash="not-the-checkpoint")
    with pytest.raises(ValueError, match="different backbone checkpoint"):
        get_itop_feature_loaders(
            cache,
            backbone_checkpoint=checkpoint,
            seed=17,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )
