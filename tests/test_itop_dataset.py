"""Tests for the lightweight ITOP depth-map pipeline."""

import h5py
import numpy as np
import torch

from data.itop_dataset import (
    ITOPDepthDataset,
    ITOP_OUTPUT_GRAPH,
    compact_itop_labels,
    depth_to_point_cloud,
)
from models import EquivariantBackbone
from representations import CompilerConfig, O3RepresentationCompiler


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
    compilation = O3RepresentationCompiler.for_graph(
        ITOP_OUTPUT_GRAPH,
        CompilerConfig(covariance="graph", output_scope="global"),
    ).compile(backbone.irreps_out)
    model = compilation.build_model(backbone)
    result = model(sample, target=sample.y_pose, return_precision=True)
    assert result["mu"].shape == (1, 45)
    assert result["params"].shape == (1, 29, 3, 3)
    assert result["precision"].shape == (1, 45, 45)
    assert torch.isfinite(result["loss"])
