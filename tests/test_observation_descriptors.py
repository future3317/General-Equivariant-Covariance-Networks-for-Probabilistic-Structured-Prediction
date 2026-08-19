from __future__ import annotations

import numpy as np
import pytest
import torch

from data.observation_descriptors import (
    O3_INVARIANT_DESCRIPTOR_NAMES,
    o3_invariant_descriptor_names,
    observation_descriptor_semantics,
    point_cloud_observation_descriptors,
)


def _orthogonal_matrix(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(3, 3))
    q, _ = np.linalg.qr(matrix)
    q[:, 0] *= -1.0
    return q


def _write_cache(path, points: np.ndarray) -> None:
    sample_count, point_count, _ = points.shape
    neighbours = np.tile(np.arange(min(4, point_count)), (sample_count, point_count, 1))
    np.save(path / "points.npy", points)
    np.save(path / "neighbors.npy", neighbours)
    np.save(path / "centroids.npy", points.mean(axis=1))
    np.save(path / "visible_joints.npy", np.ones((sample_count, 15), dtype=bool))
    np.save(path / "frame_indices.npy", np.arange(sample_count, dtype=np.int64))


def test_only_o3_invariant_descriptors_enter_equivariant_scalar_contract():
    assert all(
        observation_descriptor_semantics(name) == "o3_invariant_scalar"
        for name in O3_INVARIANT_DESCRIPTOR_NAMES
    )
    assert observation_descriptor_semantics("centroid_depth") == "camera_frame_scalar"
    with pytest.raises(ValueError, match="camera-frame"):
        o3_invariant_descriptor_names(["centroid_norm", "centroid_depth"])


def test_o3_invariant_descriptors_are_stable_under_random_reflection(tmp_path):
    generator = np.random.default_rng(42)
    points = generator.normal(size=(3, 12, 3)).astype(np.float32)
    transformed = np.einsum("bpi,ij->bpj", points, _orthogonal_matrix(7))
    original_path = tmp_path / "original"
    transformed_path = tmp_path / "transformed"
    original_path.mkdir()
    transformed_path.mkdir()
    _write_cache(original_path, points)
    _write_cache(transformed_path, transformed)

    original = point_cloud_observation_descriptors(original_path, view_id=0)
    rotated = point_cloud_observation_descriptors(transformed_path, view_id=0)
    for name in O3_INVARIANT_DESCRIPTOR_NAMES:
        torch.testing.assert_close(original[name], rotated[name], rtol=1e-5, atol=1e-6)
