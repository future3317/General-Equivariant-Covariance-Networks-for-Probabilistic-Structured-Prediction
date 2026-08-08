import json

import numpy as np

from data.observation_descriptors import point_cloud_observation_descriptors


def test_geometry_descriptors_use_observations_and_keep_visibility_diagnostic(tmp_path):
    points = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    neighbours = np.array(
        [[[1, 2], [0, 3], [0, 3], [1, 2]], [[1, 2], [0, 3], [0, 3], [1, 2]]],
        dtype=np.uint16,
    )
    np.save(tmp_path / "points.npy", points)
    np.save(tmp_path / "neighbors.npy", neighbours)
    np.save(tmp_path / "centroids.npy", np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]))
    np.save(tmp_path / "visible_joints.npy", np.ones((2, 15), dtype=np.bool_))
    np.save(tmp_path / "frame_indices.npy", np.array([7, 9], dtype=np.int64))
    (tmp_path / "metadata.json").write_text(json.dumps({"test": True}))

    result = point_cloud_observation_descriptors(tmp_path, view_id=1)
    assert result["sample_id"].tolist() == [(1 << 32) + 7, (1 << 32) + 9]
    assert result["radius_mean"].shape == (2,)
    assert result["knn_distance_mean"].shape == (2,)
    assert result["visible_fraction_diagnostic_only"].tolist() == [1.0, 1.0]
