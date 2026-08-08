"""Label-free descriptors computed from immutable observed point-cloud caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def point_cloud_observation_descriptors(
    cache: str | Path, *, view_id: int
) -> dict[str, torch.Tensor]:
    """Return inference-time geometry descriptors and diagnostic visibility.

    Visibility is returned under an explicit diagnostic-only key and must not
    be used by observation-only prediction probes.
    """
    cache = Path(cache)
    points = np.load(cache / "points.npy", mmap_mode="r")
    neighbours = np.load(cache / "neighbors.npy", mmap_mode="r")
    centroids = torch.from_numpy(np.array(np.load(cache / "centroids.npy"))).float()
    visible = torch.from_numpy(
        np.array(np.load(cache / "visible_joints.npy"), dtype=np.bool_)
    )
    frame = torch.from_numpy(np.array(np.load(cache / "frame_indices.npy"))).long()
    records: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(points), 128):
        stop = min(start + 128, len(points))
        cloud = torch.from_numpy(np.array(points[start:stop])).float()
        index = torch.from_numpy(np.array(neighbours[start:stop], dtype=np.int64))
        radius = torch.linalg.vector_norm(cloud, dim=-1)
        covariance = torch.einsum("bpi,bpj->bij", cloud, cloud) / cloud.shape[1]
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-12)
        extent = cloud.amax(1) - cloud.amin(1)
        batch = torch.arange(cloud.shape[0])[:, None, None]
        neighbour_points = cloud[batch, index]
        distances = torch.linalg.vector_norm(
            cloud[:, :, None, :] - neighbour_points, dim=-1
        )
        chunk = {
            "centroid_norm": torch.linalg.vector_norm(centroids[start:stop], dim=-1),
            "centroid_depth": centroids[start:stop, 2],
            "radius_mean": radius.mean(1),
            "radius_std": radius.std(1, unbiased=False),
            "radius_q90": torch.quantile(radius, 0.90, dim=1),
            "covariance_eigenvalue_min": eigenvalues[:, 0],
            "covariance_eigenvalue_mid": eigenvalues[:, 1],
            "covariance_eigenvalue_max": eigenvalues[:, 2],
            "covariance_anisotropy": eigenvalues[:, 2] / eigenvalues.sum(1),
            "extent_volume": extent.prod(1),
            "knn_distance_mean": distances.mean((1, 2)),
            "knn_distance_q90": torch.quantile(distances.flatten(1), 0.90, dim=1),
        }
        for name, values in chunk.items():
            records.setdefault(name, []).append(values)
    result = {name: torch.cat(values) for name, values in records.items()}
    result["visible_fraction_diagnostic_only"] = visible.float().mean(1)
    result["sample_id"] = frame + view_id * (1 << 32)
    return result
