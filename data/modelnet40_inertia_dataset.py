"""ModelNet40 point-cloud dataset adapter for the GECN framework.

Loads precomputed point clouds and inertia tensors from the ICML cache and
converts each sample into a PyG ``Data`` object compatible with the existing
``EquivariantBackbone``. Point clouds are turned into k-NN graphs; edge
spherical harmonics, radial basis functions, and cutoff weights are computed
on the fly.

Supports two targets:

* ``'inertia'``: the 3x3 inertia tensor of the point cloud (precomputed).
* ``'shape_covariance'``: the 3x3 shape covariance / second-moment tensor
  computed on the fly from the centered point cloud.

Target tensors are normalized by a single scalar standard deviation shared
across all Voigt components. Scalar scaling commutes with O(3) rotations and
therefore preserves the equivariance of the learning problem.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import joblib
import numpy as np
import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from data.tensor_conversions import voigt_to_irreps


DEFAULT_CACHE_PATH = "data/modelnet40/cache/modelnet40_inertia_dataset.pkl"


def _knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Pure-PyTorch k-NN graph (no torch-cluster dependency).

    Args:
        pos: (N, 3) point coordinates.
        k: Number of nearest neighbors.

    Returns:
        edge_index: (2, N*k) tensor of directed edges.
    """
    N = pos.shape[0]
    distances = torch.cdist(pos, pos)
    distances.fill_diagonal_(float("inf"))
    _, knn_indices = torch.topk(distances, k=k, largest=False, dim=-1)
    src = torch.arange(N, device=pos.device).repeat_interleave(k)
    dst = knn_indices.flatten()
    return torch.stack([src, dst], dim=0)


def _compute_edge_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    max_radius: float,
    num_basis: int,
    lmax: int,
) -> dict[str, torch.Tensor]:
    """Compute edge vectors, spherical harmonics, RBF, and cutoff weights."""
    row, col = edge_index
    edge_vec = pos[col] - pos[row]
    edge_len = edge_vec.norm(dim=-1)

    irreps_sh = o3.Irreps.spherical_harmonics(lmax)
    edge_sh = o3.spherical_harmonics(
        irreps_sh,
        edge_vec,
        normalize=True,
        normalization="component",
    )

    edge_rbf = soft_one_hot_linspace(
        edge_len,
        start=0.0,
        end=max_radius,
        number=num_basis,
        basis="gaussian",
        cutoff=False,
    )

    # Smooth cutoff that goes to zero at max_radius.
    edge_weights = 0.5 * (torch.cos(np.pi * edge_len / max_radius) + 1.0)
    edge_weights = edge_weights * (edge_len < max_radius).float()

    return {
        "edge_vec": edge_vec,
        "edge_sh": edge_sh,
        "edge_rbf": edge_rbf,
        "edge_weights": edge_weights,
    }


def _shape_covariance_voigt(points: np.ndarray) -> np.ndarray:
    """Return the 6D Voigt vector of the shape covariance tensor.

    S = (1/N) sum_k (p_k - mu) (p_k - mu)^T.
    """
    centered = points - points.mean(axis=0)
    S = (centered.T @ centered) / len(points)
    S = 0.5 * (S + S.T)
    return np.array([
        S[0, 0], S[1, 1], S[2, 2],
        S[1, 2], S[0, 2], S[0, 1],
    ], dtype=np.float32)


def _scalar_normalize_voigt(
    target_voigt: np.ndarray,
    stats: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize a Voigt target by a single scalar standard deviation.

    A scalar scale commutes with the O(3) representation matrix, so this
    normalization preserves equivariance. No mean shift is applied because a
    scalar shift vector is generally not invariant under rotations.
    """
    target = torch.from_numpy(target_voigt).float()
    std_all = torch.from_numpy(stats["std"]).float()
    global_std = std_all.mean().item()
    if global_std < 1e-8:
        global_std = 1.0
    std = torch.full_like(std_all, global_std)
    mean = torch.zeros_like(std_all)
    return target / std, mean, std


def _point_cloud_to_data(
    points: np.ndarray,
    target_voigt: np.ndarray,
    label: int,
    stats: dict[str, np.ndarray],
    num_points: int,
    num_neighbors: int,
    max_radius: float,
    num_basis: int,
    lmax: int,
) -> Data:
    """Convert a single point cloud + symmetric tensor target to a PyG Data object."""
    points = points[:num_points]
    pos = torch.from_numpy(points).float()

    # k-NN graph. Equivariant: rotating the cloud rotates edge_vec.
    edge_index = _knn_graph(pos, k=num_neighbors)

    edge_features = _compute_edge_features(
        pos, edge_index, max_radius, num_basis, lmax
    )

    # Constant atomic type for all points -> single learnable embedding vector.
    z = torch.zeros(pos.shape[0], dtype=torch.long)

    # Scalar-normalize target in Voigt space, then convert to 0e+2e irrep coefficients.
    target_voigt_norm, mean, std = _scalar_normalize_voigt(target_voigt, stats)
    y_irreps = voigt_to_irreps(target_voigt_norm).unsqueeze(0)

    data = Data(
        pos=pos,
        z=z,
        edge_index=edge_index,
        edge_sh=edge_features["edge_sh"],
        edge_rbf=edge_features["edge_rbf"],
        edge_weights=edge_features["edge_weights"],
        y_irreps=y_irreps,
        y_voigt_mean=mean,
        y_voigt_std=std,
        label=torch.tensor(label, dtype=torch.long),
        num_neighbors=num_neighbors,
    )
    data.batch = torch.zeros(pos.shape[0], dtype=torch.long)
    return data


class ModelNet40InertiaDataset(Dataset):
    """ModelNet40 point-cloud dataset with symmetric rank-2 tensor targets.

    The precomputed cache is expected to contain ``train``, ``test`` and
    ``stats`` dictionaries, where each split has ``points`` (N, P, 3),
    ``inertia`` (N, 6) in Voigt notation, and ``labels`` (N,).

    Targets are always normalized by a single scalar standard deviation so that
    the learning problem remains strictly equivariant under O(3).

    Args:
        target_type: ``'inertia'`` uses the precomputed inertia tensor;
            ``'shape_covariance'`` computes the second-moment tensor from the
            point cloud on the fly.
    """

    def __init__(
        self,
        cache_path: str = DEFAULT_CACHE_PATH,
        split: str = "train",
        target_type: str = "inertia",
        num_points: int = 1024,
        num_neighbors: int = 16,
        max_radius: float = 2.0,
        num_basis: int = 8,
        lmax: int = 2,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        if target_type not in {"inertia", "shape_covariance"}:
            raise ValueError(
                f"target_type must be 'inertia' or 'shape_covariance', got {target_type}"
            )

        self.cache_path = cache_path
        self.split = split
        self.target_type = target_type
        self.num_points = num_points
        self.num_neighbors = num_neighbors
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.lmax = lmax

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"ModelNet40 cache not found: {cache_path}")

        self._data = self._load_cache(cache_path)
        self.points = self._data[split]["points"]
        self.labels = self._data[split]["labels"]

        if target_type == "inertia":
            self.targets_voigt = self._data[split]["inertia"]
            self.stats = self._data["stats"]
        else:
            # Compute shape-covariance targets from the same subsampled points that
            # the model sees. This keeps input and target consistent.
            self.targets_voigt = np.stack(
                [_shape_covariance_voigt(p[:num_points]) for p in self.points],
                axis=0,
            )
            if split == "train":
                self.stats = {
                    "mean": self.targets_voigt.mean(axis=0),
                    "std": self.targets_voigt.std(axis=0),
                }
            else:
                # Reuse training statistics.
                train_targets = np.stack(
                    [_shape_covariance_voigt(p[:num_points]) for p in self._data["train"]["points"]],
                    axis=0,
                )
                self.stats = {
                    "mean": train_targets.mean(axis=0),
                    "std": train_targets.std(axis=0),
                }

        # Guard against zero std.
        self.stats["std"] = np.maximum(self.stats["std"], 1e-8)

    @staticmethod
    def _load_cache(cache_path: str) -> dict[str, Any]:
        try:
            return joblib.load(cache_path)
        except Exception:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> Data:
        return _point_cloud_to_data(
            self.points[idx],
            self.targets_voigt[idx],
            int(self.labels[idx]),
            self.stats,
            self.num_points,
            self.num_neighbors,
            self.max_radius,
            self.num_basis,
            self.lmax,
        )


def get_modelnet40_inertia_loaders(
    cache_path: str = DEFAULT_CACHE_PATH,
    target_type: str = "inertia",
    batch_size: int = 16,
    num_points: int = 1024,
    num_neighbors: int = 16,
    max_radius: float = 2.0,
    num_basis: int = 8,
    lmax: int = 2,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    val_frac: float = 0.1,
    seed: int = 42,
):
    """Create train/val/test PyG data loaders for ModelNet40.

    The original cache only has ``train`` and ``test`` splits. A validation
    set is carved out of the training split using ``val_frac``.
    """
    full_train = ModelNet40InertiaDataset(
        cache_path=cache_path,
        split="train",
        target_type=target_type,
        num_points=num_points,
        num_neighbors=num_neighbors,
        max_radius=max_radius,
        num_basis=num_basis,
        lmax=lmax,
    )
    test_dataset = ModelNet40InertiaDataset(
        cache_path=cache_path,
        split="test",
        target_type=target_type,
        num_points=num_points,
        num_neighbors=num_neighbors,
        max_radius=max_radius,
        num_basis=num_basis,
        lmax=lmax,
    )

    n_val = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    loader_kwargs: dict = {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = PyGDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = PyGDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs
    )
    test_loader = PyGDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs
    )

    return train_loader, val_loader, test_loader
