"""ModelNet40 inertia tensor dataset adapter for the GECN framework.

Loads precomputed point clouds and inertia tensors from the ICML cache and
converts each sample into a PyG ``Data`` object compatible with the existing
``EquivariantBackbone``. Point clouds are turned into k-NN graphs; edge
spherical harmonics, radial basis functions, and cutoff weights are computed
on the fly.
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

from data.tensor_conversions import voigt_to_irreps, irreps_to_voigt


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


def _point_cloud_to_data(
    points: np.ndarray,
    inertia_voigt: np.ndarray,
    label: int,
    stats: dict[str, np.ndarray],
    num_points: int,
    num_neighbors: int,
    max_radius: float,
    num_basis: int,
    lmax: int,
) -> Data:
    """Convert a single point cloud + inertia tensor to a PyG Data object."""
    points = points[:num_points]
    pos = torch.from_numpy(points).float()

    # k-NN graph. Equivariant: rotating the cloud rotates edge_vec.
    edge_index = _knn_graph(pos, k=num_neighbors)

    edge_features = _compute_edge_features(
        pos, edge_index, max_radius, num_basis, lmax
    )

    # Constant atomic type for all points -> single learnable embedding vector.
    z = torch.zeros(pos.shape[0], dtype=torch.long)

    # Normalize inertia in Voigt space, then convert to 0e+2e irrep coefficients.
    mean = torch.from_numpy(stats["mean"]).float()
    std = torch.from_numpy(stats["std"]).float()
    inertia_voigt_t = torch.from_numpy(inertia_voigt).float()
    inertia_voigt_norm = (inertia_voigt_t - mean) / std
    y_irreps = voigt_to_irreps(inertia_voigt_norm).unsqueeze(0)

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
    """ModelNet40 point-cloud dataset with inertia-tensor targets.

    The precomputed cache is expected to contain ``train``, ``test`` and
    ``stats`` dictionaries, where each split has ``points`` (N, P, 3),
    ``inertia`` (N, 6) in Voigt notation, and ``labels`` (N,).
    """

    def __init__(
        self,
        cache_path: str = DEFAULT_CACHE_PATH,
        split: str = "train",
        num_points: int = 1024,
        num_neighbors: int = 16,
        max_radius: float = 2.0,
        num_basis: int = 8,
        lmax: int = 2,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.cache_path = cache_path
        self.split = split
        self.num_points = num_points
        self.num_neighbors = num_neighbors
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.lmax = lmax

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"ModelNet40 cache not found: {cache_path}")

        self._data = self._load_cache(cache_path)
        self.points = self._data[split]["points"]
        self.inertia = self._data[split]["inertia"]
        self.labels = self._data[split]["labels"]
        self.stats = self._data["stats"]

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
            self.inertia[idx],
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
    """Create train/val/test PyG data loaders for ModelNet40 inertia.

    The original cache only has ``train`` and ``test`` splits. A validation
    set is carved out of the training split using ``val_frac``.
    """
    full_train = ModelNet40InertiaDataset(
        cache_path=cache_path,
        split="train",
        num_points=num_points,
        num_neighbors=num_neighbors,
        max_radius=max_radius,
        num_basis=num_basis,
        lmax=lmax,
    )
    test_dataset = ModelNet40InertiaDataset(
        cache_path=cache_path,
        split="test",
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
