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

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from compatibility.torch_geometric import Data, PyGDataLoader

from data.tensor_conversions import voigt_to_irreps
from data.point_cloud_graph import compute_edge_features, knn_graph
from data.paths import dataset_dir

# Backward-compatible names for existing experiment utilities and tests. The
# implementations live in one shared module and are not duplicated here.
_knn_graph = knn_graph
_compute_edge_features = compute_edge_features


MODELNET40_CACHE_RELATIVE_PATH = Path("cache/modelnet40_inertia_dataset_clean.pkl")


def default_modelnet40_cache_path() -> Path:
    """Return the ModelNet40 cache below the configured dataset root."""
    return dataset_dir(None, "modelnet40") / MODELNET40_CACHE_RELATIVE_PATH


def default_modelnet40_graph_cache_path(
    cache_path: str | Path,
    num_points: int,
    num_neighbors: int,
) -> Path:
    """Return the parameter-bound cache path for fixed k-NN neighbors."""
    source = Path(cache_path).expanduser()
    return source.with_name(f"{source.stem}.knn_n{num_points}_k{num_neighbors}.pt")


def _shape_covariance_voigt(points: np.ndarray) -> np.ndarray:
    """Return the 6D Voigt vector of the shape covariance tensor.

    S = (1/N) sum_k (p_k - mu) (p_k - mu)^T.
    """
    centered = points - points.mean(axis=0)
    S = (centered.T @ centered) / len(points)
    S = 0.5 * (S + S.T)
    return np.array(
        [
            S[0, 0],
            S[1, 1],
            S[2, 2],
            S[1, 2],
            S[0, 2],
            S[0, 1],
        ],
        dtype=np.float32,
    )


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
    target_irreps: torch.Tensor,
    label: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    neighbors: torch.Tensor | None,
    edge_source: torch.Tensor | None,
    num_points: int,
    num_neighbors: int,
    max_radius: float,
    num_basis: int,
    lmax: int,
) -> Data:
    """Convert a single point cloud + symmetric tensor target to a PyG Data object."""
    points = points[:num_points]
    pos = torch.from_numpy(points).float()

    # k-NN graph. Neighbor identities are invariant to orthogonal transforms and
    # can therefore be cached without changing the equivariant computation.
    if neighbors is None:
        edge_index = knn_graph(pos, k=num_neighbors)
    else:
        if edge_source is None:
            raise ValueError("edge_source is required with cached neighbors")
        edge_target = neighbors.reshape(-1).to(dtype=torch.long)
        edge_index = torch.stack((edge_source, edge_target), dim=0)

    edge_features = compute_edge_features(pos, edge_index, max_radius, num_basis, lmax)

    # Constant atomic type for all points -> single learnable embedding vector.
    z = torch.zeros(pos.shape[0], dtype=torch.long)

    data = Data(
        pos=pos,
        z=z,
        edge_index=edge_index,
        edge_sh=edge_features["edge_sh"],
        edge_rbf=edge_features["edge_rbf"],
        edge_weights=edge_features["edge_weights"],
        y_irreps=target_irreps.unsqueeze(0),
        # Keep a leading graph dimension so PyG batches these as [B, 6]
        # instead of flattening them to [B * 6].
        y_voigt_mean=target_mean.unsqueeze(0),
        y_voigt_std=target_std.unsqueeze(0),
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
        cache_path: str | Path | None = None,
        split: str = "train",
        target_type: str = "inertia",
        num_points: int = 1024,
        num_neighbors: int = 16,
        max_radius: float = 2.0,
        num_basis: int = 8,
        lmax: int = 2,
        graph_cache_path: str | Path | None = None,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
        if target_type not in {"inertia", "shape_covariance"}:
            raise ValueError(
                f"target_type must be 'inertia' or 'shape_covariance', got {target_type}"
            )

        cache_path = (
            Path(cache_path).expanduser()
            if cache_path is not None
            else default_modelnet40_cache_path()
        )
        self.cache_path = cache_path
        self.split = split
        self.target_type = target_type
        self.num_points = num_points
        self.num_neighbors = num_neighbors
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.lmax = lmax

        if not cache_path.is_file():
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
                    [
                        _shape_covariance_voigt(p[:num_points])
                        for p in self._data["train"]["points"]
                    ],
                    axis=0,
                )
                self.stats = {
                    "mean": train_targets.mean(axis=0),
                    "std": train_targets.std(axis=0),
                }

        # Guard against zero std.
        self.stats["std"] = np.maximum(self.stats["std"], 1e-8)

        # The target transform is fixed and independent of graph augmentation.
        # CartesianTensor.from_cartesian has substantial per-call overhead for a
        # six-component target, so apply the exact same linear transform once to
        # the whole split rather than once per sample on every epoch.
        targets_voigt_norm, target_mean, target_std = _scalar_normalize_voigt(
            np.asarray(self.targets_voigt), self.stats
        )
        self.targets_irreps = voigt_to_irreps(targets_voigt_norm).contiguous()
        self.target_mean = target_mean
        self.target_std = target_std

        default_graph_cache = default_modelnet40_graph_cache_path(
            cache_path, num_points, num_neighbors
        )
        requested_graph_cache = (
            Path(graph_cache_path).expanduser()
            if graph_cache_path is not None
            else default_graph_cache
        )
        if requested_graph_cache.is_file():
            graph_cache = self._load_graph_cache(requested_graph_cache)
            self.neighbors = self._validate_graph_cache(graph_cache)
            self.graph_cache_path: Path | None = requested_graph_cache
            self.edge_source = torch.arange(
                num_points, dtype=torch.long
            ).repeat_interleave(num_neighbors)
        elif graph_cache_path is not None:
            raise FileNotFoundError(
                f"precomputed graph cache not found: {requested_graph_cache}"
            )
        else:
            self.neighbors = None
            self.graph_cache_path = None
            self.edge_source = None

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_cache(cache_path: str | Path) -> dict[str, Any]:
        try:
            return joblib.load(cache_path)
        except Exception:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_graph_cache(graph_cache_path: str | Path) -> dict[str, Any]:
        return torch.load(graph_cache_path, map_location="cpu", weights_only=True)

    def _validate_graph_cache(self, graph_cache: dict[str, Any]) -> torch.Tensor:
        if graph_cache.get("format_version") != 1:
            raise ValueError("unsupported ModelNet40 graph-cache format")
        if graph_cache.get("num_points") != self.num_points:
            raise ValueError("graph cache num_points does not match the dataset")
        if graph_cache.get("num_neighbors") != self.num_neighbors:
            raise ValueError("graph cache num_neighbors does not match the dataset")
        if graph_cache.get("source_name") != self.cache_path.name:
            raise ValueError("graph cache was built from a different source cache")
        if graph_cache.get("source_size") != self.cache_path.stat().st_size:
            raise ValueError("graph cache source size does not match the dataset cache")
        splits = graph_cache.get("splits")
        if not isinstance(splits, dict) or self.split not in splits:
            raise ValueError(f"graph cache is missing split {self.split!r}")
        neighbors = splits[self.split]
        expected_shape = (len(self.points), self.num_points, self.num_neighbors)
        if not isinstance(neighbors, torch.Tensor):
            raise TypeError("graph-cache neighbors must be a torch.Tensor")
        if neighbors.dtype != torch.uint16:
            raise TypeError("graph-cache neighbors must use torch.uint16")
        if tuple(neighbors.shape) != expected_shape:
            raise ValueError(
                f"graph-cache shape {tuple(neighbors.shape)} != {expected_shape}"
            )
        if int(neighbors.numpy().max()) >= self.num_points:
            raise ValueError("graph-cache neighbor index is out of range")
        return neighbors

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> Data:
        return _point_cloud_to_data(
            self.points[idx],
            self.targets_irreps[idx],
            int(self.labels[idx]),
            self.target_mean,
            self.target_std,
            None if self.neighbors is None else self.neighbors[idx],
            self.edge_source,
            self.num_points,
            self.num_neighbors,
            self.max_radius,
            self.num_basis,
            self.lmax,
        )


def get_modelnet40_inertia_loaders(
    cache_path: str | Path | None = None,
    target_type: str = "inertia",
    batch_size: int = 16,
    num_points: int = 1024,
    num_neighbors: int = 16,
    max_radius: float = 2.0,
    num_basis: int = 8,
    lmax: int = 2,
    graph_cache_path: str | Path | None = None,
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
        graph_cache_path=graph_cache_path,
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
        graph_cache_path=graph_cache_path,
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
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
