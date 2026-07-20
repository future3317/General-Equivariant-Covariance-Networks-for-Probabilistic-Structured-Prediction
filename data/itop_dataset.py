"""ITOP depth-map dataset for equivariant probabilistic 3D pose prediction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from compatibility.torch_geometric import Data, PyGDataLoader
from data.point_cloud_graph import compute_edge_features, knn_graph
from representations import EquivariantOutputGraph


ITOP_CAMERA_C = 0.0035
ITOP_IMAGE_HEIGHT = 240
ITOP_IMAGE_WIDTH = 320
ITOP_JOINT_NAMES = (
    "Head",
    "Neck",
    "R Shoulder",
    "L Shoulder",
    "R Elbow",
    "L Elbow",
    "R Hand",
    "L Hand",
    "Torso",
    "R Hip",
    "L Hip",
    "R Knee",
    "L Knee",
    "R Foot",
    "L Foot",
)
ITOP_SKELETON_EDGES = (
    (0, 1),
    (1, 8),
    (1, 2),
    (2, 4),
    (4, 6),
    (1, 3),
    (3, 5),
    (5, 7),
    (8, 9),
    (9, 11),
    (11, 13),
    (8, 10),
    (10, 12),
    (12, 14),
)
ITOP_OUTPUT_GRAPH = EquivariantOutputGraph(
    num_nodes=15,
    edges=ITOP_SKELETON_EDGES,
    node_irrep="1o",
    node_names=ITOP_JOINT_NAMES,
)


def resolve_itop_h5(path: str | Path) -> Path:
    """Resolve either a direct HDF5 file or an extractor-created directory."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        nested = path / path.name
        if nested.is_file():
            return nested
    raise FileNotFoundError(path)


@lru_cache(maxsize=1)
def _pixel_grid() -> tuple[np.ndarray, np.ndarray]:
    vertical, horizontal = np.indices(
        (ITOP_IMAGE_HEIGHT, ITOP_IMAGE_WIDTH), dtype=np.float32
    )
    return horizontal, vertical


def depth_to_point_cloud(
    depth: np.ndarray,
    calibration: float = ITOP_CAMERA_C,
) -> np.ndarray:
    """Invert ITOP's documented camera projection into real-world XYZ."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (ITOP_IMAGE_HEIGHT, ITOP_IMAGE_WIDTH):
        raise ValueError(
            f"ITOP depth map must be {(ITOP_IMAGE_HEIGHT, ITOP_IMAGE_WIDTH)}, "
            f"got {depth.shape}"
        )
    horizontal, vertical = _pixel_grid()
    valid = np.isfinite(depth) & (depth > 0.0)
    z = depth[valid]
    x = (horizontal[valid] - 160.0) * calibration * z
    y = -(vertical[valid] - 120.0) * calibration * z
    return np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)


def compact_itop_labels(input_path: str | Path, output_path: str | Path) -> Path:
    """Extract the small pose/visibility fields and omit segmentation masks."""
    input_path = resolve_itop_h5(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as source:
        payload = {
            key: np.asarray(source[key])
            for key in (
                "id",
                "is_valid",
                "visible_joints",
                "image_coordinates",
                "real_world_coordinates",
            )
        }
    # h5py attaches string-encoding metadata that NumPy cannot preserve in
    # NPZ. Recreate the byte-string dtype without metadata.
    ids = payload["id"]
    if ids.dtype.kind in {"S", "U"}:
        payload["id"] = np.array(ids.tolist(), dtype=ids.dtype.str)
    np.savez_compressed(output_path, **payload)
    return output_path


def _load_label_fields(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            return {key: np.asarray(source[key]) for key in source.files}
    with h5py.File(path, "r") as source:
        return {
            key: np.asarray(source[key])
            for key in (
                "id",
                "is_valid",
                "visible_joints",
                "image_coordinates",
                "real_world_coordinates",
            )
        }


class ITOPDepthDataset(Dataset):
    """Generate centered point clouds directly from ITOP depth maps."""

    def __init__(
        self,
        depth_path: str | Path,
        labels_path: str | Path,
        *,
        view: str,
        num_points: int = 1024,
        num_neighbors: int = 16,
        max_radius: float = 0.5,
        num_basis: int = 8,
        lmax: int = 2,
        training: bool = False,
        depth_noise_std: float = 0.0,
        point_dropout: float = 0.0,
        occlusion_fraction: float = 0.0,
    ):
        if view not in {"side", "top"}:
            raise ValueError("view must be 'side' or 'top'")
        if num_points <= num_neighbors:
            raise ValueError("num_points must be greater than num_neighbors")
        if not 0.0 <= point_dropout < 1.0:
            raise ValueError("point_dropout must be in [0, 1)")
        if not 0.0 <= occlusion_fraction < 1.0:
            raise ValueError("occlusion_fraction must be in [0, 1)")

        self.depth_path = resolve_itop_h5(depth_path)
        self.labels_path = Path(labels_path)
        self.view = view
        self.num_points = num_points
        self.num_neighbors = num_neighbors
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.lmax = lmax
        self.training = training
        self.depth_noise_std = depth_noise_std
        self.point_dropout = point_dropout
        self.occlusion_fraction = occlusion_fraction
        self._depth_file: h5py.File | None = None

        labels = _load_label_fields(self.labels_path)
        self.ids = labels["id"]
        self.is_valid = labels["is_valid"].astype(bool)
        self.visible_joints = labels["visible_joints"].astype(bool)
        self.image_coordinates = labels["image_coordinates"]
        self.real_world_coordinates = labels["real_world_coordinates"].astype(
            np.float32
        )
        self.indices = np.flatnonzero(self.is_valid)

        with h5py.File(self.depth_path, "r") as depth_file:
            depth_count = len(depth_file["data"])
            depth_ids = np.asarray(depth_file["id"])
        if depth_count != len(self.is_valid):
            raise ValueError(
                f"depth/label length mismatch: {depth_count} vs {len(self.is_valid)}"
            )
        if not np.array_equal(depth_ids, self.ids):
            raise ValueError("depth and label frame IDs are not aligned")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_depth_file"] = None
        return state

    def _depth_data(self):
        if self._depth_file is None:
            self._depth_file = h5py.File(self.depth_path, "r")
        return self._depth_file["data"]

    def __len__(self) -> int:
        return len(self.indices)

    def _degrade_points(self, points: np.ndarray) -> np.ndarray:
        if not self.training:
            return points
        if self.point_dropout > 0.0:
            keep = np.random.random(len(points)) >= self.point_dropout
            points = points[keep]
        if self.occlusion_fraction > 0.0 and len(points) > self.num_points:
            center = points[np.random.randint(len(points))]
            distance = np.linalg.norm(points - center, axis=-1)
            remove_count = int(len(points) * self.occlusion_fraction)
            keep = np.ones(len(points), dtype=bool)
            keep[np.argpartition(distance, remove_count)[:remove_count]] = False
            points = points[keep]
        return points

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            raise ValueError("depth map contains no valid points")
        replace = len(points) < self.num_points
        if self.training:
            selected = np.random.choice(len(points), self.num_points, replace=replace)
        else:
            selected = np.linspace(0, len(points) - 1, self.num_points).astype(np.int64)
        return points[selected]

    def __getitem__(self, item: int) -> Data:
        frame_index = int(self.indices[item])
        depth = np.asarray(self._depth_data()[frame_index], dtype=np.float32)
        if self.training and self.depth_noise_std > 0.0:
            valid = depth > 0.0
            noise = np.random.normal(0.0, self.depth_noise_std, depth.shape)
            depth = depth.copy()
            depth[valid] = np.maximum(depth[valid] + noise[valid], 1e-5)

        observed = self._degrade_points(depth_to_point_cloud(depth))
        centroid = observed.mean(axis=0, dtype=np.float64).astype(np.float32)
        points = self._sample_points(observed - centroid)
        joints = self.real_world_coordinates[frame_index] - centroid

        pos = torch.from_numpy(points).float()
        edge_index = knn_graph(pos, self.num_neighbors)
        edge_features = compute_edge_features(
            pos,
            edge_index,
            self.max_radius,
            self.num_basis,
            self.lmax,
        )
        return Data(
            pos=pos,
            z=torch.zeros(self.num_points, dtype=torch.long),
            edge_index=edge_index,
            edge_sh=edge_features["edge_sh"],
            edge_rbf=edge_features["edge_rbf"],
            edge_weights=edge_features["edge_weights"],
            y_pose=torch.from_numpy(joints.reshape(1, -1)).float(),
            visible_joints=torch.from_numpy(
                self.visible_joints[frame_index].reshape(1, -1)
            ),
            centroid=torch.from_numpy(centroid.reshape(1, 3)).float(),
            frame_index=torch.tensor([frame_index], dtype=torch.long),
            view_id=torch.tensor([0 if self.view == "side" else 1], dtype=torch.long),
        )


def _itop_paths(data_dir: Path, view: str, split: str) -> tuple[Path, Path]:
    depth = resolve_itop_h5(data_dir / f"ITOP_{view}_{split}_depth_map.h5")
    compact = data_dir / f"ITOP_{view}_{split}_labels_compact.npz"
    labels = (
        compact
        if compact.exists()
        else resolve_itop_h5(data_dir / f"ITOP_{view}_{split}_labels.h5")
    )
    return depth, labels


def get_itop_loaders(
    data_dir: str | Path,
    *,
    train_view: str = "side",
    test_view: str | None = None,
    batch_size: int = 8,
    num_points: int = 1024,
    num_neighbors: int = 16,
    max_radius: float = 0.5,
    num_basis: int = 8,
    lmax: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    depth_noise_std: float = 0.0,
    point_dropout: float = 0.0,
    occlusion_fraction: float = 0.0,
) -> tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    """Build standard or cross-view ITOP train/validation/test loaders."""
    data_dir = Path(data_dir)
    test_view = test_view or train_view
    train_depth, train_labels = _itop_paths(data_dir, train_view, "train")
    test_depth, test_labels = _itop_paths(data_dir, test_view, "test")
    shared = dict(
        num_points=num_points,
        num_neighbors=num_neighbors,
        max_radius=max_radius,
        num_basis=num_basis,
        lmax=lmax,
    )
    train_full = ITOPDepthDataset(
        train_depth,
        train_labels,
        view=train_view,
        training=True,
        depth_noise_std=depth_noise_std,
        point_dropout=point_dropout,
        occlusion_fraction=occlusion_fraction,
        **shared,
    )
    validation_full = ITOPDepthDataset(
        train_depth, train_labels, view=train_view, training=False, **shared
    )
    test_dataset = ITOPDepthDataset(
        test_depth, test_labels, view=test_view, training=False, **shared
    )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_full), generator=generator).tolist()
    validation_count = max(1, int(len(permutation) * val_fraction))
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]
    train_dataset = Subset(train_full, train_indices)
    validation_dataset = Subset(validation_full, validation_indices)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return (
        PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs),
        PyGDataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
        ),
        PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs),
    )
