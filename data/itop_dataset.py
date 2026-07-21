"""ITOP depth-map dataset for equivariant probabilistic 3D pose prediction."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, Subset

from compatibility.torch_geometric import Data, PyGDataLoader
from data.point_cloud_graph import knn_graph
from data.paths import dataset_dir
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
ITOP_INDEPENDENT_GRAPH = EquivariantOutputGraph(
    num_nodes=15,
    edges=(),
    node_irrep="1o",
    node_names=ITOP_JOINT_NAMES,
)
ITOP_COMPACT_LABEL_FIELDS = (
    "id",
    "is_valid",
    "visible_joints",
    "real_world_coordinates",
)


def itop_train_validation_indices(
    length: int,
    *,
    seed: int,
    val_fraction: float = 0.1,
) -> tuple[list[int], list[int]]:
    """Return the canonical deterministic ITOP train/validation split."""
    if length < 2:
        raise ValueError("ITOP training split must contain at least two samples")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator).tolist()
    validation_count = max(1, int(length * val_fraction))
    return permutation[validation_count:], permutation[:validation_count]


def require_itop_file(path: str | Path) -> Path:
    """Require the canonical direct-file ITOP layout."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required ITOP file is missing: {path}")
    return path


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
    input_path = require_itop_file(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as source:
        payload = {
            key: np.asarray(source[key]) for key in ITOP_COMPACT_LABEL_FIELDS
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
            return {
                key: np.asarray(source[key]) for key in ITOP_COMPACT_LABEL_FIELDS
            }
    with h5py.File(path, "r") as source:
        return {
            key: np.asarray(source[key]) for key in ITOP_COMPACT_LABEL_FIELDS
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
    ):
        if view not in {"side", "top"}:
            raise ValueError("view must be 'side' or 'top'")
        if num_points <= num_neighbors:
            raise ValueError("num_points must be greater than num_neighbors")
        self.depth_path = require_itop_file(depth_path)
        self.labels_path = Path(labels_path)
        self.view = view
        self.num_points = num_points
        self.num_neighbors = num_neighbors
        self._depth_file: h5py.File | None = None

        labels = _load_label_fields(self.labels_path)
        self.ids = labels["id"]
        self.is_valid = labels["is_valid"].astype(bool)
        self.visible_joints = labels["visible_joints"].astype(bool)
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

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            raise ValueError("depth map contains no valid points")
        selected = np.linspace(0, len(points) - 1, self.num_points).astype(np.int64)
        return points[selected]

    def __getitem__(self, item: int) -> Data:
        record = self.sample_record(item)
        pos = torch.from_numpy(record["points"]).float()
        edge_index = knn_graph(pos, self.num_neighbors)
        return Data(
            pos=pos,
            z=torch.zeros(self.num_points, dtype=torch.long),
            edge_index=edge_index,
            y_pose=torch.from_numpy(record["joints"].reshape(1, -1)).float(),
            visible_joints=torch.from_numpy(record["visible_joints"].reshape(1, -1)),
            centroid=torch.from_numpy(record["centroid"].reshape(1, 3)).float(),
            frame_index=torch.tensor([record["frame_index"]], dtype=torch.long),
            view_id=torch.tensor([record["view_id"]], dtype=torch.long),
        )

    def sample_record(self, item: int) -> dict[str, np.ndarray | int]:
        """Return the deterministic geometric record before graph featurization."""
        frame_index = int(self.indices[item])
        depth = np.asarray(self._depth_data()[frame_index], dtype=np.float32)
        observed = depth_to_point_cloud(depth)
        centroid = observed.mean(axis=0, dtype=np.float64).astype(np.float32)
        points = self._sample_points(observed - centroid)
        joints = self.real_world_coordinates[frame_index] - centroid

        return {
            "points": points,
            "joints": joints,
            "visible_joints": self.visible_joints[frame_index],
            "centroid": centroid,
            "frame_index": frame_index,
            "view_id": 0 if self.view == "side" else 1,
        }


def itop_cache_dir(
    data_dir: str | Path,
    view: str,
    split: str,
    num_points: int,
    num_neighbors: int,
) -> Path:
    """Return the parameter-bound deterministic geometry-cache directory."""
    return (
        Path(data_dir)
        / "cache"
        / (f"{view}_{split}_n{num_points}_k{num_neighbors}_centered_v1")
    )


class ITOPCachedDataset(Dataset):
    """Memory-mapped deterministic ITOP point clouds and exact k-NN indices."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        view: str,
        num_points: int,
        num_neighbors: int,
    ):
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "view": view,
            "num_points": num_points,
            "num_neighbors": num_neighbors,
            "centering": "observable_point_cloud_centroid",
            "sampling": "deterministic_linspace_over_valid_depth_points",
        }
        missing = expected.keys() - self.metadata.keys()
        if missing:
            raise ValueError(f"ITOP cache metadata is missing fields: {sorted(missing)}")
        mismatches = {
            key: {"cache": self.metadata[key], "requested": value}
            for key, value in expected.items()
            if self.metadata[key] != value
        }
        if mismatches:
            raise ValueError(f"ITOP cache contract mismatch: {mismatches}")
        self.view = view
        self.num_points = num_points
        self.num_neighbors = num_neighbors
        self.points = np.load(self.cache_dir / "points.npy", mmap_mode="r")
        self.neighbors = np.load(self.cache_dir / "neighbors.npy", mmap_mode="r")
        self.joints = np.load(self.cache_dir / "joints.npy", mmap_mode="r")
        self.visible_joints = np.load(
            self.cache_dir / "visible_joints.npy", mmap_mode="r"
        )
        self.centroids = np.load(self.cache_dir / "centroids.npy", mmap_mode="r")
        self.frame_indices = np.load(
            self.cache_dir / "frame_indices.npy", mmap_mode="r"
        )
        count = int(self.metadata["num_samples"])
        expected_shapes = {
            "points": (count, num_points, 3),
            "neighbors": (count, num_points, num_neighbors),
            "joints": (count, 15, 3),
            "visible_joints": (count, 15),
            "centroids": (count, 3),
            "frame_indices": (count,),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"ITOP cache {name} shape {getattr(self, name).shape} != {shape}"
                )

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, item: int) -> Data:
        # Copy read-only memory maps into writable tensors to keep PyTorch's
        # tensor conversion warning-free and worker-safe.
        pos = torch.from_numpy(np.array(self.points[item], copy=True)).float()
        targets = torch.from_numpy(
            np.array(self.neighbors[item], dtype=np.int64, copy=True).reshape(-1)
        )
        sources = torch.arange(self.num_points).repeat_interleave(self.num_neighbors)
        return Data(
            pos=pos,
            z=torch.zeros(self.num_points, dtype=torch.long),
            edge_index=torch.stack((sources, targets)),
            y_pose=torch.from_numpy(
                np.array(self.joints[item], copy=True).reshape(1, -1)
            ).float(),
            visible_joints=torch.from_numpy(
                np.array(self.visible_joints[item], copy=True).reshape(1, -1)
            ),
            centroid=torch.from_numpy(
                np.array(self.centroids[item], copy=True).reshape(1, 3)
            ).float(),
            frame_index=torch.tensor([int(self.frame_indices[item])], dtype=torch.long),
            view_id=torch.tensor([0 if self.view == "side" else 1], dtype=torch.long),
        )


def itop_paths(data_dir: Path, view: str, split: str) -> tuple[Path, Path]:
    return (
        require_itop_file(data_dir / f"ITOP_{view}_{split}_depth_map.h5"),
        require_itop_file(data_dir / f"ITOP_{view}_{split}_labels_compact.npz"),
    )


def get_itop_loaders(
    data_dir: str | Path | None = None,
    *,
    train_view: str = "side",
    test_view: str | None = None,
    batch_size: int = 8,
    num_points: int = 1024,
    num_neighbors: int = 16,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    """Build loaders from required immutable ITOP geometry caches."""
    data_dir = dataset_dir(data_dir, "ITOP")
    test_view = test_view or train_view
    train_cache = itop_cache_dir(
        data_dir, train_view, "train", num_points, num_neighbors
    )
    test_cache = itop_cache_dir(data_dir, test_view, "test", num_points, num_neighbors)
    train_full = ITOPCachedDataset(
        train_cache,
        view=train_view,
        num_points=num_points,
        num_neighbors=num_neighbors,
    )
    validation_full = ITOPCachedDataset(
        train_cache,
        view=train_view,
        num_points=num_points,
        num_neighbors=num_neighbors,
    )
    test_dataset = ITOPCachedDataset(
        test_cache,
        view=test_view,
        num_points=num_points,
        num_neighbors=num_neighbors,
    )

    train_indices, validation_indices = itop_train_validation_indices(
        len(train_full), seed=seed, val_fraction=val_fraction
    )
    train_dataset = Subset(train_full, train_indices)
    validation_dataset = Subset(validation_full, validation_indices)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_sampler = RandomSampler(
        train_dataset,
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        PyGDataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            generator=torch.Generator().manual_seed(seed + 1),
            **loader_kwargs,
        ),
        PyGDataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
        ),
        PyGDataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
        ),
    )


def get_itop_split_loader(
    data_dir: str | Path | None = None,
    *,
    view: str,
    split: str,
    batch_size: int,
    num_points: int,
    num_neighbors: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> PyGDataLoader:
    """Build one deterministic loader from its required geometry cache."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = dataset_dir(data_dir, "ITOP")
    cache = itop_cache_dir(root, view, split, num_points, num_neighbors)
    dataset = ITOPCachedDataset(
        cache,
        view=view,
        num_points=num_points,
        num_neighbors=num_neighbors,
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
