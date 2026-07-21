"""Precompute deterministic ITOP point clouds and exact k-NN neighborhoods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.itop_dataset import (
    ITOPDepthDataset,
    itop_cache_dir,
    itop_paths,
)
from data.paths import dataset_dir
from data.point_cloud_graph import knn_graph


def write_itop_geometry_cache(
    data_dir: str | Path,
    *,
    view: str,
    split: str,
    num_points: int,
    num_neighbors: int,
) -> Path:
    """Write one immutable, parameter-bound cache without touching raw data."""
    root = dataset_dir(data_dir, "ITOP").resolve()
    depth, labels = itop_paths(root, view, split)
    destination = itop_cache_dir(root, view, split, num_points, num_neighbors)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite ITOP cache: {destination}")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(f"stale partial cache must be inspected: {staging}")
    staging.mkdir(parents=True)

    source = ITOPDepthDataset(
        depth,
        labels,
        view=view,
        num_points=num_points,
        num_neighbors=num_neighbors,
        training=False,
    )
    count = len(source)
    arrays = {
        "points": np.lib.format.open_memmap(
            staging / "points.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, num_points, 3),
        ),
        "neighbors": np.lib.format.open_memmap(
            staging / "neighbors.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(count, num_points, num_neighbors),
        ),
        "joints": np.lib.format.open_memmap(
            staging / "joints.npy", mode="w+", dtype=np.float32, shape=(count, 15, 3)
        ),
        "visible_joints": np.lib.format.open_memmap(
            staging / "visible_joints.npy", mode="w+", dtype=np.bool_, shape=(count, 15)
        ),
        "centroids": np.lib.format.open_memmap(
            staging / "centroids.npy", mode="w+", dtype=np.float32, shape=(count, 3)
        ),
        "frame_indices": np.lib.format.open_memmap(
            staging / "frame_indices.npy", mode="w+", dtype=np.int64, shape=(count,)
        ),
    }
    for index in tqdm(range(count), desc=f"cache ITOP {view} {split}"):
        record = source.sample_record(index)
        points = np.asarray(record["points"], dtype=np.float32)
        edge_index = knn_graph(torch.from_numpy(points), num_neighbors)
        arrays["points"][index] = points
        arrays["neighbors"][index] = (
            edge_index[1].reshape(num_points, num_neighbors).numpy().astype(np.uint16)
        )
        arrays["joints"][index] = np.asarray(
            record["joints"], dtype=np.float32
        ).reshape(15, 3)
        arrays["visible_joints"][index] = np.asarray(
            record["visible_joints"], dtype=np.bool_
        )
        arrays["centroids"][index] = np.asarray(record["centroid"], dtype=np.float32)
        arrays["frame_indices"][index] = int(record["frame_index"])
    for array in arrays.values():
        array.flush()
        array._mmap.close()
    arrays.clear()

    metadata = {
        "schema_version": 1,
        "view": view,
        "split": split,
        "num_samples": count,
        "num_points": num_points,
        "num_neighbors": num_neighbors,
        "centering": "observable_point_cloud_centroid",
        "sampling": "deterministic_linspace_over_valid_depth_points",
        "neighbors": "exact_scipy_ckdtree_knn_excluding_self",
        "source": {
            "depth_file": depth.name,
            "depth_bytes": depth.stat().st_size,
            "labels_file": labels.name,
            "labels_bytes": labels.stat().st_size,
        },
    }
    (staging / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    staging.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--view", choices=("side", "top", "all"), default="side")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--num_points", type=int, choices=(256, 512), default=512)
    parser.add_argument("--num_neighbors", type=int, default=16)
    args = parser.parse_args()
    views = ("side", "top") if args.view == "all" else (args.view,)
    splits = ("train", "test") if args.split == "all" else (args.split,)
    for view in views:
        for split in splits:
            output = write_itop_geometry_cache(
                args.data_dir,
                view=view,
                split=split,
                num_points=args.num_points,
                num_neighbors=args.num_neighbors,
            )
            print(output)


if __name__ == "__main__":
    main()
