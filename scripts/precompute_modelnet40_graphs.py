"""Precompute compact, exact k-NN neighbor caches for ModelNet40."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

from data.modelnet40_inertia_dataset import (
    default_modelnet40_cache_path,
    default_modelnet40_graph_cache_path,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _precompute_split(
    points: np.ndarray,
    *,
    num_points: int,
    num_neighbors: int,
    description: str,
) -> torch.Tensor:
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [N, P, 3], got {points.shape}")
    if points.shape[1] < num_points:
        raise ValueError(
            f"requested {num_points} points but the cache contains {points.shape[1]}"
        )
    if not 1 <= num_neighbors < num_points:
        raise ValueError("num_neighbors must satisfy 1 <= k < num_points")
    if num_points > np.iinfo(np.uint16).max:
        raise ValueError("uint16 graph caches support at most 65535 points")

    neighbors = np.empty((len(points), num_points, num_neighbors), dtype=np.uint16)
    for index, point_cloud in enumerate(tqdm(points, desc=description)):
        coordinates = np.asarray(point_cloud[:num_points])
        _, indices = cKDTree(coordinates).query(
            coordinates, k=num_neighbors + 1, workers=1
        )
        neighbors[index] = np.asarray(indices[:, 1:], dtype=np.uint16)
    return torch.from_numpy(neighbors)


def write_graph_cache(
    source: str | Path,
    destination: str | Path,
    *,
    num_points: int,
    num_neighbors: int,
) -> Path:
    """Write a parameter-bound graph cache without overwriting existing data."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ModelNet40 source cache not found: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite graph cache: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataset = joblib.load(source)
    splits = {
        split: _precompute_split(
            dataset[split]["points"],
            num_points=num_points,
            num_neighbors=num_neighbors,
            description=f"k-NN {split}",
        )
        for split in ("train", "test")
    }
    payload = {
        "format_version": 1,
        "source_name": source.name,
        "source_size": source.stat().st_size,
        "source_sha256": _sha256(source),
        "num_points": num_points,
        "num_neighbors": num_neighbors,
        "splits": splits,
    }

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        torch.load(temporary, map_location="cpu", weights_only=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_neighbors", type=int, default=16)
    args = parser.parse_args()

    source = (
        Path(args.cache_path) if args.cache_path else default_modelnet40_cache_path()
    )
    destination = (
        Path(args.output_path)
        if args.output_path
        else default_modelnet40_graph_cache_path(
            source, args.num_points, args.num_neighbors
        )
    )
    output = write_graph_cache(
        source,
        destination,
        num_points=args.num_points,
        num_neighbors=args.num_neighbors,
    )
    print(output)


if __name__ == "__main__":
    main()
