"""Pack per-graph dielectric ``.pt`` files into cache-friendly shards."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from data.paths import dataset_dir


def shard_split(
    data_dir: str | Path,
    split: str,
    *,
    shard_size: int = 256,
) -> Path:
    """Create one immutable sharded copy of a precomputed graph split."""
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    data_dir = Path(data_dir)
    source = data_dir / f"{split}_graphs_full"
    destination = data_dir / f"{split}_graphs_shards"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing shard directory: {destination}"
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=data_dir))

    graph_files = sorted(source.glob("*.pt"), key=lambda path: int(path.stem))
    shards = []
    try:
        for shard_index, start in enumerate(range(0, len(graph_files), shard_size)):
            selected = graph_files[start : start + shard_size]
            graphs = [torch.load(path, weights_only=False) for path in selected]
            filename = f"shard_{shard_index:05d}.pt"
            torch.save(graphs, staging / filename)
            shards.append({"file": filename, "start": start, "count": len(graphs)})

        shutil.copy2(source / "metadata.json", staging / "metadata.json")
        manifest = {
            "version": 1,
            "num_samples": len(graph_files),
            "shard_size": shard_size,
            "source": source.name,
            "shards": shards,
        }
        with (staging / "manifest.json").open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
        staging.replace(destination)
    except Exception:
        # The uniquely named staging directory is the only destructive target.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--shard_size", type=int, default=256)
    args = parser.parse_args()
    args.data_dir = dataset_dir(args.data_dir, "mp_dielectric")
    for split in args.splits:
        output = shard_split(args.data_dir, split, shard_size=args.shard_size)
        print(output)


if __name__ == "__main__":
    main()
