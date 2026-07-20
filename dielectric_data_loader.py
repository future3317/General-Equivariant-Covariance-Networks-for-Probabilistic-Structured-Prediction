"""
dielectric_data_loader.py
------------------------
Data loader for PRECOMPUTED graphs only.
This is the ONLY way to load data for training - no on-the-fly computation.
"""

import os
import json
import copy
from bisect import bisect_right
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, Sampler
from compatibility.torch_geometric import Data


class DielectricDataset(Dataset):
    """
    Ultra-fast dataset that loads precomputed graphs from disk.
    All heavy computations (neighbor lists, spherical harmonics, RBF) are done offline.

    This is the ONLY dataset class for training - no on-the-fly computation allowed.
    """

    def __init__(
        self,
        base_dir,
        split,
        *,
        storage="files",
        shard_cache_size=2,
    ):
        """
        Initialize with path to precomputed graphs.

        Args:
            base_dir: Base data directory (for example,
                ``$EQUIVCOMPILER_DATA_ROOT/mp_dielectric``).
            split: Dataset split ('train', 'val', 'test')
        """
        if storage not in {"files", "shards"}:
            raise ValueError("storage must be 'files' or 'shards'")
        if shard_cache_size < 1:
            raise ValueError("shard_cache_size must be positive")
        self.storage = storage
        suffix = "graphs_full" if storage == "files" else "graphs_shards"
        self.graph_dir = os.path.join(base_dir, f"{split}_{suffix}")
        self.shard_cache_size = shard_cache_size
        self._shard_cache = OrderedDict()

        # Verify graph directory exists
        if not os.path.exists(self.graph_dir):
            raise FileNotFoundError(f"Graph directory not found: {self.graph_dir}")

        # Load metadata
        metadata_file = os.path.join(self.graph_dir, "metadata.json")
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        if self.storage == "files":
            # Per-graph files are named 0.pt, 1.pt, ...
            self.graph_files = sorted(
                [f for f in os.listdir(self.graph_dir) if f.endswith(".pt")],
                key=lambda x: int(x.split(".")[0]),
            )
            self._shards = []
            self._shard_starts = []
            self._num_samples = len(self.graph_files)
        else:
            manifest_path = os.path.join(self.graph_dir, "manifest.json")
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Shard manifest not found: {manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            if manifest.get("version") != 1:
                raise ValueError(
                    f"unsupported dielectric shard version: {manifest.get('version')}"
                )
            self._shards = manifest["shards"]
            self._shard_starts = [entry["start"] for entry in self._shards]
            self._num_samples = int(manifest["num_samples"])
            self.graph_files = []

        # Store normalization parameters (loaded from metadata)
        self.log_mean = self.metadata["log_mean"]
        self.log_std = self.metadata["log_std"]
        # [FIX] Component-wise normalization for proper denormalization
        self.component_mean = self.metadata.get(
            "component_mean", [self.log_mean] * 3 + [0.0] * 3
        )
        self.component_std = self.metadata.get("component_std", [self.log_std] * 6)

        print(
            f"[DielectricDataset] Loaded {self._num_samples} precomputed graphs "
            f"from {self.storage} storage"
        )
        print(f"[DielectricDataset] Graph directory: {self.graph_dir}")

    def __len__(self):
        """Return number of samples."""
        return self._num_samples

    @property
    def shard_ranges(self):
        """Contiguous global-index ranges used by shard-aware batching."""
        if self.storage != "shards":
            return None
        return tuple(
            (int(entry["start"]), int(entry["start"] + entry["count"]))
            for entry in self._shards
        )

    def _load_shard(self, shard_index):
        cached = self._shard_cache.pop(shard_index, None)
        if cached is not None:
            self._shard_cache[shard_index] = cached
            return cached
        entry = self._shards[shard_index]
        shard_path = os.path.join(self.graph_dir, entry["file"])
        graphs = torch.load(shard_path, weights_only=False)
        if not isinstance(graphs, list) or len(graphs) != entry["count"]:
            raise ValueError(
                f"invalid shard {shard_path}: expected {entry['count']} graphs"
            )
        self._shard_cache[shard_index] = graphs
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return graphs

    def __getitem__(self, idx):
        """
        Load and return a precomputed PyG Data object.

        Sharded storage reuses the containing shard through a small LRU cache.
        """
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        if self.storage == "files":
            graph_path = os.path.join(self.graph_dir, self.graph_files[idx])
            data = torch.load(graph_path, weights_only=False)
        else:
            shard_index = bisect_right(self._shard_starts, idx) - 1
            entry = self._shards[shard_index]
            data = copy.copy(self._load_shard(shard_index)[idx - entry["start"]])

        if not isinstance(data, Data):
            raise TypeError(f"loaded object must be PyG Data, got {type(data)}")

        # ✅ Stable IDs are already embedded in the saved Data object
        # from preprocessing step (pre_idx and orig_idx)
        # No need to generate from filename

        # Ensure the IDs are tensors (in case they were saved as integers)
        if hasattr(data, "pre_idx") and not isinstance(data.pre_idx, torch.Tensor):
            data.pre_idx = torch.tensor(data.pre_idx, dtype=torch.long)
        if hasattr(data, "orig_idx") and not isinstance(data.orig_idx, torch.Tensor):
            data.orig_idx = torch.tensor(data.orig_idx, dtype=torch.long)

        # Return the complete Data object (everything is already computed)
        return data


class ShardBatchSampler(Sampler):
    """Shuffle shard order and samples while keeping I/O locally sequential.

    Batches are emitted from one shard at a time except for a carried partial
    batch at a shard boundary. Every sample appears exactly once per epoch.
    """

    def __init__(self, shard_ranges, batch_size, *, shuffle=True, drop_last=False):
        self.shard_ranges = tuple(shard_ranges)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.num_samples = sum(end - start for start, end in self.shard_ranges)

    def __iter__(self):
        shard_order = torch.arange(len(self.shard_ranges))
        if self.shuffle:
            shard_order = shard_order[torch.randperm(len(shard_order))]
        pending = []
        for shard_index in shard_order.tolist():
            start, end = self.shard_ranges[shard_index]
            indices = torch.arange(start, end)
            if self.shuffle:
                indices = indices[torch.randperm(indices.numel())]
            pending.extend(indices.tolist())
            while len(pending) >= self.batch_size:
                yield pending[: self.batch_size]
                del pending[: self.batch_size]
        if pending and not self.drop_last:
            yield pending

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size
