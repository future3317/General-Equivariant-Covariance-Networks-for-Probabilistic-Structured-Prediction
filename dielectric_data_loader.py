"""
dielectric_data_loader.py
------------------------
Data loader for PRECOMPUTED graphs only.
This is the ONLY way to load data for training - no on-the-fly computation.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from compatibility.torch_geometric import Data, PyGDataLoader


class DielectricDataset(Dataset):
    """
    Ultra-fast dataset that loads precomputed graphs from disk.
    All heavy computations (neighbor lists, spherical harmonics, RBF) are done offline.

    This is the ONLY dataset class for training - no on-the-fly computation allowed.
    """

    def __init__(self, base_dir, split):
        """
        Initialize with path to precomputed graphs.

        Args:
            base_dir: Base data directory (e.g., data/mp_dielectric)
            split: Dataset split ('train', 'val', 'test')
        """
        self.graph_dir = os.path.join(base_dir, f"{split}_graphs_full")

        # Verify graph directory exists
        if not os.path.exists(self.graph_dir):
            raise FileNotFoundError(f"Graph directory not found: {self.graph_dir}")

        # Load metadata
        metadata_file = os.path.join(self.graph_dir, 'metadata.json')
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        # List all graph files (they should be named 0.pt, 1.pt, ...)
        self.graph_files = sorted([
            f for f in os.listdir(self.graph_dir) if f.endswith('.pt')
        ], key=lambda x: int(x.split('.')[0]))

        # Store normalization parameters (loaded from metadata)
        self.log_mean = self.metadata['log_mean']
        self.log_std = self.metadata['log_std']
        # [FIX] Component-wise normalization for proper denormalization
        self.component_mean = self.metadata.get('component_mean', [self.log_mean]*3 + [0.0]*3)
        self.component_std = self.metadata.get('component_std', [self.log_std]*6)

        print(f"[DielectricDataset] Loaded {len(self.graph_files)} precomputed graphs")
        print(f"[DielectricDataset] Graph directory: {self.graph_dir}")

    def __len__(self):
        """Return number of samples."""
        return len(self.graph_files)

    def __getitem__(self, idx):
        """
        Load and return a precomputed PyG Data object.

        This is extremely fast - just a torch.load() call!
        """
        # Load the precomputed graph
        graph_path = os.path.join(self.graph_dir, self.graph_files[idx])
        data = torch.load(graph_path, weights_only=False)

        # Ensure the data has all required fields
        assert isinstance(data, Data), f"Loaded object should be PyG Data, got {type(data)}"

        # ✅ Stable IDs are already embedded in the saved Data object
        # from preprocessing step (pre_idx and orig_idx)
        # No need to generate from filename

        # Ensure the IDs are tensors (in case they were saved as integers)
        if hasattr(data, 'pre_idx') and not isinstance(data.pre_idx, torch.Tensor):
            data.pre_idx = torch.tensor(data.pre_idx, dtype=torch.long)
        if hasattr(data, 'orig_idx') and not isinstance(data.orig_idx, torch.Tensor):
            data.orig_idx = torch.tensor(data.orig_idx, dtype=torch.long)

        # Return the complete Data object (everything is already computed)
        return data


def get_dielectric_data_loaders(data_dir="data/mp_dielectric",
                                batch_size=32,
                                num_workers=0,
                                train_subset=None):
    """
    Get data loaders using PRECOMPUTED graphs.
    This is the ONLY data loading function - no on-the-fly computation.

    Args:
        data_dir: Base data directory containing {split}_graphs_full folders
        batch_size: Batch size for training
        num_workers: Number of workers (0 for single-threaded)
        train_subset: Use subset of training data

    Returns:
        train_loader, val_loader, test_loader

    Note:
        persistent_workers is disabled to prevent memory leaks with torch.load in workers.
        pin_memory is enabled only when CUDA is available.
    """
    print(f"\nLoading PRECOMPUTED graphs (batch_size={batch_size})")
    print("=" * 60)
    print("[+] No neighbor list computation!")
    print("[+] No spherical harmonics computation!")
    print("[+] No radial basis function computation!")
    print("=" * 60)

    # Create datasets
    train_dataset = DielectricDataset(data_dir, "train")
    val_dataset = DielectricDataset(data_dir, "val")
    test_dataset = DielectricDataset(data_dir, "test")

    # Apply subset if specified
    if train_subset is not None and train_subset < len(train_dataset):
        import random
        indices = random.sample(range(len(train_dataset)), train_subset)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using subset of {train_subset} samples for training")

    # Process parameters
    use_workers = 0
    # 禁用 persistent_workers 以防止内存泄漏
    # torch.load 在 worker 进程中配合 persistent_workers 会导致内存无法释放
    use_pin = False

    print(f"\nDataLoader Configuration:")
    print(f"  num_workers: {use_workers}")
    print(f"  persistent_workers: False (disabled to prevent memory leaks)")
    print(f"  pin_memory: {use_pin}")

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=use_workers,
        persistent_workers=False,
        pin_memory=use_pin,
        drop_last=True,
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=use_workers,
        persistent_workers=False,
        pin_memory=use_pin,
        drop_last=False,
    )

    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=use_workers,
        persistent_workers=False,
        pin_memory=use_pin,
        drop_last=False,
    )

    # Print dataset sizes and normalization
    print(f"\nDataset sizes (PRECOMPUTED):")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Get normalization from training set
    if hasattr(train_dataset, 'dataset'):
        # Subset case
        norm_dataset = train_dataset.dataset
    else:
        norm_dataset = train_dataset

    print(f"\nNormalization parameters (from training set):")
    print(f"  log_mean = {norm_dataset.log_mean:.4f}")
    print(f"  log_std = {norm_dataset.log_std:.4f}")

    return train_loader, val_loader, test_loader
