"""Dielectric dataset adapter that exposes targets in ``0e + 2e`` irrep space."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from dielectric_data_loader import DielectricDataset
from data.tensor_conversions import km_to_irreps


class DielectricIrrepsDataset(Dataset):
    """Wrapper around precomputed dielectric graphs with irrep targets.

    The underlying precomputed graphs store ``data.y`` as a **normalized**
    Kelvin-Mandel log-tensor. This wrapper denormalizes it, converts it to the
    ``0e + 2e`` irrep basis, and stores the normalization parameters so that
    the training script can convert predictions back to physical tensors.

    Args:
        base_dir: Data directory containing ``{split}_graphs_full`` folders.
        split: ``'train'``, ``'val'`` or ``'test'``.
    """

    def __init__(self, base_dir: str, split: str):
        self._base = DielectricDataset(base_dir, split)

        # Normalization parameters from the preprocessed graphs.
        self.component_mean = torch.tensor(self._base.component_mean, dtype=torch.float32)
        self.component_std = torch.tensor(self._base.component_std, dtype=torch.float32)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        data = self._base[idx]

        # y is [6] normalized log-KM; reshape to [1, 6] for conversion helpers.
        y_km_norm = data.y.view(1, -1)

        # Denormalize to physical log-KM.
        y_km = y_km_norm * self.component_std.to(y_km_norm.device) + self.component_mean.to(y_km_norm.device)

        # Convert to irreps (log-tensor in irrep space).
        y_irreps = km_to_irreps(y_km).squeeze(0)

        # Attach the physical log-KM and normalization params for evaluation.
        data.y_irreps = y_irreps
        data.y_km = y_km.squeeze(0)
        return data


def get_dielectric_irreps_loaders(
    data_dir: str = "data/mp_dielectric",
    batch_size: int = 32,
    train_subset: int | None = None,
):
    """Create PyG data loaders with irrep-space dielectric targets."""
    train_dataset = DielectricIrrepsDataset(data_dir, "train")
    val_dataset = DielectricIrrepsDataset(data_dir, "val")
    test_dataset = DielectricIrrepsDataset(data_dir, "test")

    if train_subset is not None and train_subset < len(train_dataset):
        import random
        indices = random.sample(range(len(train_dataset)), train_subset)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    loader_kwargs = {"num_workers": 0, "persistent_workers": False, "pin_memory": False}

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
