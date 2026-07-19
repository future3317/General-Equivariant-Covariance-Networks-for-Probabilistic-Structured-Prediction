"""Elasticity tensor dataset with irrep-space targets and low-rank covariance."""

from __future__ import annotations

import pickle
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from pymatgen.core import Structure
from e3nn import o3
from e3nn.math import soft_one_hot_linspace

from atom_features import create_composite_atom_features
from data.tensor_conversions import elasticity_21d_to_irreps


# 21D vector indices as upper-triangular positions in a 6x6 Voigt matrix.
_ELASTICITY_21_INDICES = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
]


def _matrix6x6_to_21d(C_6x6: np.ndarray) -> np.ndarray:
    """Convert symmetric 6x6 Voigt matrix to 21D vector."""
    C_21d = np.zeros(21)
    for idx, (i, j) in enumerate(_ELASTICITY_21_INDICES):
        C_21d[idx] = C_6x6[i, j] if i <= j else C_6x6[j, i]
    return C_21d


class ElasticityIrrepsDataset(Dataset):
    """Elasticity dataset with targets converted to e3nn irreps.

    The 21D elasticity vector is normalized using training-set statistics,
    then converted to the irrep basis of the rank-4 elasticity tensor.

    Args:
        data_path: Path to the pickle file containing structures and C_voigt.
        split: ``'train'``, ``'val'`` or ``'test'``.
        max_radius: Neighbor cutoff.
        num_neighbors: Unused, kept for compatibility.
        train_stats: ``(mean, std)`` tuple for normalization. Required for
            ``val``/``test``; computed automatically for ``train``.
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        max_radius: float = 5.0,
        num_neighbors: int = 20,
        train_stats: tuple[np.ndarray, np.ndarray] | None = None,
        lmax: int | None = None,
    ):
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.lmax = lmax
        self._edge_sh_dim = o3.Irreps.spherical_harmonics(lmax).dim if lmax is not None else None

        with open(data_path, "rb") as f:
            df = pickle.load(f)

        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        self.structures = df["structure"].values
        self.C_matrices = df["C_voigt"].values

        self.elasticity_21d = np.stack(
            [_matrix6x6_to_21d(np.array(C)) for C in self.C_matrices]
        )

        if train_stats is None:
            if split != "train":
                raise ValueError("train_stats must be provided for val/test")
            self.mean_21d = self.elasticity_21d.mean(axis=0)
            self.std_21d = self.elasticity_21d.std(axis=0) + 1e-8
        else:
            self.mean_21d, self.std_21d = train_stats

        # Precompute normalized 21D targets.
        self.target_21d_norm = (self.elasticity_21d - self.mean_21d) / self.std_21d

        print(f"[ElasticityIrrepsDataset] {split}: {len(self)} samples")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure_data = self.structures[idx]
        if isinstance(structure_data, dict):
            structure = Structure.from_dict(structure_data)
        else:
            structure = structure_data

        atomic_numbers = np.array([site.specie.Z for site in structure])
        atom_features = create_composite_atom_features(
            torch.tensor(atomic_numbers, dtype=torch.long)
        )

        target_21d = torch.tensor(self.target_21d_norm[idx], dtype=torch.float32)
        target_irreps = elasticity_21d_to_irreps(target_21d.unsqueeze(0))

        data = self._build_graph(structure, atom_features)
        if self._edge_sh_dim is not None and data.edge_sh.shape[-1] != self._edge_sh_dim:
            data.edge_sh = data.edge_sh[..., : self._edge_sh_dim]

        data.y = target_21d.unsqueeze(0)
        data.y_irreps = target_irreps
        data.y_km = target_21d.unsqueeze(0)
        return data

    def _build_graph(self, structure, atom_features):
        num_atoms = len(structure)
        neighbor_list = structure.get_neighbor_list(self.max_radius)

        if len(neighbor_list) == 0 or neighbor_list[0].shape[0] == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_sh = torch.zeros(1, 25)
            edge_rbf = torch.zeros(1, 8)
            edge_weights = torch.ones(1)
        else:
            center_indices = torch.tensor(neighbor_list[0], dtype=torch.long)
            neighbor_indices = torch.tensor(neighbor_list[1], dtype=torch.long)
            edge_vectors = torch.tensor(neighbor_list[2], dtype=torch.float32)
            edge_distances = torch.tensor(neighbor_list[3], dtype=torch.float32)

            edge_index = torch.stack([center_indices, neighbor_indices])
            edge_sh = o3.spherical_harmonics(
                o3.Irreps.spherical_harmonics(4),
                edge_vectors,
                normalize=True,
                normalization="component",
            )
            edge_rbf = soft_one_hot_linspace(
                edge_distances,
                start=0.0,
                end=self.max_radius,
                number=8,
                basis="smooth_finite",
                cutoff=True,
            )
            edge_weights = torch.exp(-(edge_distances / self.max_radius) ** 2)

        return Data(
            node_features=torch.tensor(atom_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_sh=edge_sh,
            edge_rbf=edge_rbf,
            edge_weights=edge_weights,
            batch=torch.zeros(num_atoms, dtype=torch.long),
        )


def get_elasticity_irreps_loaders(
    data_dir: str = "data/mp_elastic",
    batch_size: int = 16,
    num_workers: int = 0,
    train_subset: int | None = None,
    max_radius: float = 5.0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    lmax: int | None = None,
):
    """Create PyG data loaders for irrep-space elasticity targets."""
    train_path = f"{data_dir}/train.pkl"
    val_path = f"{data_dir}/val.pkl"
    test_path = f"{data_dir}/test.pkl"

    train_dataset = ElasticityIrrepsDataset(
        train_path, "train", max_radius=max_radius, lmax=lmax
    )
    train_stats = (train_dataset.mean_21d, train_dataset.std_21d)

    val_dataset = ElasticityIrrepsDataset(
        val_path, "val", max_radius=max_radius, train_stats=train_stats, lmax=lmax
    )
    test_dataset = ElasticityIrrepsDataset(
        test_path, "test", max_radius=max_radius, train_stats=train_stats, lmax=lmax
    )

    if train_subset is not None and train_subset < len(train_dataset):
        import random
        indices = random.sample(range(len(train_dataset)), train_subset)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

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
