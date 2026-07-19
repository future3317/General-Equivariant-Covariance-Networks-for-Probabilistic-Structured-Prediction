"""
preprocess_edges_full.py
-------------------------
Complete preprocessing pipeline for graph features (edge_index, edge_sh, edge_rbf, etc.).
This will make training much faster by eliminating ALL redundant computations.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import pandas as pd

# Import required libraries
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import chemical_symbols
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Import local modules
from voigt_utils import tensor_to_kelvin_mandel_log, voigt_to_tensor
from atom_features import create_composite_atom_features


# ==============================================================================
# Original Data Loader for Preprocessing
# ==============================================================================
class DielectricDatasetRaw(Dataset):
    """
    Raw dataset loader for preprocessing.
    Loads original .pkl files and provides access to structure data.
    Used ONLY by preprocess_edges_full.py to generate precomputed graphs.
    """

    def __init__(self, data_dir, split, max_radius=5.0, lmax=4, num_basis=8):
        """
        Load raw data from .pkl file.

        Args:
            data_dir: Base data directory (e.g., 'data/mp_dielectric')
            split: Dataset split ('train', 'val', 'test')
            max_radius: Cutoff radius for neighbor list
            lmax: Maximum degree of spherical harmonics
            num_basis: Number of radial basis functions
        """
        import pickle

        pkl_path = os.path.join(data_dir, f"{split}.pkl")
        print(f"[DielectricDatasetRaw] Loading {pkl_path}...")

        with open(pkl_path, 'rb') as f:
            self.df = pickle.load(f)

        print(f"[DielectricDatasetRaw] Loaded {len(self.df)} samples")

        # Parse all structures once
        self.parsed_data = []
        self.filtered_indices = []

        # Counters for filtering statistics
        filter_stats = {
            'too_small': 0,      # num_atoms <= 2
            'too_large': 0,      # num_atoms > 30
            'non_spd': 0,        # eigenvalues <= 1e-4
            'out_of_range': 0,   # dielectric < -10 or > 50
            'diag_too_small': 0, # diag < 1.0
            'eigen_error': 0,    # eigenvalue decomposition error
            'parse_error': 0,    # parsing errors
        }

        for idx in range(len(self.df)):
            try:
                row = self.df.iloc[idx]
                structure = row['structure']
                epsilon_vec = row['epsilon_vec']

                # Extract structure data (pymatgen dict format)
                if isinstance(structure, dict):
                    cell = np.array(structure['lattice']['matrix'])
                    positions = np.array([site['xyz'] for site in structure['sites']])
                    atomic_numbers = np.array([self._element_to_Z(site['species'][0]['element'])
                                              for site in structure['sites']])
                elif hasattr(structure, 'cell'):
                    # ASE Atoms format (fallback)
                    if hasattr(structure.cell, 'array'):
                        cell = structure.cell.array
                    else:
                        cell = structure.cell
                    positions = structure.positions
                    atomic_numbers = structure.numbers
                else:
                    raise ValueError(f"Unknown structure format: {type(structure)}")

                # Handle epsilon_vec - convert from Voigt format (6 elements) to 3x3 tensor
                eps_array = np.array(epsilon_vec)
                if eps_array.shape == (6,):
                    # Voigt format [C11, C22, C33, C23, C13, C12]
                    dielectric = voigt_to_tensor(torch.from_numpy(eps_array).float()).numpy()
                elif eps_array.shape == (3, 3):
                    dielectric = eps_array
                else:
                    raise ValueError(f"Unknown epsilon_vec shape: {eps_array.shape}")

                num_atoms = len(atomic_numbers)

                # ==================== FILTERING ====================
                # 1. Filter by number of atoms
                if num_atoms <= 2:
                    filter_stats['too_small'] += 1
                    continue
                if num_atoms > 30:
                    filter_stats['too_large'] += 1
                    continue

                # 2. Check positive definiteness (SPD)
                try:
                    evals = np.linalg.eigvalsh(dielectric)
                    if np.any(evals <= 1e-4):
                        filter_stats['non_spd'] += 1
                        continue
                except np.linalg.LinAlgError:
                    filter_stats['eigen_error'] += 1
                    continue

                # 3. Filter by dielectric value range
                if np.any(dielectric < -10) or np.any(dielectric > 50):
                    filter_stats['out_of_range'] += 1
                    continue

                # 4. Check diagonal values
                diag = np.diag(dielectric)
                if np.any(diag < 1.0):
                    filter_stats['diag_too_small'] += 1
                    continue
                # ==================== FILTERING END ====================

                sample_data = {
                    'positions': positions,
                    'atomic_numbers': atomic_numbers,
                    'dielectric': dielectric,
                    'cell': cell,
                }

                self.parsed_data.append(sample_data)
                self.filtered_indices.append(len(self.parsed_data) - 1)

            except Exception as e:
                filter_stats['parse_error'] += 1
                print(f"[Warning] Failed to parse sample {idx}: {e}")
                continue

        # Print filtering statistics
        print(f"[DielectricDatasetRaw] Filtering statistics:")
        print(f"  Total samples: {len(self.df)}")
        print(f"  Passed: {len(self.parsed_data)}")
        for key, val in filter_stats.items():
            if val > 0:
                print(f"  Filtered ({key}): {val}")

        # Compute normalization parameters - COMPONENT-WISE!
        # Collect all 6 components separately for proper normalization
        all_eps = []
        for data in self.parsed_data:
            eps = data['dielectric']
            # Convert to Kelvin-Mandel and compute log
            eps_tensor = torch.from_numpy(eps).float()
            km_tensor = tensor_to_kelvin_mandel_log(eps_tensor.unsqueeze(0)).squeeze(0)
            all_eps.append(km_tensor.numpy())  # All 6 components!

        all_eps = np.array(all_eps)  # [N, 6]
        self.component_mean = all_eps.mean(axis=0).tolist()  # [6]
        self.component_std = all_eps.std(axis=0).tolist()    # [6]

        # For backward compatibility, keep log_mean/log_std (use diagonal components)
        self.log_mean = float(self.component_mean[0])
        self.log_std = float(self.component_std[0])

        print(f"[DielectricDatasetRaw] Component-wise normalization:")
        for i, name in enumerate(['ε11', 'ε22', 'ε33', 'ε23', 'ε13', 'ε12']):
            print(f"  {name}: mean={self.component_mean[i]:.4f}, std={self.component_std[i]:.4f}")
        print(f"[DielectricDatasetRaw] Successfully parsed {len(self.parsed_data)} structures")

    def __len__(self):
        return len(self.parsed_data)

    def __getitem__(self, idx):
        """Not used during preprocessing, but required by Dataset."""
        return self.parsed_data[idx]

    @staticmethod
    def _element_to_Z(element_symbol):
        """Convert element symbol to atomic number."""
        try:
            return chemical_symbols.index(element_symbol)
        except ValueError:
            print(f"[Warning] Unknown element: {element_symbol}")
            return 1  # Default to hydrogen


def build_graph_from_structure(
    positions,
    atomic_numbers,
    cell,
    dielectric_tensor,
    target_km,  # Precomputed target in Kelvin-Mandel space
    max_radius,
    lmax,
    num_basis,
    log_mean,
    log_std,
    irreps_sh,  # Precomputed spherical harmonics irreps
):
    """
    Build a PyG Data object from structure with ALL edge features precomputed.

    Args:
        positions: Atomic positions (numpy array)
        atomic_numbers: Atomic numbers (numpy array)
        cell: Unit cell matrix (3x3 numpy array)
        dielectric_tensor: 3x3 dielectric tensor (numpy array)
        target_km: Target in Kelvin-Mandel space (already normalized)
        max_radius: Maximum radius for neighbor search
        lmax: Maximum degree of spherical harmonics
        num_basis: Number of radial basis functions
        log_mean: Log mean for normalization
        log_std: Log std for normalization

    Returns:
        PyG Data object with all features precomputed
    """
    # Create ASE atoms object
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=True
    )

    # Compute neighbor list
    i_idx, j_idx, S = neighbor_list(
        'ijS', atoms, cutoff=max_radius, self_interaction=False
    )

    # Initialize empty tensors for no-edge case
    if len(i_idx) == 0:
        node_features = create_composite_atom_features(
            torch.tensor(atomic_numbers), use_onehot=True
        )

        data = Data(
            pos=torch.tensor(positions, dtype=torch.float32),
            z=torch.tensor(atomic_numbers, dtype=torch.long),
            y=target_km,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_sh=torch.empty((0, (lmax + 1) ** 2), dtype=torch.float32),
            edge_rbf=torch.empty((0, num_basis), dtype=torch.float32),
            edge_weights=torch.empty(0, dtype=torch.float32),
            edge_shift=torch.empty((0, 3), dtype=torch.float32),
            node_features=node_features,
            # Store original dielectric for debugging
            dielectric=torch.tensor(dielectric_tensor, dtype=torch.float32),
            # Store metadata
            num_nodes=len(atomic_numbers),
            num_edges=0,
        )
        return data

    # Convert to torch tensors
    pos = torch.tensor(positions, dtype=torch.float32)
    cell_t = torch.tensor(cell, dtype=torch.float32)
    S_t = torch.tensor(S, dtype=torch.float32)

    # Compute edge vectors and lengths
    shifted_pos = pos[j_idx] + torch.matmul(S_t, cell_t)
    edge_vec = shifted_pos - pos[i_idx]
    edge_lengths = torch.norm(edge_vec, dim=1)

    # Compute edge weights
    edge_weights = torch.where(
        edge_lengths < max_radius,
        0.5 * (1 + torch.cos(np.pi * edge_lengths / max_radius)),
        torch.zeros_like(edge_lengths)
    )

    # Filter edges by weight
    mask = edge_weights > 1e-5
    if mask.any():
        # Convert torch mask to numpy for numpy arrays
        mask_np = mask.numpy()
        i_idx = i_idx[mask_np]
        j_idx = j_idx[mask_np]
        # Keep using torch mask for torch tensors
        edge_vec = edge_vec[mask]
        edge_lengths = edge_lengths[mask]
        edge_weights = edge_weights[mask]
        S_t = S_t[mask]

    # Compute spherical harmonics (using precomputed irreps_sh)
    edge_vec_norm = edge_vec / edge_lengths.unsqueeze(1)
    edge_sh = o3.spherical_harmonics(
        irreps_sh,
        edge_vec_norm,
        True,
        normalization="component"
    )

    # Compute radial basis functions
    edge_rbf = soft_one_hot_linspace(
        edge_lengths,
        0.0,
        max_radius,
        number=num_basis,
        basis='cosine',
        cutoff=True
    ).mul(num_basis ** 0.5)

    # Create edge_index tensor
    edge_index = torch.stack([
        torch.tensor(i_idx, dtype=torch.long),
        torch.tensor(j_idx, dtype=torch.long)
    ], dim=0)

    # Create node features (49-dimensional composite features)
    node_features = create_composite_atom_features(
        torch.tensor(atomic_numbers), use_onehot=True
    )

    # Build PyG Data object
    data = Data(
        pos=pos,
        z=torch.tensor(atomic_numbers, dtype=torch.long),
        y=target_km,
        edge_index=edge_index,
        edge_sh=edge_sh,
        edge_rbf=edge_rbf,
        edge_weights=edge_weights,
        edge_shift=S_t,
        node_features=node_features,
        # Store original dielectric for debugging (optional)
        dielectric=torch.tensor(dielectric_tensor, dtype=torch.float32),
        # Store metadata
        num_nodes=len(atomic_numbers),
        num_edges=len(edge_index[0]),
    )

    return data


def preprocess_dataset(
    dataset,
    out_dir,
    max_radius,
    lmax,
    num_basis,
):
    """
    Preprocess ALL samples in a dataset and save to disk.

    Args:
        dataset: DielectricDatasetOptimized instance
        out_dir: Output directory for precomputed graphs
        max_radius: Maximum radius for neighbor search
        lmax: Maximum degree of spherical harmonics
        num_basis: Number of radial basis functions
    """
    # Precompute spherical harmonics irreps (do this once, not per sample!)
    irreps_sh = o3.Irreps.spherical_harmonics(lmax)
    print(f"[Preprocess] Using precomputed irreps_sh: {irreps_sh}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save dataset metadata
    metadata = {
        'num_samples': len(dataset),
        'max_radius': max_radius,
        'lmax': lmax,
        'num_basis': num_basis,
        'log_mean': dataset.log_mean,
        'log_std': dataset.log_std,
        # [FIX] Component-wise normalization
        'component_mean': dataset.component_mean,  # [6] for each component
        'component_std': dataset.component_std,    # [6] for each component
        # ✅ 记录稳定ID信息
        'stable_ids': {
            'pre_idx': '预计算图文件名索引 (0.pt, 1.pt, ...)',
            'orig_idx': '原始数据集中的真实索引'
        }
    }

    with open(out_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[Preprocess] Processing {len(dataset)} samples...")
    print(f"[Preprocess] Output directory: {out_path}")
    print(f"[Preprocess] Metadata saved to {out_path / 'metadata.json'}")

    # Save index mapping
    index_mapping = {}

    # Process each sample
    for idx in tqdm(range(len(dataset)), desc="Preprocessing graphs"):
        # Get actual data
        actual_idx = dataset.filtered_indices[idx]
        sample_data = dataset.parsed_data[actual_idx]

        # Save mapping from dataset index to original index
        index_mapping[str(idx)] = int(actual_idx)

        # Extract components
        positions = sample_data['positions']
        atomic_numbers = sample_data['atomic_numbers']
        dielectric = sample_data['dielectric']
        cell = sample_data['cell']

        # Precompute target in Kelvin-Mandel space
        dielectric_tensor = torch.from_numpy(dielectric).float()
        target_km = tensor_to_kelvin_mandel_log(dielectric_tensor.unsqueeze(0)).squeeze(0)

        # [FIX] Component-wise normalization instead of scalar normalization
        mean_vec = torch.tensor(dataset.component_mean, dtype=torch.float32)
        std_vec = torch.tensor(dataset.component_std, dtype=torch.float32)
        target_km = (target_km - mean_vec) / std_vec
        target_km = torch.nan_to_num(target_km, nan=0.0, posinf=0.0, neginf=0.0)

        # Build graph with ALL features precomputed
        data = build_graph_from_structure(
            positions=positions,
            atomic_numbers=atomic_numbers,
            cell=cell,
            dielectric_tensor=dielectric,
            target_km=target_km,
            max_radius=max_radius,
            lmax=lmax,
            num_basis=num_basis,
            log_mean=dataset.log_mean,
            log_std=dataset.log_std,
            irreps_sh=irreps_sh,  # Pass precomputed irreps_sh
        )

        # ✅ Add stable IDs to the Data object
        # pre_idx: 预计算图文件名对应的 idx (0, 1, 2, ...)
        # orig_idx: 原始/filtered之前的真实 idx
        data.pre_idx = int(idx)          # 对应文件名 {idx}.pt
        data.orig_idx = int(actual_idx)  # 对应原始数据集中的索引

        # Save to disk
        torch.save(data, out_path / f"{idx}.pt")

        # Save index mapping periodically
        if idx % 100 == 0 or idx == len(dataset) - 1:
            with open(out_path / 'index_mapping.json', 'w') as f:
                json.dump(index_mapping, f, indent=2)

    print(f"[Preprocess] Complete! Saved {len(dataset)} graphs to {out_path}")
    print(f"[Preprocess] Each graph file contains ALL edge features precomputed")
    print(f"[Preprocess] Features include: edge_index, edge_sh, edge_rbf, edge_weights, edge_shift")
    return str(out_path)


def main():
    """
    Main function to run preprocessing.
    Example usage:
    python preprocess_edges_full.py
    """
    # Configuration (match your training config)
    config = {
        'data_dir': 'data/mp_dielectric',
        'max_radius': 5.0,
        'lmax': 4,
        'num_basis': 8,
    }

    # ==================== Step 1: Process training set ====================
    print(f"\n{'='*60}")
    print(f"PREPROCESSING TRAIN SET (computing normalization statistics)")
    print(f"{'='*60}")

    train_dataset = DielectricDatasetRaw(
        data_dir=config['data_dir'],
        split='train',
        max_radius=config['max_radius'],
        lmax=config['lmax'],
        num_basis=config['num_basis'],
    )

    # Save training set normalization parameters
    train_log_mean = train_dataset.log_mean
    train_log_std = train_dataset.log_std
    train_component_mean = train_dataset.component_mean  # [FIX] Component-wise stats
    train_component_std = train_dataset.component_std

    print(f"\n[Normalization] Training set statistics (will be used for all splits):")
    print(f"  log_mean = {train_log_mean:.4f}")
    print(f"  log_std = {train_log_std:.4f}")
    print(f"  component_mean = {train_component_mean}")
    print(f"  component_std = {train_component_std}")

    # Preprocess training set
    out_dir = os.path.join(config['data_dir'], "train_graphs_full")
    preprocess_dataset(
        dataset=train_dataset,
        out_dir=out_dir,
        max_radius=config['max_radius'],
        lmax=config['lmax'],
        num_basis=config['num_basis'],
    )
    print(f"[Preprocess] Train set complete!")

    # ==================== Step 2: Process val/test with train stats ====================
    for split in ['val', 'test']:
        print(f"\n{'='*60}")
        print(f"PREPROCESSING {split.upper()} SET (using train normalization)")
        print(f"{'='*60}")

        # Load dataset
        dataset = DielectricDatasetRaw(
            data_dir=config['data_dir'],
            split=split,
            max_radius=config['max_radius'],
            lmax=config['lmax'],
            num_basis=config['num_basis'],
        )

        # OVERRIDE with training set normalization (critical!)
        dataset.log_mean = train_log_mean
        dataset.log_std = train_log_std
        dataset.component_mean = train_component_mean  # [FIX] Use train component-wise stats
        dataset.component_std = train_component_std

        print(f"[Normalization] Using training set statistics for {split} set")

        # Preprocess
        out_dir = os.path.join(config['data_dir'], f"{split}_graphs_full")
        preprocess_dataset(
            dataset=dataset,
            out_dir=out_dir,
            max_radius=config['max_radius'],
            lmax=config['lmax'],
            num_basis=config['num_basis'],
        )

        print(f"[Preprocess] {split} set complete!")

    print(f"\n{'='*60}")
    print("ALL SETS PREPROCESSED!")
    print("Now you can use DielectricPrecomputedDataset for lightning-fast training")
    print("No more neighbor list computation!")
    print("No more spherical harmonics computation!")
    print("No more radial basis function computation!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()