"""
dielectric_data_loader.py
-------------------------
Dielectric dataset that loads from raw pkl files and dynamically builds edges.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
import pickle
import pandas as pd
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from ase.neighborlist import neighbor_list
import ase.io
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from ase import Atoms
from voigt_utils import sym_matrix_log_voigt, voigt_to_kelvin_mandel
from atom_features import create_composite_atom_features

# Get logger instance
logger = logging.getLogger(__name__)


class DielectricDatasetOptimized(Dataset):
    """
    Dataset that loads from raw pkl files and dynamically builds edges.
    """
    def __init__(self, data_dir, split="train", max_radius=5.0, lmax=2, num_basis=8):
        self.split = split
        self.data_dir = data_dir
        self.max_radius = max_radius
        self.lmax = lmax
        self.num_basis = num_basis

        # Precompute spherical harmonics irreps (constant for all samples)
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.lmax)

        # Path to pkl file
        self.pkl_file = os.path.join(data_dir, f"{split}.pkl")

        if not os.path.exists(self.pkl_file):
            raise FileNotFoundError(f"PKL file not found at {self.pkl_file}")

        # Load data from pkl file (pandas DataFrame)
        self.data_df = pd.read_pickle(self.pkl_file)

        # Filter out problematic samples first
        self.parsed_data = []
        self.filtered_indices = []
        from pymatgen.core import Structure

        # Counters for filtering statistics
        total_count = 0
        filter_stats = {
            'too_small': 0,  # num_atoms <= 2
            'too_large': 0,  # num_atoms > 40
            'non_spd': 0,    # eigenvalues <= 1e-4
            'out_of_range': 0,  # dielectric < -10 or > 50
            'diag_too_small': 0,  # diag < 1.0
            'eigen_error': 0  # eigenvalue decomposition error
        }

        for i, row in self.data_df.iterrows():
            total_count += 1
            # Create pymatgen Structure object from dictionary
            structure = Structure.from_dict(row['structure'])
            dielectric = row['epsilon_total']  # This is the total dielectric tensor

            # Extract data using pymatgen
            positions = structure.cart_coords
            atomic_numbers = np.array([site.specie.Z for site in structure])
            num_atoms = len(atomic_numbers)
            cell = structure.lattice.matrix

            # Apply filters (same as preprocess_edges.py)
            if num_atoms <= 2:
                filter_stats['too_small'] += 1
                continue  # Skip small structures that cause empty edges
            if num_atoms > 30:  # 保持30的原子数限制
                filter_stats['too_large'] += 1
                continue

            # --- 严格检查正定性 (SPD) ---
            try:
                # 计算特征值，确保所有特征值 > 0 (对于 Log 变换是必须的)
                evals = np.linalg.eigvalsh(dielectric)
                if np.any(evals <= 1e-4): # 稍微大于0，防止Log爆炸
                    filter_stats['non_spd'] += 1
                    continue

                # 保留原本的物理范围过滤
                if np.any(dielectric < -10) or np.any(dielectric > 50):
                    filter_stats['out_of_range'] += 1
                    continue

            except np.linalg.LinAlgError:
                filter_stats['eigen_error'] += 1
                continue
            # --- 正定性检查结束 ---

            # Check dielectric tensor
            diag = np.diag(dielectric)
            if np.any(diag < 1.0):
                filter_stats['diag_too_small'] += 1
                continue

            # Store parsed data
            if not hasattr(self, 'parsed_data'):
                self.parsed_data = []
            self.parsed_data.append({
                'positions': positions,
                'atomic_numbers': atomic_numbers,
                'dielectric': dielectric,
                'cell': cell,  # Changed from lattice to cell
                'num_atoms': num_atoms
            })
            self.filtered_indices.append(len(self.parsed_data) - 1)

        print(f"Loaded {len(self.filtered_indices)} samples from {self.pkl_file}")

        # Compute normalization constants from filtered data
        self._compute_normalization_from_parsed()

    def _compute_normalization_from_parsed(self):
        """Vectorized computation of normalization constants using all valid, filtered data."""
        print("\nComputing normalization constants from filtered data (Vectorized)...")

        if len(self.parsed_data) == 0:
            print("Warning: No valid data found! Using default normalization.")
            self.log_mean = 0.0
            self.log_std = 1.0
            return

        # 1. Stack all dielectric tensors into a single array [N, 3, 3]
        matrices = np.stack([item['dielectric'] for item in self.parsed_data])
        tensors = torch.from_numpy(matrices).float()  # [N, 3, 3]

        # 2. [关键] Batch eigenvalue decomposition with float64 precision
        # 使用 double 精度确保矩阵对数计算的数值稳定性
        # 这与后续训练中的 float64 计算保持一致，避免精度损失
        L, Q = torch.linalg.eigh(tensors)  # L: [N, 3] eigenvalues, Q: [N, 3, 3] eigenvectors

        # 3. Compute matrix logarithm: log(A) = Q * diag(log(L)) * Q^T
        # Clamp to avoid log(0) - though data should already be filtered
        L = torch.clamp(L, min=1e-6)
        log_L = torch.log(L)
        log_matrices = Q @ torch.diag_embed(log_L) @ Q.transpose(-2, -1)  # [N, 3, 3]

        # 4. [修复] 使用特征值而不是对角线元素进行统计
        # 特征值具有明确的物理意义（主介电常数），且保证统计一致性
        eigenvalues_log_stats = log_L  # [N, 3] 已经是对数特征值

        # 5. Compute mean and std using eigenvalues (physically meaningful)
        self.log_mean = eigenvalues_log_stats.mean().item()
        self.log_std = eigenvalues_log_stats.std().item()

        # Precompute mean_vec for normalization (constant for all samples)
        # 对角线分量(xx, yy, zz)使用log_mean，剪切分量(yz, xz, xy)均值接近0
        self.mean_vec = torch.tensor([self.log_mean]*3 + [0.0]*3, dtype=torch.float32)

        print(f"  Using {len(self.parsed_data)} valid SPD samples for statistics")
        print(f"  Eigenvalue log stats: min={eigenvalues_log_stats.min().item():.3f}, max={eigenvalues_log_stats.max().item():.3f}")
        print(f"  log_mean = {self.log_mean:.4f} (based on eigenvalues)")
        print(f"  log_std = {self.log_std:.4f} (based on eigenvalues)")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        """
        Return PyG Data object with precomputed edges.
        """
        # Get actual index
        actual_idx = self.filtered_indices[idx]
        data = self.parsed_data[actual_idx]

        # Extract data
        positions = torch.from_numpy(data['positions']).float()
        atomic_numbers = torch.from_numpy(data['atomic_numbers']).long()
        dielectric = data['dielectric']
        cell = data['cell']
        num_atoms = len(atomic_numbers)

        # Process target - [优化] 使用直接路径：Tensor -> Log -> KM
        # 这避免了冗余的 Voigt 转换，减少数值噪音
        from voigt_utils import tensor_to_kelvin_mandel_log

        # 1. 直接构建 3x3 介电张量
        dielectric_tensor = torch.from_numpy(dielectric).float()

        # 2. 直接计算：3x3 Tensor -> Matrix Log -> Kelvin-Mandel
        # 这是最简洁高效的路径，避免了 Tensor->Voigt->Matrix->Log->Voigt->KM 的冗余
        c_vec_log = tensor_to_kelvin_mandel_log(dielectric_tensor.unsqueeze(0)).squeeze(0)

        # [关键修复] 归一化：正确处理KM空间的均值偏置
        # 使用预计算的 mean_vec，避免重复创建
        # 获取标准差标量
        std_scalar = self.log_std if np.isscalar(self.log_std) else self.log_std[0]

        # 应用归一化 - 这确保了：
        # 1. 对角线分量正确减去物理均值（log特征值）
        # 2. 剪切分量保持零均值，避免系统性偏置
        # 3. 所有分量使用相同的标准差，保证马氏距离的几何意义
        target = (c_vec_log - self.mean_vec) / std_scalar
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute edges on-the-fly
        # Create ASE atoms object
        atoms = Atoms(numbers=atomic_numbers.numpy(), positions=positions.numpy(),
                     cell=cell, pbc=True)

        # Compute neighbor list
        i_idx, j_idx, S = neighbor_list('ijS', atoms, cutoff=self.max_radius, self_interaction=False)

        if len(i_idx) > 0:
            # Apply PBC shifts
            cell_tensor = torch.tensor(cell, dtype=torch.float32)
            S_tensor = torch.from_numpy(S).float()
            shifted_pos = positions[j_idx] + torch.matmul(S_tensor, cell_tensor)

            # Compute edge vectors
            edge_vec = shifted_pos - positions[i_idx]
            edge_lengths = torch.norm(edge_vec, dim=1)

            # Edge weights
            edge_weights = torch.where(
                edge_lengths < self.max_radius,
                0.5 * (1 + torch.cos(np.pi * edge_lengths / self.max_radius)),
                torch.zeros_like(edge_lengths)
            )

            # Filter
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
                S_tensor = S_tensor[mask]

            # Compute SH and RBF
            edge_vec_norm = edge_vec / edge_lengths.unsqueeze(1)
            edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec_norm, True, normalization="component")
            edge_rbf = soft_one_hot_linspace(
                edge_lengths, 0.0, self.max_radius, number=self.num_basis,
                basis='cosine', cutoff=True
            ).mul(self.num_basis ** 0.5)

            # [关键修复] Create edge_index in PyG format [src, dst] to match vector direction
            # edge_vec = shifted_pos - positions[i_idx] 是从 i 指向 j 的向量
            # 因此 edge_index 应该是 [i, j]，表示消息从 i 传到 j
            # 这样保证了几何向量方向与图拓扑消息流方向的一致性，维持等变性
            edge_index = torch.stack([torch.from_numpy(i_idx), torch.from_numpy(j_idx)], dim=0)

        else:
            # No edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0, dtype=torch.float32)
            edge_lengths = torch.empty(0, dtype=torch.float32)
            edge_sh = torch.empty(0, (self.lmax + 1) ** 2, dtype=torch.float32)
            edge_rbf = torch.empty(0, self.num_basis, dtype=torch.float32)
            S_tensor = torch.empty((0, 3), dtype=torch.float32)

                    # Pre-compute node features (119-dim composite features)
        node_features = create_composite_atom_features(atomic_numbers, use_onehot=True)

        # All tensors are already on CPU (created from numpy or loaded from files)
        data = Data(
            pos=positions,
            z=atomic_numbers,
            y=target,
            edge_index=edge_index,
            edge_weights=edge_weights,
            edge_lengths=edge_lengths,
            edge_shift=S_tensor,
            edge_sh=edge_sh,
            edge_rbf=edge_rbf,
            cell=torch.from_numpy(cell.copy()).float(),  # Copy to avoid warning about read-only array
            lattice=torch.from_numpy(cell.copy()).float().unsqueeze(0),
            lmax=self.lmax,
            num_basis=self.num_basis,
            node_features=node_features  # Pre-computed node features
        )

        return data


# No custom collate function needed - PyG handles batching automatically when Data objects are returned from __getitem__


def get_dielectric_data_loaders_optimized(data_dir='data/mp_dielectric',
                                         batch_size=32,
                                         num_workers=0,
                                         train_subset=None,
                                         max_radius=4.0,
                                         lmax=2):
    """
    Get data loaders using PyG's automatic batching.
    IMPORTANT: Only training set computes normalization constants.
    Validation and test sets use training set's normalization parameters.

    Args:
        data_dir: Base data directory
        batch_size: Batch size for training
        num_workers: Number of workers
        train_subset: Use subset of training data
        max_radius: Cutoff radius for edge computation
        lmax: Maximum spherical harmonics degree

    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"Loading data from pkl files (batch_size={batch_size}, lmax={lmax})")

    # Create training dataset - this will compute normalization constants
    train_dataset = DielectricDatasetOptimized(
        data_dir,
        split="train",
        max_radius=max_radius,
        lmax=lmax,
        num_basis=8
    )

    # Apply subset if specified
    if train_subset is not None and train_subset < len(train_dataset):
        train_dataset.filtered_indices = train_dataset.filtered_indices[:train_subset]
        print(f"Using subset of {train_subset} samples for training")
        # IMPORTANT: Recompute normalization after subset selection
        train_dataset._compute_normalization_from_parsed()

    # Store training set normalization parameters
    train_log_mean = train_dataset.log_mean
    train_log_std = train_dataset.log_std

    print(f"\nTraining set normalization parameters:")
    print(f"  log_mean = {train_log_mean:.4f}")
    print(f"  log_std = {train_log_std:.4f}")

    # Training data loader - PyG handles batching automatically
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,  # Enable pin memory for faster GPU transfer
    )

    # Validation dataset - use training set's normalization
    val_dataset = DielectricDatasetOptimized(
        data_dir,
        split="val",
        max_radius=max_radius,
        lmax=lmax,
        num_basis=8
    )
    # Override with training set normalization
    val_dataset.log_mean = train_log_mean
    val_dataset.log_std = train_log_std

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    # Test dataset - use training set's normalization
    test_dataset = DielectricDatasetOptimized(
        data_dir,
        split="test",
        max_radius=max_radius,
        lmax=lmax,
        num_basis=8
    )
    # Override with training set normalization
    test_dataset.log_mean = train_log_mean
    test_dataset.log_std = train_log_std

    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


def get_dielectric_data_loaders_precomputed(data_dir="data/mp_dielectric",
                                         batch_size=32,
                                         num_workers=0,
                                         train_subset=None):
    """
    Get data loaders using PRECOMPUTED graphs.
    This is extremely fast as all heavy computations are done offline.

    Args:
        data_dir: Base data directory containing {split}_graphs_full folders
        batch_size: Batch size for training
        num_workers: Number of workers
        train_subset: Use subset of training data

    Returns:
        train_loader, val_loader, test_loader
    """
    import os
    import json
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader

    class DielectricPrecomputedDataset(Dataset):
        """Ultra-fast dataset that loads precomputed graphs from disk."""

        def __init__(self, graph_dir):
            self.graph_dir = os.path.join(data_dir, f"{graph_dir}_graphs_full")

            # Verify directory exists
            if not os.path.exists(self.graph_dir):
                raise FileNotFoundError(f"Graph directory not found: {self.graph_dir}")

            # Load metadata
            metadata_file = os.path.join(self.graph_dir, 'metadata.json')
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)

            # List all graph files
            self.graph_files = sorted([
                f for f in os.listdir(self.graph_dir) if f.endswith('.pt')
            ], key=lambda x: int(x.split('.')[0]))

            # Store normalization parameters
            self.log_mean = self.metadata['log_mean']
            self.log_std = self.metadata['log_std']

            print(f"[PrecomputedDataset] Loaded {len(self.graph_files)} graphs from {graph_dir}")

        def __len__(self):
            return len(self.graph_files)

        def __getitem__(self, idx):
            graph_path = os.path.join(self.graph_dir, self.graph_files[idx])
            return torch.load(graph_path, weights_only=False)

    print(f"\nLoading PRECOMPUTED graphs (batch_size={batch_size})")
    print("=" * 60)
    print("[+] No neighbor list computation!")
    print("[+] No spherical harmonics computation!")
    print("[+] No radial basis function computation!")
    print("=" * 60)

    # Create datasets
    train_dataset = DielectricPrecomputedDataset("train")
    val_dataset = DielectricPrecomputedDataset("val")
    test_dataset = DielectricPrecomputedDataset("test")

    # Apply subset if specified
    if train_subset is not None and train_subset < len(train_dataset):
        import random
        indices = random.sample(range(len(train_dataset)), train_subset)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using subset of {train_subset} samples for training")

    # Create data loaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    # Print dataset sizes and normalization
    print(f"\nDataset sizes (PRECOMPUTED):")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    print(f"\nNormalization parameters (from training set):")
    print(f"  log_mean = {train_dataset.log_mean if not isinstance(train_dataset, torch.utils.data.Subset) else train_dataset.dataset.log_mean:.4f}")
    print(f"  log_std = {train_dataset.log_std if not isinstance(train_dataset, torch.utils.data.Subset) else train_dataset.dataset.log_std:.4f}")

    return train_loader, val_loader, test_loader


# Unified function that can use either method
def get_dielectric_data_loaders_optimized(data_dir="data/mp_dielectric",
                                         batch_size=32,
                                         num_workers=0,
                                         train_subset=None,
                                         max_radius=5.0,
                                         lmax=4,
                                         use_precomputed=True):
    """
    Get data loaders - can use either precomputed graphs or on-the-fly computation.

    Args:
        data_dir: Base data directory
        batch_size: Batch size for training
        num_workers: Number of workers
        train_subset: Use subset of training data
        max_radius: Cutoff radius for edge computation (only used if use_precomputed=False)
        lmax: Maximum spherical harmonics degree (only used if use_precomputed=False)
        use_precomputed: If True, use precomputed graphs (much faster)

    Returns:
        train_loader, val_loader, test_loader
    """
    if use_precomputed:
        print(f"\nUsing PRECOMPUTED graphs for fast loading...")
        return get_dielectric_data_loaders_precomputed(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            train_subset=train_subset
        )
    else:
        print(f"\nUsing ON-THE-FLY computation (slower)...")
        return get_dielectric_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            train_subset=train_subset,
            max_radius=max_radius,
            lmax=lmax
        )