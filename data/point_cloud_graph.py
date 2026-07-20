"""Shared point-cloud graph construction for equivariant datasets."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from compatibility.e3nn import o3, soft_one_hot_linspace


def knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Construct a directed k-NN graph with a CPU KD-tree.

    This avoids the quadratic ``torch.cdist`` allocation and does not require
    the optional ``torch-cluster`` extension.
    """
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"pos must have shape (N, 3), got {tuple(pos.shape)}")
    num_points = pos.shape[0]
    if not 1 <= k < num_points:
        raise ValueError(f"k must satisfy 1 <= k < N, got k={k}, N={num_points}")
    coordinates = pos.detach().cpu().numpy()
    _, neighbors = cKDTree(coordinates).query(coordinates, k=k + 1, workers=1)
    neighbors = np.asarray(neighbors[:, 1:], dtype=np.int64)
    source = torch.arange(num_points, dtype=torch.long).repeat_interleave(k)
    target = torch.from_numpy(neighbors.reshape(-1))
    return torch.stack([source, target], dim=0).to(pos.device)


def compute_edge_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    max_radius: float,
    num_basis: int,
    lmax: int,
) -> dict[str, torch.Tensor]:
    """Compute equivariant edge geometry, RBFs, and smooth cutoff weights."""
    source, target = edge_index
    edge_vec = pos[target] - pos[source]
    edge_len = edge_vec.norm(dim=-1)
    edge_sh = o3.spherical_harmonics(
        o3.Irreps.spherical_harmonics(lmax),
        edge_vec,
        normalize=True,
        normalization="component",
    )
    edge_rbf = soft_one_hot_linspace(
        edge_len,
        start=0.0,
        end=max_radius,
        number=num_basis,
        basis="gaussian",
        cutoff=False,
    )
    edge_weights = 0.5 * (torch.cos(torch.pi * edge_len / max_radius) + 1.0)
    edge_weights = edge_weights * (edge_len < max_radius).to(edge_len.dtype)
    return {
        "edge_vec": edge_vec,
        "edge_sh": edge_sh,
        "edge_rbf": edge_rbf,
        "edge_weights": edge_weights,
    }
