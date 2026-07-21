"""Numerical equivariance validation for trained predictors."""

from __future__ import annotations

import torch
from compatibility.torch_geometric import Data


def _rotate_data(
    data: Data,
    R: torch.Tensor,
    max_radius: float = 3.0,
    num_basis: int = 8,
    lmax: int = 2,
):
    """Rotate a PyG Data object and recompute edge features.

    This is a minimal helper. For full validation it should match the
    preprocessing used during training.
    """
    from e3nn import o3
    from e3nn.math import soft_one_hot_linspace

    data_rot = data.clone()
    data_rot.pos = data.pos @ R.T

    # Recompute radius graph and features.
    pos = data_rot.pos
    src, dst = [], []
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
    edge_length = edge_vec.norm(dim=-1)
    mask = edge_length < max_radius
    edge_index = edge_index[:, mask]
    edge_vec = edge_vec[mask]
    edge_length = edge_length[mask]

    irreps_sh = o3.Irreps.spherical_harmonics(lmax)
    edge_sh = o3.spherical_harmonics(
        irreps_sh, edge_vec, normalize=True, normalization="component"
    )
    edge_rbf = soft_one_hot_linspace(
        edge_length,
        start=0.0,
        end=max_radius,
        number=num_basis,
        basis="gaussian",
        cutoff=False,
    )
    edge_weights = 0.5 * (torch.cos(torch.pi * edge_length / max_radius) + 1.0)
    edge_weights[edge_length > max_radius] = 0.0

    data_rot.edge_index = edge_index
    data_rot.edge_sh = edge_sh
    data_rot.edge_rbf = edge_rbf
    data_rot.edge_weights = edge_weights
    return data_rot


def mean_equivariance_error(
    model,
    data: Data,
    R: torch.Tensor,
    output_spec,
) -> float:
    """Mean equivariance error of predicted mean under rotation.

    Computes :math:`\\|\\mu(Rx) - \\rho(R)\\mu(x)\\|_2 / \\|\\mu(x)\\|_2`.

    Args:
        model: Trained predictor.
        data: Single PyG Data object.
        R: Rotation matrix ``(3, 3)``.
        output_spec: Output representation spec (e.g. ``O3IrrepsSpec``).

    Returns:
        Relative mean equivariance error.
    """
    model.eval()
    data_rot = _rotate_data(data, R)
    rho_R = output_spec.representation_matrix(R)

    with torch.no_grad():
        mu = model(data, return_scale=False)["mu"]
        mu_rot = model(data_rot, return_scale=False)["mu"]
        mu_rot_expected = mu @ rho_R.T

    err = torch.norm(mu_rot - mu_rot_expected, dim=-1)
    norm = torch.norm(mu, dim=-1) + 1e-12
    return (err / norm).mean().item()


def scale_equivariance_error(
    model,
    data: Data,
    R: torch.Tensor,
    output_spec,
) -> float:
    """Relative scale/covariance equivariance error under rotation.

    Computes :math:`\\|S(Rx) - \\rho(R) S(x) \\rho(R)^T\\|_F / \\|S(x)\\|_F`.
    """
    model.eval()
    data_rot = _rotate_data(data, R)
    rho_R = output_spec.representation_matrix(R)

    with torch.no_grad():
        S = model(data, return_scale=True)["scale"]
        S_rot = model(data_rot, return_scale=True)["scale"]
        S_rot_expected = rho_R @ S @ rho_R.T

    err = torch.norm(S_rot - S_rot_expected, dim=(-2, -1))
    norm = torch.norm(S, dim=(-2, -1)) + 1e-12
    return (err / norm).mean().item()


def average_equivariance_error(
    model,
    data: Data,
    output_spec,
    num_rotations: int = 10,
) -> dict[str, float]:
    """Average equivariance error over random rotations."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    mean_errors = []
    scale_errors = []
    for _ in range(num_rotations):
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.get_default_dtype())
        mean_errors.append(mean_equivariance_error(model, data, R, output_spec))
        scale_errors.append(scale_equivariance_error(model, data, R, output_spec))

    return {
        "mean_equivariance_error": float(np.mean(mean_errors)),
        "scale_equivariance_error": float(np.mean(scale_errors)),
        "max_mean_equivariance_error": float(np.max(mean_errors)),
        "max_scale_equivariance_error": float(np.max(scale_errors)),
    }
