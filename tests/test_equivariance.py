"""Equivariance tests using real PyG Data objects."""

import pytest
import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from torch_geometric.data import Data

from representations import O3IrrepsSpec
from spd_maps import MatrixExponentialMap
from distributions import GaussianNLL
from models import (
    EquivariantBackbone,
    EquivariantMeanHead,
    O3EquivariantSymmetricOperatorHead,
    StructuredProbabilisticPredictor,
)


def _random_rotation():
    """Return a random rotation matrix in SO(3)."""
    q = torch.randn(4)
    q = q / q.norm()
    w, x, y, z = q.unbind()
    R = torch.tensor([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)],
    ])
    return R


def _build_graph(pos, node_features=None, max_radius=3.0, num_basis=8, lmax=2):
    """Build a PyG Data object from positions."""
    # Fully-connected within radius.
    src_list, dst_list = [], []
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i != j:
                src_list.append(i)
                dst_list.append(j)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

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

    # Simple polynomial cutoff.
    edge_weights = 0.5 * (torch.cos(torch.pi * edge_length / max_radius) + 1.0)
    edge_weights[edge_length > max_radius] = 0.0

    if node_features is None:
        node_features = torch.randn(pos.shape[0], 49)

    return Data(
        node_features=node_features,
        pos=pos,
        edge_index=edge_index,
        edge_sh=edge_sh,
        edge_rbf=edge_rbf,
        edge_weights=edge_weights,
        batch=torch.zeros(pos.shape[0], dtype=torch.long),
    )


def _rotate_data(data, R):
    """Rotate a Data object and recompute edge features."""
    max_radius = 3.0
    num_basis = 8
    lmax = 2
    pos_rot = data.pos @ R.T
    return _build_graph(
        pos_rot,
        node_features=data.node_features,
        max_radius=max_radius,
        num_basis=num_basis,
        lmax=lmax,
    )


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_predictor_equivariance(output_irreps):
    torch.manual_seed(42)
    R = _random_rotation()
    output_spec = O3IrrepsSpec(output_irreps)

    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8,
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3EquivariantSymmetricOperatorHead(
        backbone.irreps_out, output_spec, pool=True
    )
    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    )
    model.eval()

    pos = torch.randn(8, 3)
    data = _build_graph(pos)
    data_rot = _rotate_data(data, R)

    with torch.no_grad():
        out = model(data, return_scale=True)
        out_rot = model(data_rot, return_scale=True)

    rho_R = output_spec.representation_matrix(R)
    mu_pred = out["mu"]
    mu_pred_rot = out_rot["mu"]
    mu_rot_expected = mu_pred @ rho_R.T

    err_mu = torch.max(torch.abs(mu_pred_rot - mu_rot_expected)).item()
    assert err_mu < 1e-5

    S = out["scale"]
    S_rot = out_rot["scale"]
    S_rot_expected = rho_R @ S @ rho_R.T

    err_S = torch.max(torch.abs(S_rot - S_rot_expected)).item()
    assert err_S < 1e-4
