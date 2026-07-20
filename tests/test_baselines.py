"""Tests for baseline heads and SPD maps."""

import pytest
import torch
from torch_geometric.data import Data

from representations import O3IrrepsSpec
from spd_maps import IsotropicMap, IrrepBlockDiagonalMap
from models import (
    EquivariantBackbone,
    DeterministicHead,
    IsotropicCovarianceHead,
    IrrepBlockDiagonalCovarianceHead,
    BaselineProbabilisticPredictor,
)
from distributions import GaussianNLL


def _make_graph_data(num_graphs=2, num_nodes=6):
    pos = torch.randn(num_graphs * num_nodes, 3)
    src, dst = [], []
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i != j and (pos[i] - pos[j]).norm() < 2.0:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
    edge_length = edge_vec.norm(dim=-1)
    from e3nn import o3
    from e3nn.math import soft_one_hot_linspace

    irreps_sh = o3.Irreps.spherical_harmonics(2)
    edge_sh = o3.spherical_harmonics(
        irreps_sh, edge_vec, normalize=True, normalization="component"
    )
    edge_rbf = soft_one_hot_linspace(
        edge_length, start=0.0, end=3.0, number=8, basis="gaussian", cutoff=False
    )
    edge_weights = torch.ones(edge_length.shape)
    batch = torch.arange(num_graphs).repeat_interleave(num_nodes)
    return Data(
        node_features=torch.randn(num_graphs * num_nodes, 49),
        edge_index=edge_index,
        edge_sh=edge_sh,
        edge_rbf=edge_rbf,
        edge_weights=edge_weights,
        batch=batch,
    )


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_deterministic_head_forward(output_irreps):
    output_spec = O3IrrepsSpec(output_irreps)
    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8
    )
    head = DeterministicHead(backbone.irreps_out, output_spec, pool=True)
    data = _make_graph_data()
    node_features, batch = backbone(data)
    mu = head(node_features, batch)
    assert mu.shape == (2, output_spec.dim)


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_isotropic_head_and_map(output_irreps):
    output_spec = O3IrrepsSpec(output_irreps)
    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8
    )
    head = IsotropicCovarianceHead(backbone.irreps_out, output_spec, pool=True)
    data = _make_graph_data()
    node_features, batch = backbone(data)
    mu, params = head(node_features, batch)
    assert mu.shape == (2, output_spec.dim)
    assert params.shape == (2, 1)

    spd_map = IsotropicMap(dim=output_spec.dim)
    S = spd_map(params)
    assert S.shape == (2, output_spec.dim, output_spec.dim)
    assert torch.allclose(S, S.transpose(-1, -2))
    eigs = torch.linalg.eigvalsh(S)
    assert eigs.min().item() > 0


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_irrep_block_diag_head_and_map(output_irreps):
    output_spec = O3IrrepsSpec(output_irreps)
    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8
    )
    head = IrrepBlockDiagonalCovarianceHead(backbone.irreps_out, output_spec, pool=True)
    data = _make_graph_data()
    node_features, batch = backbone(data)
    mu, params = head(node_features, batch)
    num_blocks = sum(mul for mul, _ in output_spec.irreps)
    assert mu.shape == (2, output_spec.dim)
    assert params.shape == (2, num_blocks)

    spd_map = IrrepBlockDiagonalMap(output_spec.irreps)
    S = spd_map(params)
    assert S.shape == (2, output_spec.dim, output_spec.dim)
    eigs = torch.linalg.eigvalsh(S)
    assert eigs.min().item() > 0


def test_baseline_predictor_isotropic():
    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8
    )
    head = IsotropicCovarianceHead(backbone.irreps_out, output_spec, pool=True)
    spd_map = IsotropicMap(dim=output_spec.dim)
    model = BaselineProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        baseline_head=head,
        spd_map=spd_map,
        distribution=GaussianNLL(),
    )
    data = _make_graph_data()
    target = torch.randn(2, output_spec.dim)
    result = model(data, target=target)
    assert "mu" in result
    assert "scale" in result
    assert "loss" in result
    result["loss"].backward()


def test_deterministic_baseline_uses_unified_predictor_path():
    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=8,
        lmax=2,
        num_layers=1,
        atom_feature_dim=49,
        num_basis=8,
    )
    model = BaselineProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        baseline_head=DeterministicHead(backbone.irreps_out, output_spec),
        spd_map=None,
        distribution=None,
    )
    target = torch.randn(2, output_spec.dim)
    result = model(_make_graph_data(), target=target)
    assert set(result) == {"mu", "loss"}
    assert model.baseline_head is model.joint_head
    assert not any(key.startswith("baseline_head.") for key in model.state_dict())
