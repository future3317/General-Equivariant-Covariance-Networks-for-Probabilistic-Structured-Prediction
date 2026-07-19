"""Integration tests for full predictor."""

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from torch_geometric.data import Data

from representations import O3IrrepsSpec
from spd_maps import MatrixExponentialMap, LowRankPlusIsotropicMap
from distributions import GaussianNLL
from models import (
    EquivariantBackbone,
    EquivariantMeanHead,
    O3EquivariantSymmetricOperatorHead,
    O3EquivariantLowRankCovarianceHead,
    StructuredProbabilisticPredictor,
)


def _make_data(num_graphs=2, num_nodes=6):
    pos = torch.randn(num_graphs * num_nodes, 3)
    # Simple radius graph.
    src, dst = [], []
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i != j and (pos[i] - pos[j]).norm() < 2.0:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
    edge_length = edge_vec.norm(dim=-1)
    irreps_sh = o3.Irreps.spherical_harmonics(2)
    edge_sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization="component")
    edge_rbf = soft_one_hot_linspace(
        edge_length, start=0.0, end=3.0, number=8, basis="gaussian", cutoff=False
    )
    edge_weights = torch.ones(edge_length.shape)
    batch = torch.arange(num_graphs).repeat_interleave(num_nodes)
    return Data(
        node_features=torch.randn(num_graphs * num_nodes, 49),
        z=torch.randint(1, 100, (num_graphs * num_nodes,)),
        edge_index=edge_index,
        edge_sh=edge_sh,
        edge_rbf=edge_rbf,
        edge_weights=edge_weights,
        batch=batch,
    )


def test_full_rank_rank2_forward_backward():
    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8,
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3EquivariantSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)
    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    )
    data = _make_data()
    target = torch.randn(2, output_spec.dim)
    result = model(data, target, return_scale=True)
    assert result["mu"].shape == (2, output_spec.dim)
    assert result["scale"].shape == (2, output_spec.dim, output_spec.dim)
    result["loss"].backward()


def test_low_rank_rank2_forward_backward():
    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8,
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3EquivariantLowRankCovarianceHead(backbone.irreps_out, output_spec, rank=4, pool=True)
    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=LowRankPlusIsotropicMap(dim=output_spec.dim, rank=4),
        distribution=GaussianNLL(),
    )
    data = _make_data()
    target = torch.randn(2, output_spec.dim)
    result = model(data, target, return_scale=True)
    assert result["mu"].shape == (2, output_spec.dim)
    assert result["scale"].shape == (2, output_spec.dim, output_spec.dim)
    result["loss"].backward()


def test_vector_output_forward_backward():
    output_spec = O3IrrepsSpec("1o")
    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8,
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3EquivariantSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)
    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    )
    data = _make_data()
    target = torch.randn(2, output_spec.dim)
    result = model(data, target, return_scale=True)
    assert result["mu"].shape == (2, 3)
    assert result["scale"].shape == (2, 3, 3)
    result["loss"].backward()


def test_learnable_atom_features_forward_backward():
    output_spec = O3IrrepsSpec("1o")
    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8,
        atom_features="learnable",
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3EquivariantSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)
    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    )
    data = _make_data()
    target = torch.randn(2, output_spec.dim)
    result = model(data, target, return_scale=True)
    assert result["mu"].shape == (2, 3)
    assert result["scale"].shape == (2, 3, 3)
    result["loss"].backward()
    assert backbone.atom_embedding.weight.grad is not None
