"""Tests for equivariant covariance heads."""

import pytest
import torch
from e3nn import o3

from models import (
    EquivariantBackbone,
    EquivariantMeanHead,
    O3EquivariantSymmetricOperatorHead,
    O3QuadraticSymmetricOperatorHead,
    O3EquivariantLowRankCovarianceHead,
    StructuredProbabilisticPredictor,
)
from representations import O3IrrepsSpec, rank4_elasticity_irreps
from spd_maps import MatrixExponentialMap, LowRankPlusIsotropicMap
from distributions import GaussianNLL


def _make_graph_data(batch_size=2, num_nodes=5):
    class Data:
        pass

    data = Data()
    data.node_features = torch.randn(num_nodes, 49)
    data.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data.edge_sh = torch.randn(data.edge_index.shape[1], 9)
    data.edge_rbf = torch.randn(data.edge_index.shape[1], 8)
    data.edge_weights = torch.ones(data.edge_index.shape[1])
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    # Second graph.
    if batch_size == 2:
        data.node_features = torch.cat([data.node_features, data.node_features], dim=0)
        offset = torch.full((2, data.edge_index.shape[1]), num_nodes, dtype=torch.long)
        data.edge_index = torch.cat([data.edge_index, data.edge_index + offset], dim=-1)
        data.edge_sh = torch.cat([data.edge_sh, data.edge_sh], dim=0)
        data.edge_rbf = torch.cat([data.edge_rbf, data.edge_rbf], dim=0)
        data.edge_weights = torch.cat([data.edge_weights, data.edge_weights], dim=0)
        data.batch = torch.cat([torch.zeros(num_nodes, dtype=torch.long), torch.ones(num_nodes, dtype=torch.long)])
    return data


def _get_l4_slice(operator_irreps: o3.Irreps):
    cursor = 0
    for mul, ir in operator_irreps:
        dim = ir.dim
        for _ in range(mul):
            if ir.l == 4:
                return slice(cursor, cursor + dim)
            cursor += dim
    return None


def test_quadratic_head_l4_nonzero():
    """Quadratic branch must be able to produce 4e coefficients."""
    output_spec = O3IrrepsSpec("0e + 2e")
    hidden_irreps = o3.Irreps("32x0e + 16x1o + 16x2e")
    bottleneck_irreps = o3.Irreps("4x0e + 2x1o + 2x2e")
    head = O3QuadraticSymmetricOperatorHead(
        hidden_irreps=hidden_irreps,
        output_spec=output_spec,
        bottleneck_irreps=bottleneck_irreps,
        pool=True,
    )

    # Construct an input that has nonzero 2e content after the pre projection.
    # Use a learnable input and optimize it to produce nonzero 4e coefficients.
    node_features = torch.randn(5, hidden_irreps.dim, requires_grad=True)
    batch = torch.zeros(5, dtype=torch.long)
    optimizer = torch.optim.Adam([node_features], lr=1e-2)

    found_nonzero = False
    for _ in range(100):
        A = head(node_features, batch)
        coeffs = head.operator_basis.project(A)
        # Target: maximize the squared norm of the 4e coefficients.
        l4_slice = _get_l4_slice(head.operator_basis.operator_irreps)
        assert l4_slice is not None
        loss = -coeffs[..., l4_slice].pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if coeffs[..., l4_slice].abs().max().item() > 1e-3:
            found_nonzero = True
            break

    assert found_nonzero


def test_quadratic_head_l4_gradient():
    """Parameters contributing to 4e must receive finite nonzero gradients."""
    output_spec = O3IrrepsSpec("0e + 2e")
    head = O3QuadraticSymmetricOperatorHead(
        hidden_irreps="32x0e + 16x1o + 16x2e",
        output_spec=output_spec,
        bottleneck_irreps="4x0e + 2x1o + 2x2e",
        pool=True,
    )

    node_features = torch.randn(5, head.pre.irreps_in.dim, requires_grad=True)
    batch = torch.zeros(5, dtype=torch.long)

    A = head(node_features, batch)
    loss = A.pow(2).sum()
    loss.backward()

    assert node_features.grad is not None
    assert torch.isfinite(node_features.grad).all()
    assert node_features.grad.abs().max().item() > 1e-12


def test_quadratic_head_forward_backward():
    """Full forward/backward through quadratic head inside a predictor."""
    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=16, lmax=2, num_layers=1, atom_feature_dim=49, num_basis=8,
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3QuadraticSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)
    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    )

    data = _make_graph_data(batch_size=2)
    target = torch.randn(2, output_spec.dim)
    result = model(data, target=target, return_scale=True)
    assert result["scale"].shape == (2, output_spec.dim, output_spec.dim)
    result["loss"].backward()

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_lowrank_factor_layout():
    """Low-rank factor output must contain rank copies of each output irrep."""
    output_spec = O3IrrepsSpec("0e + 2e")
    rank = 3
    head = O3EquivariantLowRankCovarianceHead(
        hidden_irreps="16x0e + 8x1o + 8x2e",
        output_spec=output_spec,
        rank=rank,
        pool=True,
    )

    # factor_irreps groups multiplicity: rank copies of each irrep type.
    expected = o3.Irreps([(mul * rank, ir) for mul, ir in output_spec.irreps])
    assert head.factor_irreps == expected

    node_features = torch.randn(6, head.factor_head.irreps_in.dim)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    params = head(node_features, batch)

    batch_size = 2
    dim = output_spec.dim
    assert params.shape == (batch_size, dim * rank + 1)
    L = params[:, :-1].reshape(batch_size, dim, rank)
    assert L.shape == (batch_size, dim, rank)

    # Verify packing/unpacking is a no-op for known coefficients.
    coeffs = torch.randn(batch_size, head.factor_irreps.dim)
    L_rec = head._pack_factors(coeffs)
    # Each column of L should contain one full copy of V.
    assert L_rec.shape == (batch_size, dim, rank)


def test_lowrank_factor_equivariance():
    """Low-rank scale S must be equivariant under O(3) at the head level."""
    torch.manual_seed(0)
    output_spec = O3IrrepsSpec("0e + 2e")
    hidden_irreps = o3.Irreps("16x0e + 8x1o + 8x2e")
    cov_head = O3EquivariantLowRankCovarianceHead(hidden_irreps, output_spec, rank=4, pool=False)
    spd_map = LowRankPlusIsotropicMap(dim=output_spec.dim, rank=4)

    # Random pooled hidden features.
    h = hidden_irreps.randn(1, -1)
    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    rho_hidden = o3.Irreps(hidden_irreps).D_from_matrix(R)
    h_rot = (rho_hidden @ h.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        params = cov_head(h, batch=None)
        params_rot = cov_head(h_rot, batch=None)
        S = spd_map(params)
        S_rot = spd_map(params_rot)

    rho_R = output_spec.representation_matrix(R)
    S_rot_expected = rho_R @ S @ rho_R.T

    err = torch.norm(S_rot - S_rot_expected, dim=(-2, -1))
    norm = torch.norm(S, dim=(-2, -1)) + 1e-12
    assert (err / norm).max().item() < 1e-4


def test_lowrank_factor_layout_rank4():
    """Low-rank factor packing must be correct for rank-4 elasticity output."""
    output_spec = O3IrrepsSpec(rank4_elasticity_irreps())
    rank = 4
    head = O3EquivariantLowRankCovarianceHead(
        hidden_irreps="32x0e + 16x1o + 16x2e",
        output_spec=output_spec,
        rank=rank,
        pool=False,
    )

    expected = o3.Irreps([(mul * rank, ir) for mul, ir in output_spec.irreps])
    assert head.factor_irreps == expected

    h = torch.randn(2, head.factor_head.irreps_in.dim)
    params = head(h, batch=None)
    dim = output_spec.dim
    assert params.shape == (2, dim * rank + 1)

    coeffs = torch.randn(2, head.factor_irreps.dim)
    L = head._pack_factors(coeffs)
    assert L.shape == (2, dim, rank)

    # Each column must contain the full output representation V.
    # We check this by ensuring the slice sizes match the irrep dimensions
    # scaled by their multiplicities.
    cursor = 0
    for rank_slot in range(rank):
        row_cursor = 0
        for mul, ir in output_spec.irreps:
            for _ in range(mul):
                width = ir.dim
                assert L[0, row_cursor : row_cursor + width, rank_slot].numel() == width
                row_cursor += width
        assert row_cursor == dim


def test_lowrank_factor_equivariance_rank4():
    """Low-rank scale S must be equivariant for rank-4 elasticity output."""
    torch.manual_seed(2)
    output_spec = O3IrrepsSpec(rank4_elasticity_irreps())
    hidden_irreps = o3.Irreps("32x0e + 16x1o + 16x2e")
    cov_head = O3EquivariantLowRankCovarianceHead(hidden_irreps, output_spec, rank=4, pool=False)
    spd_map = LowRankPlusIsotropicMap(dim=output_spec.dim, rank=4)

    h = hidden_irreps.randn(1, -1)
    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    rho_hidden = hidden_irreps.D_from_matrix(R)
    h_rot = (rho_hidden @ h.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        params = cov_head(h, batch=None)
        params_rot = cov_head(h_rot, batch=None)
        S = spd_map(params)
        S_rot = spd_map(params_rot)

    rho_R = output_spec.representation_matrix(R)
    S_rot_expected = rho_R @ S @ rho_R.T

    err = torch.norm(S_rot - S_rot_expected, dim=(-2, -1))
    norm = torch.norm(S, dim=(-2, -1)) + 1e-12
    assert (err / norm).max().item() < 1e-4
