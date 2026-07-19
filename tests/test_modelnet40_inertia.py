"""Tests for ModelNet40 inertia dataset integration."""

import os

import numpy as np
import pytest
import torch
from e3nn import o3

from data.modelnet40_inertia_dataset import (
    ModelNet40InertiaDataset,
    _compute_edge_features,
    _point_cloud_to_data,
    _shape_covariance_voigt,
    DEFAULT_CACHE_PATH,
)
from models import (
    EquivariantBackbone,
    EquivariantMeanHead,
    O3QuadraticSymmetricOperatorHead,
    StructuredProbabilisticPredictor,
)
from representations import O3IrrepsSpec
from spd_maps import MatrixExponentialMap
from distributions import GaussianNLL


pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_CACHE_PATH),
    reason=f"ModelNet40 cache not found at {DEFAULT_CACHE_PATH}",
)


def test_dataset_loads():
    ds = ModelNet40InertiaDataset(split="train", num_points=128, num_neighbors=8)
    assert len(ds) > 0
    data = ds[0]
    assert hasattr(data, "pos")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "edge_sh")
    assert hasattr(data, "edge_rbf")
    assert hasattr(data, "edge_weights")
    assert hasattr(data, "y_irreps")
    assert hasattr(data, "y_voigt_mean")
    assert hasattr(data, "y_voigt_std")
    assert hasattr(data, "z")


def test_graph_sizes_match(num_points=256, num_neighbors=8):
    ds = ModelNet40InertiaDataset(
        split="train", num_points=num_points, num_neighbors=num_neighbors
    )
    data = ds[0]
    assert data.pos.shape[0] == num_points
    assert data.z.shape[0] == num_points
    assert data.edge_index.shape[1] == num_points * num_neighbors
    num_edges = data.edge_index.shape[1]
    assert data.edge_sh.shape == (num_edges, o3.Irreps.spherical_harmonics(2).dim)
    assert data.edge_rbf.shape[1] == 8
    assert data.edge_weights.shape[0] == num_edges


def test_target_shape_and_finite():
    ds = ModelNet40InertiaDataset(split="train", num_points=128, num_neighbors=8)
    data = ds[0]
    assert data.y_irreps.shape == (1, 6)
    assert torch.isfinite(data.y_irreps).all()


def test_rotation_equivariance_of_edge_features():
    """Rotating the point cloud must rotate edge_sh accordingly."""
    points = torch.randn(64, 3)
    edge_index = torch.stack([
        torch.arange(64).repeat_interleave(4),
        torch.randint(0, 64, (256,)),
    ], dim=0)

    feats = _compute_edge_features(points, edge_index, max_radius=2.0, num_basis=8, lmax=2)

    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    points_rot = points @ R.T
    feats_rot = _compute_edge_features(points_rot, edge_index, max_radius=2.0, num_basis=8, lmax=2)

    # Edge lengths and RBF are invariant; edge vectors rotate.
    assert torch.allclose(feats["edge_rbf"], feats_rot["edge_rbf"], atol=1e-5)
    assert torch.allclose(
        feats["edge_vec"] @ R.T,
        feats_rot["edge_vec"],
        atol=1e-5,
    )


def test_full_model_forward_backward():
    ds = ModelNet40InertiaDataset(split="train", num_points=128, num_neighbors=8)
    data = ds[0]

    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=16,
        lmax=2,
        num_layers=1,
        atom_feature_dim=49,
        num_basis=8,
        atom_features="learnable",
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

    result = model(data, target=data.y_irreps, return_scale=False)
    assert "loss" in result
    assert result["mu"].shape == (1, 6)

    result["loss"].backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_model_equivariance_under_point_cloud_rotation():
    """A rotated point cloud should yield a rotated irrep prediction."""
    ds = ModelNet40InertiaDataset(split="train", num_points=128, num_neighbors=8)
    data = ds[0]

    output_spec = O3IrrepsSpec("0e + 2e")
    backbone = EquivariantBackbone(
        hidden_dim=16,
        lmax=2,
        num_layers=1,
        atom_feature_dim=49,
        num_basis=8,
        atom_features="learnable",
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
    model.eval()

    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    data_rot = _rotate_data(data, R)

    with torch.no_grad():
        mu = model(data, return_scale=False)["mu"]
        mu_rot = model(data_rot, return_scale=False)["mu"]

    D = output_spec.representation_matrix(R)
    err = torch.norm(mu_rot - mu @ D.T, dim=-1).max().item()
    assert err < 1e-4


def test_shape_covariance_dataset_loads():
    ds = ModelNet40InertiaDataset(
        split="train", target_type="shape_covariance", num_points=128, num_neighbors=8
    )
    assert len(ds) > 0
    data = ds[0]
    assert data.y_irreps.shape == (1, 6)
    assert torch.isfinite(data.y_irreps).all()


def test_shape_covariance_rotation_equivariance():
    """Scalar-normalized shape covariance target is rotation equivariant."""
    from data.tensor_conversions import voigt_to_irreps

    ds = ModelNet40InertiaDataset(
        split="train", target_type="shape_covariance", num_points=128, num_neighbors=8
    )
    data = ds[0]
    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    # Scalar std is shared across all components.
    scalar_std = data.y_voigt_std[0].item()

    pos_rot = data.pos @ R.T
    S_rot_voigt = _shape_covariance_voigt(pos_rot.numpy())
    S_rot_norm = torch.from_numpy(S_rot_voigt).float() / scalar_std
    y_rot_expected = voigt_to_irreps(S_rot_norm.unsqueeze(0)).squeeze(0)

    output_spec = O3IrrepsSpec("0e + 2e")
    D = output_spec.representation_matrix(R)
    y_rot_pred = D @ data.y_irreps.squeeze(0)
    assert torch.allclose(y_rot_pred, y_rot_expected, atol=1e-4)


def test_shape_covariance_computation():
    """Manual shape covariance matches the helper for a random point cloud."""
    points = np.random.randn(64, 3).astype(np.float32)
    centered = points - points.mean(axis=0)
    S = (centered.T @ centered) / len(points)
    expected = np.array([
        S[0, 0], S[1, 1], S[2, 2], S[1, 2], S[0, 2], S[0, 1]
    ], dtype=np.float32)
    assert np.allclose(_shape_covariance_voigt(points), expected, atol=1e-5)


def _rotate_data(data, R):
    """Rotate a point-cloud Data object and recompute edge features."""
    from e3nn.math import soft_one_hot_linspace
    from data.modelnet40_inertia_dataset import _knn_graph

    data_rot = data.clone()
    pos_rot = data.pos @ R.T
    data_rot.pos = pos_rot
    k = data.num_neighbors if hasattr(data, "num_neighbors") else 8
    data_rot.edge_index = _knn_graph(pos_rot, k=k)

    row, col = data_rot.edge_index
    edge_vec = pos_rot[col] - pos_rot[row]
    edge_len = edge_vec.norm(dim=-1)
    irreps_sh = o3.Irreps.spherical_harmonics(2)
    data_rot.edge_sh = o3.spherical_harmonics(
        irreps_sh, edge_vec, normalize=True, normalization="component"
    )
    data_rot.edge_rbf = soft_one_hot_linspace(
        edge_len, start=0.0, end=2.0, number=8, basis="gaussian", cutoff=False
    )
    data_rot.edge_weights = 0.5 * (torch.cos(torch.pi * edge_len / 2.0) + 1.0)
    data_rot.edge_weights = data_rot.edge_weights * (edge_len < 2.0).float()
    return data_rot
