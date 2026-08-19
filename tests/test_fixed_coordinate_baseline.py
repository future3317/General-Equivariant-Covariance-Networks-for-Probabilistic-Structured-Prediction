"""Tests for the conventional fixed-coordinate uncertainty control."""

import torch

from models.fixed_coordinate_baseline import (
    FixedCoordinateCholeskyMap,
    FixedCoordinateCholeskyReadout,
    FixedCoordinateDiagonalMap,
    FixedCoordinateDiagonalReadout,
)


def test_fixed_coordinate_diagonal_map_statistics_match_dense_reference():
    mapping = FixedCoordinateDiagonalMap()
    params = torch.randn(5, 45, dtype=torch.float64)
    residual = torch.randn(5, 45, dtype=torch.float64)
    scale = mapping(params)

    logdet, quadratic = mapping.statistics(params, residual)
    assert torch.allclose(logdet, torch.linalg.slogdet(scale).logabsdet)
    assert torch.allclose(
        quadratic,
        torch.einsum("bi,bij,bj->b", residual, torch.linalg.inv(scale), residual),
    )


def test_fixed_coordinate_readout_emits_one_log_variance_per_coordinate():
    readout = FixedCoordinateDiagonalReadout(feature_dim=12, output_dim=45)
    params = readout(torch.randn(7, 12))
    assert params.shape == (7, 45)


def test_fixed_coordinate_cholesky_map_is_strict_spd_and_matches_statistics():
    torch.manual_seed(4)
    mapping = FixedCoordinateCholeskyMap(output_dim=6)
    params = torch.randn(5, mapping.parameter_count, dtype=torch.float64)
    residual = torch.randn(5, 6, dtype=torch.float64)
    scale = mapping(params)
    eigenvalues = torch.linalg.eigvalsh(scale)
    assert bool((eigenvalues > 0).all())
    logdet, quadratic = mapping.statistics(params, residual)
    torch.testing.assert_close(logdet, torch.linalg.slogdet(scale).logabsdet)
    torch.testing.assert_close(
        quadratic,
        torch.einsum("bi,bij,bj->b", residual, torch.linalg.inv(scale), residual),
        rtol=1e-10,
        atol=1e-10,
    )


def test_fixed_coordinate_cholesky_readout_uses_triangular_parameter_count():
    readout = FixedCoordinateCholeskyReadout(feature_dim=12, output_dim=6)
    params = readout(torch.randn(7, 12))
    assert readout.parameter_count == 21
    assert params.shape == (7, 21)


def test_fixed_coordinate_cholesky_readout_starts_at_identity_scale():
    readout = FixedCoordinateCholeskyReadout(feature_dim=12, output_dim=6)
    mapping = FixedCoordinateCholeskyMap(output_dim=6)
    params = readout(torch.randn(7, 12))
    scale = mapping(params)
    torch.testing.assert_close(scale, torch.eye(6).expand(7, 6, 6), atol=1e-5, rtol=1e-5)
