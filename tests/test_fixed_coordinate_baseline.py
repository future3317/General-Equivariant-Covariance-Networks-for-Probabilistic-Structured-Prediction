"""Tests for the conventional fixed-coordinate uncertainty control."""

import torch

from models.fixed_coordinate_baseline import (
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
