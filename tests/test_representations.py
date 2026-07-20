"""Tests for representation layer."""

import pytest
import torch
from representations import (
    O3IrrepsSpec,
    O3SymmetricOperatorBasis,
    rank2_symmetric_irreps,
    rank4_elasticity_irreps,
)


def test_representation_matrix_preserves_floating_dtype():
    rotation = torch.eye(3, dtype=torch.float64)
    representation = O3IrrepsSpec("0e + 2e").representation_matrix(rotation)
    assert representation.dtype == torch.float64


@pytest.mark.parametrize(
    "irreps,expected_operator_dim",
    [
        ("1o", 6),
        ("0e + 2e", 21),
    ],
)
def test_symmetric_square_dimension(irreps, expected_operator_dim):
    basis = O3SymmetricOperatorBasis(irreps)
    assert basis.operator_dim == expected_operator_dim
    assert basis.output_dim == basis.output_irreps.dim


def test_rank4_elasticity_irreps():
    irreps = rank4_elasticity_irreps()
    basis = O3SymmetricOperatorBasis(irreps)
    assert basis.output_dim == 21
    assert basis.operator_dim == 21 * 22 // 2


def test_basis_symmetry():
    basis = O3SymmetricOperatorBasis(rank2_symmetric_irreps())
    B = basis.basis
    sym_err = torch.max(torch.abs(B - B.transpose(-1, -2))).item()
    assert sym_err < 1e-6


def test_basis_orthonormality():
    basis = O3SymmetricOperatorBasis(rank2_symmetric_irreps())
    B = basis.basis
    inner = torch.einsum("qij,pij->qp", B, B)
    identity = torch.eye(basis.operator_dim)
    err = torch.max(torch.abs(inner - identity)).item()
    assert err < 1e-5


def test_coefficient_round_trip():
    basis = O3SymmetricOperatorBasis(rank2_symmetric_irreps())
    coeffs = torch.randn(4, basis.operator_dim)
    A = basis.assemble(coeffs)
    coeffs_back = basis.project(A)
    err = torch.max(torch.abs(coeffs - coeffs_back)).item()
    assert err < 1e-5
