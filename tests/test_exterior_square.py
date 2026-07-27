import torch

from compatibility.e3nn import o3
from representations import O3SkewOperatorBasis, exterior_square_irreps


def test_exterior_square_rank2_decomposition_and_basis():
    irreps = o3.Irreps("0e + 2e")
    assert str(exterior_square_irreps(irreps)) == "1x1e+1x2e+1x3e"
    basis = O3SkewOperatorBasis(irreps)
    assert basis.operator_dim == 15
    gram = torch.einsum("qij,rij->qr", basis.basis, basis.basis)
    torch.testing.assert_close(gram, torch.eye(15), atol=1e-6, rtol=1e-6)
    coefficients = torch.randn(4, 15)
    matrix = basis.assemble(coefficients)
    torch.testing.assert_close(matrix, -matrix.transpose(-1, -2))
    torch.testing.assert_close(basis.project(matrix), coefficients, atol=1e-6, rtol=1e-6)


def test_exterior_square_dimension_matches_d_choose_2():
    for irreps in ("1o", "0e + 2e", "2x1o", "1o + 2e"):
        basis = O3SkewOperatorBasis(o3.Irreps(irreps))
        d = o3.Irreps(irreps).dim
        assert basis.operator_dim == d * (d - 1) // 2
