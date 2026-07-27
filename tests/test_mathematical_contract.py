import math

import torch

from compatibility.e3nn import o3
from evaluation.calibration import marginal_interval_quantile
from representations.symmetric_square import O3SymmetricOperatorBasis
from spd_maps.centered_spectral_window import CenteredSpectralWindowMap
from spd_maps.graph_precision import GraphStructuredPrecisionMap
from spd_maps.isotypic_block import IsotypicBlockMap
from spd_maps.low_rank import LowRankPlusIsotropicMap
from representations.graph_structure import EquivariantOutputGraph


def test_student_t_marginal_uses_scale_and_t_quantile():
    assert math.isclose(
        marginal_interval_quantile(0.95, reference="student_t", student_t_dof=5.0),
        2.570581835636305,
        rel_tol=1e-6,
    )
    assert 2.2 < marginal_interval_quantile(
        0.95, reference="student_t", student_t_dof=5.0
    )


def test_float64_basis_is_constructed_natively():
    basis = O3SymmetricOperatorBasis(o3.Irreps("0e+2e"), dtype=torch.float64)
    gram = torch.einsum("qij,rij->qr", basis.basis, basis.basis)
    assert basis.basis.dtype == torch.float64
    assert torch.allclose(gram, torch.eye(21, dtype=torch.float64), atol=1e-10, rtol=1e-10)


def test_low_rank_zero_isotropic_and_full_rank_degeneracies():
    torch.manual_seed(0)
    zero = LowRankPlusIsotropicMap(4, 0, min_sigma2=0.0)
    params = torch.randn(3, 1)
    residual = torch.randn(3, 4)
    scale = zero(params)
    sigma2 = torch.nn.functional.softplus(params[:, 0])
    assert torch.allclose(scale, sigma2[:, None, None] * torch.eye(4), atol=1e-6)
    assert torch.allclose(
        zero.precision_action(params, residual), residual.square().sum(-1) / sigma2, atol=1e-6
    )


def test_graph_empty_and_single_node_degenerate_to_local_spd():
    torch.manual_seed(0)
    graph = EquivariantOutputGraph(1, ())
    mapping = GraphStructuredPrecisionMap(graph)
    params = torch.randn(2, 1, 3, 3)
    params = 0.5 * (params + params.transpose(-1, -2))
    local = torch.linalg.matrix_exp(params[:, 0])
    assert torch.allclose(mapping.precision(params), local, atol=1e-6)
    assert torch.allclose(mapping(params), torch.linalg.inv(local), atol=1e-6)


def test_centered_shape_zero_isotropic_volume():
    mapping = CenteredSpectralWindowMap(-2.0, 2.0, -8.0, 8.0)
    A = torch.eye(4) * 0.7
    scale = mapping(A)
    assert torch.allclose(scale, scale[0, 0] * torch.eye(4), atol=1e-6)


def test_isotypic_block_has_no_fixed_floor_by_default():
    mapping = IsotypicBlockMap(o3.Irreps("1x0e"))
    assert mapping.min_diagonal == 0.0
