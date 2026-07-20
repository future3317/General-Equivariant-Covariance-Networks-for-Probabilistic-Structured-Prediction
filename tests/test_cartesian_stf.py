"""Exactness tests for STF-coordinate and dense-projector lowering."""

from contextlib import contextmanager

import pytest
import torch
from e3nn import o3

from models import O3QuadraticSymmetricOperatorHead
from representations import (
    CompilerConfig,
    MultiplicityFirstCartesianTensorSquare,
    MultiplicityFirstDenseTensorProduct,
    O3IrrepsSpec,
    O3RepresentationCompiler,
    O3SymmetricOperatorBasis,
    Rank2CartesianSTFOperatorBasis,
    rank4_elasticity_irreps,
)


SEED = o3.Irreps("4x0e + 2x1o + 2x2e")
MIXED_PARITY_SEED = o3.Irreps(
    "4x0e + 2x1e + 2x1o + 2x2e + 2x2o"
)
OPERATOR_IRREPS = o3.Irreps("2x0e + 2x2e + 1x4e")


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _transform(features: torch.Tensor, irreps: o3.Irreps, matrix: torch.Tensor):
    return features @ irreps.D_from_matrix(matrix).T


def test_frozen_stf_projectors_are_complete_orthonormal_symmetric_basis():
    basis = Rank2CartesianSTFOperatorBasis().double()
    assert basis.P0.shape == (1, 25)
    assert basis.P2.shape == (5, 25)
    assert basis.P4.shape == (9, 25)
    projector = torch.cat([basis.P0, basis.P2, basis.P4])
    torch.testing.assert_close(
        projector @ projector.T,
        torch.eye(15, dtype=torch.float64),
        atol=1e-6,
        rtol=1e-6,
    )
    assert not any(name in basis.state_dict() for name in {"P0", "P2", "P4"})


def test_cartesian_abpqh_formula_is_bijective_and_matches_spherical_basis():
    torch.manual_seed(0)
    cartesian = Rank2CartesianSTFOperatorBasis().double()
    spherical = O3SymmetricOperatorBasis("0e + 2e").double()
    components = torch.randn(7, 21, dtype=torch.float64)
    coefficients = cartesian.irreps_from_components(components)
    recovered = cartesian.components_from_irreps(coefficients)
    torch.testing.assert_close(recovered, components, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(
        cartesian.assemble_components(components),
        spherical.assemble(coefficients),
        atol=2e-7,
        rtol=2e-7,
    )
    assert torch.linalg.matrix_rank(cartesian.canonical_basis.flatten(1)) == 21


def test_exact_multiplicity_first_square_matches_cg_and_all_gradients():
    torch.manual_seed(1)
    spherical = o3.TensorSquare(SEED, irreps_out=OPERATOR_IRREPS).double()
    cartesian = MultiplicityFirstCartesianTensorSquare(
        SEED, OPERATOR_IRREPS
    ).double()
    cartesian.load_e3nn_weights(spherical)

    spherical_input = SEED.randn(5, -1, dtype=torch.float64).requires_grad_()
    cartesian_input = spherical_input.detach().clone().requires_grad_()
    cotangent = torch.randn(5, OPERATOR_IRREPS.dim, dtype=torch.float64)
    spherical_output = spherical(spherical_input)
    cartesian_output = cartesian(cartesian_input)
    spherical_gradients = torch.autograd.grad(
        (spherical_output * cotangent).sum(),
        (spherical_input, spherical.weight),
    )
    cartesian_gradients = torch.autograd.grad(
        (cartesian_output * cotangent).sum(),
        (cartesian_input, cartesian.weight),
    )

    torch.testing.assert_close(
        cartesian_output, spherical_output, atol=2e-7, rtol=2e-7
    )
    for actual, expected in zip(cartesian_gradients, spherical_gradients):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)


def test_dense_projector_lowering_preserves_fctp_weight_coordinates():
    with _default_dtype(torch.float64):
        spherical = o3.FullyConnectedTensorProduct(
            MIXED_PARITY_SEED, MIXED_PARITY_SEED, o3.Irreps("3x0e + 3x2e + 1x4e")
        )
        lowered = MultiplicityFirstDenseTensorProduct(
            MIXED_PARITY_SEED, MIXED_PARITY_SEED, spherical.irreps_out
        )
        lowered.load_state_dict(spherical.state_dict(), strict=True)
        spherical_input = (
            0.1 * MIXED_PARITY_SEED.randn(4, -1, dtype=torch.float64)
        ).requires_grad_()
        lowered_input = spherical_input.detach().clone().requires_grad_()
        cotangent = torch.randn(4, spherical.irreps_out.dim, dtype=torch.float64)
        spherical_output = spherical(spherical_input, spherical_input)
        lowered_output = lowered(lowered_input, lowered_input)
        spherical_gradients = torch.autograd.grad(
            (spherical_output * cotangent).sum(),
            (spherical_input, spherical.weight),
        )
        lowered_gradients = torch.autograd.grad(
            (lowered_output * cotangent).sum(),
            (lowered_input, lowered.weight),
        )
    torch.testing.assert_close(lowered_output, spherical_output, atol=2e-12, rtol=2e-12)
    for actual, expected in zip(lowered_gradients, spherical_gradients):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize(
    "matrix",
    [
        torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
    ],
)
def test_cartesian_stf_operator_is_o3_equivariant(matrix):
    torch.manual_seed(2)
    square = MultiplicityFirstCartesianTensorSquare(SEED).double()
    basis = Rank2CartesianSTFOperatorBasis().double()
    features = SEED.randn(4, -1, dtype=torch.float64)
    transformed = _transform(features, SEED, matrix)
    operator = basis.assemble(square(features))
    transformed_operator = basis.assemble(square(transformed))
    rho = O3IrrepsSpec("0e + 2e").representation_matrix(matrix)
    expected = rho @ operator @ rho.T
    torch.testing.assert_close(
        transformed_operator, expected, atol=8e-7, rtol=8e-7
    )


def test_mapped_heads_match_per_sample_output_and_loss_gradients():
    torch.manual_seed(3)
    kwargs = {
        "hidden_irreps": SEED,
        "output_spec": O3IrrepsSpec("0e + 2e"),
        "bottleneck_irreps": SEED,
        "pool": False,
    }
    spherical = O3QuadraticSymmetricOperatorHead(
        **kwargs, backend="spherical_cg"
    ).double()
    cartesian = O3QuadraticSymmetricOperatorHead(
        **kwargs, backend="cartesian_stf"
    ).double()
    # Old spherical checkpoints include deterministic e3nn code-generation
    # buffers.  Strict loading drops only those buffers and maps every weight.
    cartesian.load_state_dict(spherical.state_dict(), strict=True)

    spherical_input = SEED.randn(6, -1, dtype=torch.float64).requires_grad_()
    cartesian_input = spherical_input.detach().clone().requires_grad_()
    spherical_operator = spherical(spherical_input)
    cartesian_operator = cartesian(cartesian_input)
    assert torch.linalg.matrix_norm(
        spherical_operator - cartesian_operator, ord="fro", dim=(-2, -1)
    ).max().item() < 1e-6

    cotangent = torch.randn_like(spherical_operator)
    spherical_loss = (torch.matrix_exp(spherical_operator) * cotangent).sum()
    cartesian_loss = (torch.matrix_exp(cartesian_operator) * cotangent).sum()
    spherical_gradients = torch.autograd.grad(
        spherical_loss,
        (spherical_input, spherical.square.weight),
    )
    cartesian_gradients = torch.autograd.grad(
        cartesian_loss,
        (cartesian_input, cartesian.square.weight),
    )
    torch.testing.assert_close(
        cartesian_loss, spherical_loss, atol=2e-6, rtol=2e-7
    )
    for actual, expected in zip(cartesian_gradients, spherical_gradients):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


def test_compiler_selects_exact_stf_and_labels_truncation_as_approximate():
    exact = O3RepresentationCompiler(
        "0e + 2e",
        CompilerConfig(covariance="full", output_scope="dense"),
    ).compile(SEED)
    assert exact.backend == "cartesian_stf"
    assert exact.backend_exact
    assert exact.stf_contraction_rank is None
    exact_head = exact.build_head()
    assert exact_head.lifting.stages[0].tensor_product.is_exact

    truncated = O3RepresentationCompiler(
        "0e + 2e",
        CompilerConfig(
            covariance="full",
            output_scope="dense",
            backend="cartesian_stf",
            stf_contraction_rank=1,
        ),
    ).compile(SEED)
    assert truncated.backend == "cartesian_stf"
    assert not truncated.backend_exact
    assert truncated.stf_contraction_rank == 1
    assert not truncated.build_head().lifting.stages[0].tensor_product.is_exact


def test_compiler_lowers_mixed_parity_stage_without_discarding_channels():
    lowered_compilation = O3RepresentationCompiler(
        "0e + 2e",
        CompilerConfig(covariance="full", output_scope="dense"),
    ).compile(MIXED_PARITY_SEED)
    spherical_compilation = O3RepresentationCompiler(
        "0e + 2e",
        CompilerConfig(
            covariance="full", output_scope="dense", backend="spherical_cg"
        ),
    ).compile(MIXED_PARITY_SEED)
    assert lowered_compilation.backend == "cartesian_stf"
    lowered = lowered_compilation.build_head()
    spherical = spherical_compilation.build_head()
    lowered.load_state_dict(spherical.state_dict(), strict=True)
    tensor_product = lowered.lifting.stages[0].tensor_product
    assert tensor_product.irreps_in1 == MIXED_PARITY_SEED
    assert tensor_product.irreps_in2 == MIXED_PARITY_SEED
    features = 0.1 * MIXED_PARITY_SEED.randn(3, -1)
    spherical_mean, spherical_operator = spherical(features)
    lowered_mean, lowered_operator = lowered(features)
    torch.testing.assert_close(lowered_mean, spherical_mean, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        lowered_operator, spherical_operator, atol=2e-6, rtol=2e-6
    )


def test_high_order_compilation_uses_spherical_backend_and_rejects_fake_stf():
    automatic = O3RepresentationCompiler(
        rank4_elasticity_irreps(),
        CompilerConfig(covariance="full", backend="auto"),
    ).compile(SEED)
    assert automatic.backend == "spherical_cg"
    assert automatic.backend_exact
    with pytest.raises(ValueError, match="cartesian_stf requires"):
        O3RepresentationCompiler(
            rank4_elasticity_irreps(),
            CompilerConfig(covariance="full", backend="cartesian_stf"),
        ).compile(SEED)
