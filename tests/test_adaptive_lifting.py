"""Tests for the representation compiler and target-directed CG lifting."""

import pytest
import torch
from e3nn import o3

from equivcompiler import (
    FeatureSpec,
    FirstFeasible,
    FullCovariance,
    IsotypicBlockCovariance,
    LowRankCovariance,
    plan_readout,
)

from representations import (
    O3AdaptiveLifting,
    O3IrrepsSpec,
    coverage_deficit,
    direct_sum_irreps,
    plan_lifting_graph,
    rank4_elasticity_irreps,
    required_lifting_depth,
)
from spd_maps import IsotypicBlockMap


SEED = o3.Irreps("4x0e + 2x1o + 2x2e")


def _compile(output, covariance=FullCovariance(), *, output_scope="global"):
    seed = FeatureSpec.from_irreps(SEED, scope="node")
    return plan_readout(
        seed,
        output=output,
        covariance=covariance,
        output_scope=output_scope,
    ).compilation


def test_canonical_target_is_v_plus_symmetric_square():
    output = O3IrrepsSpec("0e + 2e")
    compilation = _compile(output, output_scope="node")
    expected = direct_sum_irreps(
        output.irreps, output.symmetric_square().operator_irreps
    )
    assert compilation.canonical_target_irreps == expected
    assert str(expected) == "3x0e+3x2e+1x4e"


def test_rank2_needs_one_quadratic_edge_not_two():
    target = _compile("0e + 2e").canonical_target_irreps
    assert required_lifting_depth(SEED, target) == 1


def test_rank4_full_covariance_needs_three_edges_from_lmax2():
    compilation = _compile(rank4_elasticity_irreps())
    assert compilation.canonical_plan.depth == 3
    assert max(ir.l for _, ir in compilation.covariance_irreps) == 8
    assert coverage_deficit(
        compilation.canonical_plan.irreps_out,
        compilation.canonical_target_irreps,
    ) == {}


def test_plan_tracks_parity_and_rejects_unreachable_target():
    plan = plan_lifting_graph("1o", "0e + 1o + 2e")
    assert plan.depth == 1
    assert plan.paths["0e"] == ("1o", "0e")
    with pytest.raises(ValueError, match="parity"):
        plan_lifting_graph("0e + 2e", "1o")


def test_target_multiplicity_is_materialized_exactly():
    target = o3.Irreps("7x0e + 5x2e + 2x4e")
    plan = plan_lifting_graph(SEED, target)
    assert plan.irreps_out == target
    assert coverage_deficit(plan.irreps_out, target) == {}


def test_lifting_forward_backward_and_equivariance():
    torch.manual_seed(0)
    target = o3.Irreps("3x0e + 3x2e + 1x4e")
    lifting = O3AdaptiveLifting(SEED, target)
    features = SEED.randn(2, -1).requires_grad_()
    rotation = o3.rand_matrix()
    transformed = features @ SEED.D_from_matrix(rotation).T

    output = lifting(features)
    transformed_output = lifting(transformed)
    expected = output @ target.D_from_matrix(rotation).T
    relative_error = (transformed_output - expected).norm() / output.norm().clamp_min(1e-12)
    assert relative_error.item() < 2e-4

    output.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_cartesian_symmetry_is_compiled_and_round_trips():
    spec = O3IrrepsSpec.from_cartesian("ij=ji")
    assert str(spec.irreps) == "1x0e+1x2e"
    tensor = torch.randn(3, 3)
    tensor = 0.5 * (tensor + tensor.T)
    recovered = spec.to_cartesian_tensor(spec.from_cartesian_tensor(tensor))
    assert torch.allclose(recovered, tensor, atol=1e-5)


def test_auto_complexity_selects_full_then_low_rank():
    rank2_candidates = (
        FullCovariance(),
        LowRankCovariance(8),
        IsotypicBlockCovariance(),
    )
    rank2 = _compile(
        "0e + 2e", FirstFeasible(192, rank2_candidates)
    )
    elasticity = _compile(
        rank4_elasticity_irreps(),
        FirstFeasible(192, rank2_candidates),
    )
    assert rank2.covariance_mode == "full"
    assert rank2.covariance_parameter_count == 21
    assert elasticity.covariance_mode == "low_rank"
    assert elasticity.covariance_parameter_count == 21 * 8 + 1
    assert elasticity.active_plan.depth == 1


@pytest.mark.parametrize("scope,batch", [("dense", None), ("global", torch.tensor([0, 0, 1]))])
def test_compiled_head_supports_dense_and_global_output(scope, batch):
    output_scope = "node" if scope == "dense" else "global"
    compilation = _compile("0e + 2e", output_scope=output_scope)
    head = compilation.build_head()
    count = 2 if scope == "dense" else 3
    mean, parameters = head(SEED.randn(count, -1), batch)
    expected_count = count if scope == "dense" else 2
    assert mean.shape == (expected_count, 6)
    assert parameters.shape == (expected_count, 21)


def test_block_mode_uses_full_multiplicity_spd_blocks():
    compilation = _compile(
        "2x0e + 2x2e + 4e",
        IsotypicBlockCovariance(),
        output_scope="node",
    )
    spd_map = compilation.build_spd_map()
    assert spd_map.optimization_name == "multiplicity_block_oracle"
    assert isinstance(spd_map.delegate, IsotypicBlockMap)
    assert compilation.covariance_parameter_count == 7
    params = torch.randn(2, 7, requires_grad=True)
    covariance = spd_map(params)
    assert torch.linalg.eigvalsh(covariance).min().item() > 0
    loss = spd_map.logdet(params).mean()
    loss.backward()
    assert torch.isfinite(params.grad).all()


def test_compiled_rank4_low_rank_scale_is_equivariant():
    torch.manual_seed(3)
    compilation = _compile(
        rank4_elasticity_irreps(),
        LowRankCovariance(4),
        output_scope="node",
    )
    head = compilation.build_head()
    spd_map = compilation.build_spd_map()
    features = SEED.randn(1, -1)
    rotation = o3.rand_matrix()
    transformed = features @ SEED.D_from_matrix(rotation).T

    _, parameters = head(features)
    _, transformed_parameters = head(transformed)
    scale = spd_map(parameters)
    transformed_scale = spd_map(transformed_parameters)
    rho = compilation.output_spec.representation_matrix(rotation)
    expected = rho @ scale @ rho.T
    relative_error = (transformed_scale - expected).norm() / scale.norm().clamp_min(1e-12)
    assert relative_error.item() < 2e-4


def test_compiler_only_exposes_proper_objectives():
    student = plan_readout(
        FeatureSpec.from_irreps(SEED, scope="node"),
        output="1o",
        covariance=FullCovariance(),
        distribution="student_t",
        student_t_dof=7.0,
    ).compilation
    assert student.build_distribution().nu == 7.0
    with pytest.raises(ValueError, match="unsupported elliptical radial law"):
        plan_readout(
            FeatureSpec.from_irreps(SEED, scope="node"),
            output="1o",
            covariance=FullCovariance(),
            distribution="surrogate",
        )
