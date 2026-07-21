"""Unified representation/distribution/operator IR regression tests."""

import pytest
import torch

from equivcompiler import (
    AutoBudget,
    EllipticalDistribution,
    ExactExecutorCandidates,
    ExactOnly,
    ExecutorMeasurement,
    FeatureSpec,
    FirstFeasible,
    FullCovariance,
    GraphPrecision,
    IsotypicBlockCovariance,
    LowRankCovariance,
    MinimizeLatency,
    PreferExecutor,
    execution_signature_for_plan,
    RadialLaw,
    SpecificExecutor,
    TruncatedMultiplicityRank,
    plan_readout,
)
from representations import (
    CompilationCertificate,
    CompilationError,
    EquivariantOutputGraph,
    O3IrrepsSpec,
    O3ReachabilityAnalysis,
    O3ProgramCompiler,
    UnreachableActiveTargetError,
    analyze_lifting_graph,
)
from representations.operator_lowering import RecursiveOperatorMap
from equivcompiler.policies import OperatorFamilySpec
from representations.operator_ir import (
    FamilyRelation,
    OperatorFamilyPlan,
    OperatorIR,
    ParameterBinding,
)
from representations.representation_ir import IrrepsExpr, SymmetricSquareExpr, TrivialScalarsExpr


SEED = FeatureSpec.from_irreps("4x0e + 2x1o + 2x2e", scope="global")


def _graph():
    return EquivariantOutputGraph(
        num_nodes=3,
        edges=((0, 1), (1, 2)),
        node_irrep="1o",
    )


def _core_compiler_inputs(seed, output, family):
    feature = FeatureSpec.from_irreps(seed, scope="global")
    plan = plan_readout(feature, output=output, covariance=family)
    return feature, plan.compilation.executor_decision, plan.distribution_spec


def test_all_operator_families_share_one_typed_ir():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    graph = _graph()
    families = (
        FullCovariance(),
        LowRankCovariance(2),
        IsotypicBlockCovariance(),
    )
    roots = [family.compile(output).assembly.kind for family in families]
    assert roots == ["spectral_positive", "add", "direct_sum"]
    graph_plan = GraphPrecision(graph).compile(O3IrrepsSpec(graph.output_irreps))
    assert graph_plan.assembly.kind == "add"
    assert graph_plan.assembly.inputs[1].kind == "pullback"
    assert all(plan.compile(output).parameter_expression.as_dict() for plan in families)


def test_degenerate_structured_families_report_exact_full_coverage():
    scalar_output = O3IrrepsSpec("3x0e")
    assert (
        IsotypicBlockCovariance().compile(scalar_output).relation_to_full
        == FamilyRelation.EQUAL_TO_FULL
    )
    singleton_graph = EquivariantOutputGraph(num_nodes=1, edges=(), node_irrep="1o")
    assert (
        GraphPrecision(singleton_graph)
        .compile(O3IrrepsSpec(singleton_graph.output_irreps))
        .relation_to_full
        == FamilyRelation.EQUAL_TO_FULL
    )


def test_auto_budget_cost_objective_differs_from_user_priority():
    candidates = (
        FullCovariance(),
        LowRankCovariance(1),
        IsotypicBlockCovariance(),
    )
    automatic = plan_readout(
        SEED,
        output="ij=ji",
        covariance=AutoBudget(max_parameters=30, candidates=candidates),
    )
    prioritized = plan_readout(
        SEED,
        output="ij=ji",
        covariance=FirstFeasible(max_parameters=30, priority=candidates),
    )
    assert automatic.compilation.covariance_mode == "block"
    assert automatic.selection_reason["rule"] == "family_cost_model_under_budget"
    assert automatic.selection_reason["cost_objective"]["kind"] == "min_parameter_count"
    assert prioritized.compilation.covariance_mode == "full"
    assert prioritized.selection_reason["rule"] == "user_declared_priority"


def test_distribution_functor_separates_full_reference_and_active_parameters():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    distribution = EllipticalDistribution("student_t", student_t_dof=7.0)
    family = LowRankCovariance(2).compile(output)
    canonical = distribution.canonical_reference(output)
    active = distribution.active_parameter_rep(output, family)
    assert canonical.as_dict()["kind"] == "direct_sum"
    assert active.as_dict()["kind"] == "direct_sum"
    assert canonical.decompose_o3().irreps != active.decompose_o3().irreps
    assert distribution.as_dict()["proper"]


def test_restricted_family_treats_canonical_failure_as_diagnostic():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    family = IsotypicBlockCovariance().compile(output)
    compiler = O3ProgramCompiler(output)
    canonical = EllipticalDistribution().canonical_reference(output).decompose_o3().irreps
    active = family.active_expression(compiler._output_expression()).decompose_o3().irreps
    seed = output.irreps
    synthetic_failure = O3ReachabilityAnalysis(
        seed,
        canonical,
        None,
        CompilationCertificate(
            code="synthetic_canonical_unreachable",
            status="failure",
            message="test-only canonical diagnostic",
            details={"missing_irreps": ["4e"]},
        ),
    )
    feature, decision, distribution = _core_compiler_inputs(
        seed, output, IsotypicBlockCovariance()
    )
    compilation = compiler.compile(
        feature,
        operator_family=family,
        executor_decision=decision,
        distribution_spec=distribution,
        canonical_reachability=synthetic_failure,
        active_reachability=analyze_lifting_graph(seed, active),
    )
    report = compilation.report().as_dict()
    assert not report["representation_reachability"]["canonical"]["reachable"]
    assert report["representation_reachability"]["active"]["reachable"]
    assert report["complexity"]["canonical_lifting_edges"] is None


def test_alternative_full_family_parameterization_keeps_canonical_diagnostic():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    family = LowRankCovariance(output.dim).compile(output)
    assert family.relation_to_full == FamilyRelation.EQUAL_TO_FULL
    compiler = O3ProgramCompiler(output)
    canonical = EllipticalDistribution().canonical_reference(output).decompose_o3().irreps
    active = family.active_expression(compiler._output_expression()).decompose_o3().irreps
    seed = output.irreps
    synthetic_failure = O3ReachabilityAnalysis(
        seed,
        canonical,
        None,
        CompilationCertificate(
            code="synthetic_canonical_unreachable",
            status="failure",
            message="test-only canonical diagnostic",
            details={"missing_irreps": ["4e"]},
        ),
    )
    feature, decision, distribution = _core_compiler_inputs(
        seed, output, LowRankCovariance(output.dim)
    )
    compilation = compiler.compile(
        feature,
        operator_family=family,
        executor_decision=decision,
        distribution_spec=distribution,
        canonical_reachability=synthetic_failure,
        active_reachability=analyze_lifting_graph(seed, active),
    )
    reachability = compilation.report().as_dict()["representation_reachability"]
    assert reachability["canonical_is_diagnostic"]
    assert not reachability["canonical_is_compilation_gate"]


def test_full_family_still_rejects_the_same_active_failure():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    family = FullCovariance().compile(output)
    compiler = O3ProgramCompiler(output)
    target = family.active_expression(compiler._output_expression()).decompose_o3().irreps
    failure = O3ReachabilityAnalysis(
        output.irreps,
        target,
        None,
        CompilationCertificate(
            code="synthetic_active_unreachable",
            status="failure",
            message="test-only active failure",
            details={"missing_irreps": ["4e"]},
        ),
    )
    with pytest.raises(UnreachableActiveTargetError):
        compiler.compile(
            FeatureSpec.from_irreps(output.irreps, scope="global"),
            operator_family=family,
            executor_decision=None,
            distribution_spec=EllipticalDistribution(),
            canonical_reachability=failure,
            active_reachability=failure,
        )


def test_fidelity_executor_and_measured_cost_are_independent():
    draft = plan_readout(
        SEED,
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=ExactOnly(),
        executor=ExactExecutorCandidates(),
    )
    signature = execution_signature_for_plan(
        draft,
        batch_shape=(16, SEED.irreps.dim),
        dtype="float32",
        device="cpu",
    )
    cost = MinimizeLatency(
        signature,
        (
            ExecutorMeasurement("spherical_cg", signature, 2.0),
            ExecutorMeasurement("cartesian_stf", signature, 1.0),
        ),
    )
    exact = plan_readout(
        SEED,
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=ExactOnly(),
        executor=ExactExecutorCandidates(),
        cost=cost,
    )
    assert exact.compilation.backend == "cartesian_stf"
    assert exact.report.backend_selection_basis["method"] == "measured_autotune"
    assert exact.report.execution_fidelity["exactness"] == "exact_for_active_family"

    approximate = plan_readout(
        SEED,
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=TruncatedMultiplicityRank(1),
        executor=SpecificExecutor("cartesian_stf"),
        cost=PreferExecutor(("cartesian_stf",)),
    )
    assert approximate.report.execution_fidelity["exactness"] == "approximate_for_active_family"


def test_nonorthogonal_coordinates_require_a_gram_identity():
    with pytest.raises(ValueError, match="gram_matrix_id"):
        FeatureSpec.from_irreps(
            "1x0e", metric_kind="nonorthogonal_invariant_metric"
        )


@pytest.mark.parametrize(
    "family",
    (FullCovariance(), LowRankCovariance(2), IsotypicBlockCovariance()),
)
def test_runtime_plugins_preserve_public_projection_checkpoint_names(family):
    plan = plan_readout(SEED, output="ij=ji", covariance=family)
    keys = set(plan.compilation.build_head().state_dict())
    assert any(key.startswith("covariance_projection.") for key in keys)


class _JitteredSpectralFamily(OperatorFamilySpec):
    """Test-only fifth family assembled solely from registered primitives."""

    def compile(self, output: O3IrrepsSpec) -> OperatorFamilyPlan:
        base = IrrepsExpr(output.irreps, "output")
        operator = SymmetricSquareExpr(base)
        jitter = TrivialScalarsExpr(1)
        spectral = OperatorIR.spectral_positive(
            OperatorIR.symmetric_operator(
                parameter=OperatorIR.parameter("operator"),
                coordinate_space="output_representation",
                output_irreps=str(output.irreps),
            ),
            map="matrix_exponential",
        )
        isotropic = OperatorIR.positive_scalar_identity(
            OperatorIR.parameter("jitter"), dimension=output.dim
        )
        return OperatorFamilyPlan(
            kind="test_jittered_spectral",
            parameter_bindings=(
                ParameterBinding("operator", operator, "covariance_projection"),
                ParameterBinding("jitter", jitter, "jitter_projection"),
            ),
            parameter_count=operator.decompose_o3().irreps.dim + 1,
            domain="scatter",
            assembly=OperatorIR.add(spectral, isotropic),
            relation_to_full=FamilyRelation.EQUAL_TO_FULL,
        )

    def as_dict(self):
        return {"kind": "test_jittered_spectral"}


def test_new_composed_family_needs_no_compiler_changes():
    plan = plan_readout(SEED, output="ij=ji", covariance=_JitteredSpectralFamily())
    readout = plan.build_readout()
    result = readout(
        SEED.irreps.randn(3, -1),
        torch.tensor([0, 0, 0]),
        target=torch.randn(3, 6),
        return_scale=True,
    )
    assert readout.spd_map.__class__.__name__ == "RecursiveOperatorMap"
    assert result["params"].shape == (3, 22)
    assert torch.linalg.eigvalsh(result["scale"]).min() > 0
    samples = readout.spd_map.sample(result["mu"], result["params"], 4)
    assert samples.shape == (3, 6, 4)
    assert torch.isfinite(result["loss"])


def test_unknown_operator_node_cannot_claim_spd_or_equivariance():
    with pytest.raises(ValueError, match="verifier-derived"):
        OperatorIR.node("unknown", positivity="spd", equivariance="certified")
    certificate = OperatorIR.node("unknown").verify()
    assert not certificate.valid
    assert certificate.positivity.value == "unknown"
    assert certificate.equivariance.value == "unknown"


def test_low_rank_request_is_never_silently_clamped():
    family = LowRankCovariance(10).compile(O3IrrepsSpec.from_cartesian("ij=ji"))
    assert family.rank == 10
    assert family.parameter_count == 61
    assert family.relation_to_full == FamilyRelation.EQUAL_TO_FULL


def test_non_truncating_rank_preserves_request_and_reports_effective_exactness():
    plan = plan_readout(
        SEED,
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=TruncatedMultiplicityRank(64),
        executor=SpecificExecutor("cartesian_stf"),
        cost=PreferExecutor(("cartesian_stf",)),
    )
    decision = plan.compilation.executor_decision.capability.fidelity
    assert decision.requested_contraction_rank == 64
    assert decision.effective_contraction_rank is None
    assert decision.effective == "exact"
    assert "requested_cap_covers" in decision.normalization_reason


def test_same_dimension_different_irreps_do_not_share_tuning_signature():
    left = FeatureSpec.from_irreps("24x0e + 24x1o", scope="global")
    right = FeatureSpec.from_irreps("16x0e + 16x2e", scope="global")
    assert left.irreps.dim == right.irreps.dim == 96
    left_plan = plan_readout(left, output="ij=ji", covariance=FullCovariance())
    right_plan = plan_readout(right, output="ij=ji", covariance=FullCovariance())
    left_signature = execution_signature_for_plan(
        left_plan, batch_shape=(16, 96), dtype="float32", device="cpu"
    )
    right_signature = execution_signature_for_plan(
        right_plan, batch_shape=(16, 96), dtype="float32", device="cpu"
    )
    assert left_signature != right_signature
    assert left_signature.feature_fingerprint != right_signature.feature_fingerprint


def test_core_rejects_executor_decision_for_a_different_feature_contract():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    source = plan_readout(SEED, output=output, covariance=FullCovariance())
    different_feature = FeatureSpec.from_irreps(SEED.irreps, scope="node")
    family = FullCovariance().compile(output)
    with pytest.raises(CompilationError) as caught:
        O3ProgramCompiler(output).compile(
            different_feature,
            operator_family=family,
            executor_decision=source.compilation.executor_decision,
            distribution_spec=source.distribution_spec,
        )
    assert caught.value.certificate.code == "executor_decision_contract_mismatch"
    assert "feature_fingerprint" in caught.value.certificate.details["mismatches"]


def test_measured_cost_rejects_a_different_semantic_plan_signature():
    measured_plan = plan_readout(
        SEED, output="ij=ji", covariance=FullCovariance()
    )
    signature = execution_signature_for_plan(
        measured_plan,
        batch_shape=(8, SEED.irreps.dim),
        dtype="float32",
        device="cpu",
    )
    cost = MinimizeLatency(
        signature,
        (ExecutorMeasurement("spherical_cg", signature, 1.0),),
    )
    with pytest.raises(CompilationError) as caught:
        plan_readout(
            SEED,
            output="1o",
            covariance=FullCovariance(),
            cost=cost,
        )
    assert caught.value.certificate.code == "executor_measurement_signature_mismatch"
    assert "semantic_plan_hash" in caught.value.certificate.details["mismatches"]


class _NamedGaussianRadial(RadialLaw):
    @property
    def name(self) -> str:
        return "test_named_gaussian"

    def materialize_log_prob(self) -> torch.nn.Module:
        from distributions import GaussianNLL

        return GaussianNLL()

    def calibration_reference(self):
        return {"residual_statistic": "squared_mahalanobis", "reference": "chi_square"}

    def as_dict(self):
        return {"kind": "test_named_gaussian_radial"}


def test_third_radial_law_materializes_without_compiler_dispatch():
    distribution = EllipticalDistribution(_NamedGaussianRadial())
    plan = plan_readout(
        SEED,
        output="ij=ji",
        covariance=FullCovariance(),
        distribution=distribution,
    )
    assert plan.compilation.build_distribution().__class__.__name__ == "GaussianNLL"
    assert plan.report.objective["name"] == "test_named_gaussian"


@pytest.mark.parametrize(
    "output,family",
    (
        ("ij=ji", FullCovariance()),
        ("ij=ji", LowRankCovariance(2)),
        ("ij=ji", IsotypicBlockCovariance()),
        (_graph().output_irreps, GraphPrecision(_graph())),
    ),
)
def test_pattern_optimized_maps_match_recursive_ir_interpreter(output, family):
    torch.manual_seed(11)
    plan = plan_readout(SEED, output=output, covariance=family)
    optimized = plan.compilation.build_spd_map().double()
    recursive = RecursiveOperatorMap(plan.compilation).double()
    left = torch.randn(
        2, plan.compilation.covariance_parameter_count, dtype=torch.float64
    ).requires_grad_()
    right = left.detach().clone().requires_grad_()
    residual = torch.randn(2, plan.compilation.output_spec.dim, dtype=torch.float64)

    optimized_covariance = optimized(left)
    recursive_covariance = recursive(right)
    optimized_stats = optimized.statistics(left, residual)
    recursive_stats = recursive.statistics(right, residual)
    torch.testing.assert_close(
        optimized_covariance, recursive_covariance, atol=2e-9, rtol=2e-9
    )
    for actual, expected in zip(optimized_stats, recursive_stats):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)

    optimized_loss = optimized_covariance.square().mean() + sum(
        value.mean() for value in optimized_stats
    )
    recursive_loss = recursive_covariance.square().mean() + sum(
        value.mean() for value in recursive_stats
    )
    optimized_gradient = torch.autograd.grad(optimized_loss, left)[0]
    recursive_gradient = torch.autograd.grad(recursive_loss, right)[0]
    torch.testing.assert_close(
        optimized_gradient, recursive_gradient, atol=2e-8, rtol=2e-8
    )
