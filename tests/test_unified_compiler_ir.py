"""Unified representation/distribution/operator IR regression tests."""

from dataclasses import dataclass

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
    SpectralWindowCovariance,
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
from representations.representation_ir import (
    IrrepsExpr,
    RepeatedExpr,
    SymmetricSquareExpr,
    TrivialScalarsExpr,
)


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


def test_spectral_window_is_compiled_as_a_bounded_full_covariance_family():
    plan = plan_readout(
        SEED,
        output="ij=ji",
        covariance=SpectralWindowCovariance(-3.0, 2.0),
    )
    compilation = plan.compilation
    assert compilation.operator_family.relation_to_full == FamilyRelation.STRICT_SUBSET
    assert compilation.operator_family.assembly.attribute_dict() == {
        "log_variance_max": 2.0,
        "log_variance_min": -3.0,
        "map": "spectral_window",
    }
    spd_map = compilation.build_spd_map()
    assert spd_map.optimization_name == "spectral_window_eigendecomposition_oracle"
    parameters = torch.randn(3, compilation.covariance_parameter_count)
    scale = spd_map(parameters)
    log_spectrum = torch.log(torch.linalg.eigvalsh(scale))
    assert log_spectrum.min() >= -3.0 - 2e-6
    assert log_spectrum.max() <= 2.0 + 2e-6


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
    canonical = (
        EllipticalDistribution().canonical_reference(output).decompose_o3().irreps
    )
    active = (
        family.active_expression(compiler._output_expression()).decompose_o3().irreps
    )
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
    canonical = (
        EllipticalDistribution().canonical_reference(output).decompose_o3().irreps
    )
    active = (
        family.active_expression(compiler._output_expression()).decompose_o3().irreps
    )
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
    target = (
        family.active_expression(compiler._output_expression()).decompose_o3().irreps
    )
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
    assert (
        approximate.report.execution_fidelity["exactness"]
        == "approximate_for_active_family"
    )


def test_nonorthogonal_coordinates_require_a_gram_identity():
    with pytest.raises(ValueError, match="gram_matrix_id"):
        FeatureSpec.from_irreps("1x0e", metric_kind="nonorthogonal_invariant_metric")


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
            output_irreps=output.irreps,
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


def test_typed_leaf_is_not_a_semantic_operator_certificate():
    certificate = OperatorIR.parameter("missing").verify()
    assert not certificate.valid
    assert certificate.positivity.value == "unknown"
    assert certificate.equivariance.value == "unknown"
    assert certificate.positivity.value == "unknown"
    assert certificate.equivariance.value == "unknown"


def _full_family_with(
    output: O3IrrepsSpec,
    *,
    expression,
    parameter: OperatorIR,
    spectral_map: str = "matrix_exponential",
) -> OperatorFamilyPlan:
    assembly = OperatorIR.spectral_positive(
        OperatorIR.symmetric_operator(
            parameter=parameter,
            coordinate_space="output_representation",
            output_irreps=str(output.irreps),
        ),
        map=spectral_map,
    )
    return OperatorFamilyPlan(
        kind="test_full",
        output_irreps=output.irreps,
        parameter_bindings=(ParameterBinding("operator", expression, "projection"),),
        parameter_count=expression.decompose_o3().irreps.dim,
        domain="scatter",
        assembly=assembly,
        relation_to_full=FamilyRelation.UNKNOWN,
    )


def test_typed_verifier_rejects_unlowered_spectral_primitive():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    expression = SymmetricSquareExpr(IrrepsExpr(output.irreps, "output"))
    with pytest.raises(ValueError, match="unregistered positive spectral map"):
        _full_family_with(
            output,
            expression=expression,
            parameter=OperatorIR.parameter("operator"),
            spectral_map="multiplicity_cholesky",
        )


def test_typed_verifier_rejects_binding_that_is_not_symmetric_square():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    wrong = TrivialScalarsExpr(output.dim * (output.dim + 1) // 2)
    with pytest.raises(ValueError, match=r"must be Sym\^2\(V\)"):
        _full_family_with(
            output,
            expression=wrong,
            parameter=OperatorIR.parameter("operator"),
        )


def test_typed_verifier_rejects_parameter_slice_outside_binding():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    expression = SymmetricSquareExpr(IrrepsExpr(output.irreps, "output"))
    with pytest.raises(ValueError, match="outside transformed dimension"):
        _full_family_with(
            output,
            expression=expression,
            parameter=OperatorIR.parameter(
                "operator", stop=expression.decompose_o3().irreps.dim + 1
            ),
        )


def test_typed_verifier_checks_factor_and_scalar_binding_types():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    factor = OperatorIR.gram(
        OperatorIR.equivariant_factor(
            OperatorIR.parameter("factor"),
            rank=2,
            output_irreps=str(output.irreps),
        )
    )
    isotropic = OperatorIR.positive_scalar_identity(
        OperatorIR.parameter("scale"), dimension=output.dim, minimum=0.0
    )
    with pytest.raises(ValueError, match="r copies of V"):
        OperatorFamilyPlan(
            kind="bad_factor",
            output_irreps=output.irreps,
            parameter_bindings=(
                ParameterBinding(
                    "factor", TrivialScalarsExpr(output.dim * 2), "factor_projection"
                ),
                ParameterBinding("scale", TrivialScalarsExpr(1), "scale_projection"),
            ),
            parameter_count=output.dim * 2 + 1,
            domain="scatter",
            assembly=OperatorIR.add(isotropic, factor),
            relation_to_full=FamilyRelation.UNKNOWN,
            rank=2,
        )

    pseudoscalar = IrrepsExpr("1x0o", "nontrivial_scale")
    with pytest.raises(ValueError, match="exactly one trivial scalar"):
        OperatorFamilyPlan(
            kind="bad_scale",
            output_irreps=output.irreps,
            parameter_bindings=(ParameterBinding("scale", pseudoscalar, "projection"),),
            parameter_count=1,
            domain="scatter",
            assembly=isotropic,
            relation_to_full=FamilyRelation.UNKNOWN,
        )


def test_typed_verifier_checks_cholesky_scalar_count():
    output = O3IrrepsSpec("2x0e")
    with pytest.raises(ValueError, match=r"requires 3 trivial scalars"):
        OperatorFamilyPlan(
            kind="bad_cholesky",
            output_irreps=output.irreps,
            parameter_bindings=(
                ParameterBinding("block", TrivialScalarsExpr(2), "projection"),
            ),
            parameter_count=2,
            domain="scatter",
            assembly=OperatorIR.cholesky_positive(
                OperatorIR.parameter("block"), dimension=2
            ),
            relation_to_full=FamilyRelation.UNKNOWN,
        )


def _rewrite_parameter_layout(node: OperatorIR) -> OperatorIR:
    attributes = node.attribute_dict()
    if (
        node.kind == "parameter"
        and attributes.get("coordinate_layout") == "repeated_irrep"
    ):
        attributes["copies"] = int(attributes["copies"]) + 1
    return OperatorIR.node(
        node.kind,
        *(_rewrite_parameter_layout(child) for child in node.inputs),
        **attributes,
    )


def test_typed_verifier_checks_graph_repeated_layout_against_binding():
    graph = _graph()
    valid = GraphPrecision(graph).compile(O3IrrepsSpec(graph.output_irreps))
    with pytest.raises(ValueError, match="layout does not match the binding"):
        OperatorFamilyPlan(
            kind="bad_graph_layout",
            output_irreps=valid.output_irreps,
            parameter_bindings=valid.parameter_bindings,
            parameter_count=valid.parameter_count,
            domain=valid.domain,
            assembly=_rewrite_parameter_layout(valid.assembly),
            relation_to_full=valid.relation_to_full,
            graph=graph,
        )


def test_low_rank_request_is_never_silently_clamped():
    family = LowRankCovariance(10).compile(O3IrrepsSpec.from_cartesian("ij=ji"))
    assert family.rank == 10
    assert family.parameter_count == 61
    assert family.relation_to_full == FamilyRelation.EQUAL_TO_FULL
    restricted = LowRankCovariance(5).compile(O3IrrepsSpec.from_cartesian("ij=ji"))
    assert restricted.relation_to_full == FamilyRelation.STRICT_SUBSET


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


def test_lifting_backend_is_part_of_tuning_signature():
    e3nn_plan = plan_readout(SEED, output="ij=ji", covariance=FullCovariance())
    cueq_plan = plan_readout(
        SEED,
        output="ij=ji",
        covariance=FullCovariance(),
        lifting_backend="cueq",
    )
    e3nn_signature = execution_signature_for_plan(
        e3nn_plan, batch_shape=(16, 96), dtype="float32", device="cpu"
    )
    cueq_signature = execution_signature_for_plan(
        cueq_plan, batch_shape=(16, 96), dtype="float32", device="cpu"
    )
    assert e3nn_signature.semantic_plan_hash != cueq_signature.semantic_plan_hash


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
    measured_plan = plan_readout(SEED, output="ij=ji", covariance=FullCovariance())
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


@dataclass(frozen=True)
class _AugmentedLowRankFamily(OperatorFamilySpec):
    rank: int = 2

    def compile(self, output: O3IrrepsSpec) -> OperatorFamilyPlan:
        base = LowRankCovariance(self.rank).compile(output)
        extra = TrivialScalarsExpr(1)
        return OperatorFamilyPlan(
            kind="augmented_low_rank",
            output_irreps=output.irreps,
            parameter_bindings=base.parameter_bindings
            + (ParameterBinding("extra", extra, "extra_projection"),),
            parameter_count=base.parameter_count + 1,
            domain="scatter",
            assembly=OperatorIR.add(
                *base.assembly.inputs,
                OperatorIR.positive_scalar_identity(
                    OperatorIR.parameter("extra"),
                    dimension=output.dim,
                    minimum=0.0,
                ),
            ),
            relation_to_full=FamilyRelation.UNKNOWN,
            rank=self.rank,
        )

    def as_dict(self):
        return {"kind": "augmented_low_rank", "rank": self.rank}


@dataclass(frozen=True)
class _RenamedLowRankFamily(OperatorFamilySpec):
    rank: int = 2

    def compile(self, output: O3IrrepsSpec) -> OperatorFamilyPlan:
        output_expression = IrrepsExpr(output.irreps, "output")
        factor_expression = RepeatedExpr(output_expression, self.rank)
        scale_expression = TrivialScalarsExpr(1)
        assembly = OperatorIR.add(
            OperatorIR.positive_scalar_identity(
                OperatorIR.parameter("noise"), dimension=output.dim, minimum=0.0
            ),
            OperatorIR.gram(
                OperatorIR.equivariant_factor(
                    OperatorIR.parameter("latent"),
                    rank=self.rank,
                    output_irreps=str(output.irreps),
                )
            ),
        )
        return OperatorFamilyPlan(
            kind="renamed_low_rank",
            output_irreps=output.irreps,
            parameter_bindings=(
                ParameterBinding("latent", factor_expression, "latent_projection"),
                ParameterBinding("noise", scale_expression, "noise_projection"),
            ),
            parameter_count=output.dim * self.rank + 1,
            domain="scatter",
            assembly=assembly,
            relation_to_full=FamilyRelation.UNKNOWN,
            rank=self.rank,
        )

    def as_dict(self):
        return {"kind": "renamed_low_rank", "rank": self.rank}


@dataclass(frozen=True)
class _AugmentedGraphFamily(OperatorFamilySpec):
    graph: EquivariantOutputGraph

    def compile(self, output: O3IrrepsSpec) -> OperatorFamilyPlan:
        base = GraphPrecision(self.graph).compile(output)
        extra = TrivialScalarsExpr(1)
        return OperatorFamilyPlan(
            kind="augmented_graph",
            output_irreps=output.irreps,
            parameter_bindings=base.parameter_bindings
            + (ParameterBinding("extra", extra, "extra_projection"),),
            parameter_count=base.parameter_count + 1,
            domain="precision",
            assembly=OperatorIR.add(
                base.assembly,
                OperatorIR.positive_scalar_identity(
                    OperatorIR.parameter("extra"),
                    dimension=output.dim,
                    minimum=0.0,
                ),
            ),
            relation_to_full=FamilyRelation.UNKNOWN,
            graph=self.graph,
        )

    def as_dict(self):
        return {"kind": "augmented_graph", "graph": self.graph.as_dict()}


@pytest.mark.parametrize(
    "output,family",
    (
        ("ij=ji", _AugmentedLowRankFamily()),
        ("ij=ji", _RenamedLowRankFamily()),
        (_graph().output_irreps, _AugmentedGraphFamily(_graph())),
    ),
)
def test_near_miss_programs_fall_back_to_recursive_interpreter(output, family):
    plan = plan_readout(SEED, output=output, covariance=family)
    lowered = plan.compilation.build_spd_map()
    assert isinstance(lowered, RecursiveOperatorMap)
    optimization = plan.report.family["optimization"]
    assert optimization == {
        "optimization_name": None,
        "fallback": "generic_recursive_interpreter",
        "reason": "no_exact_registered_template_match",
    }


def test_optimization_certificate_is_exposed_in_report():
    plan = plan_readout(SEED, output="ij=ji", covariance=LowRankCovariance(2))
    lowered = plan.compilation.build_spd_map()
    certificate = lowered.optimization_certificate
    report_certificate = plan.report.family["optimization"]
    assert report_certificate == certificate.as_dict()
    assert certificate.semantic_template_hash
    assert (
        certificate.operator_program_hash
        == plan.compilation.operator_family.assembly.fingerprint
    )
    assert certificate.binding_correspondence == (
        ("factor", "factor"),
        ("scale", "scale"),
    )
    assert certificate.rank == 2
