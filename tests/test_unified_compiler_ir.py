"""Unified representation/distribution/operator IR regression tests."""

import pytest

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
    ShapeSignature,
    SpecificExecutor,
    TruncatedMultiplicityRank,
    plan_readout,
)
from representations import (
    CompilationCertificate,
    EquivariantOutputGraph,
    O3IrrepsSpec,
    O3ReachabilityAnalysis,
    O3RepresentationCompiler,
    UnreachableActiveTargetError,
    analyze_lifting_graph,
)
from representations.operator_ir import FamilyRelation


SEED = FeatureSpec.from_irreps("4x0e + 2x1o + 2x2e", scope="global")


def _graph():
    return EquivariantOutputGraph(
        num_nodes=3,
        edges=((0, 1), (1, 2)),
        node_irrep="1o",
    )


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
    assert automatic.selection_reason["rule"] == "min_parameter_count_under_budget"
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
    compiler = O3RepresentationCompiler(output)
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
    compilation = compiler.compile(
        seed,
        operator_family=family,
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
    compiler = O3RepresentationCompiler(output)
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
    compilation = compiler.compile(
        seed,
        operator_family=family,
        canonical_reachability=synthetic_failure,
        active_reachability=analyze_lifting_graph(seed, active),
    )
    reachability = compilation.report().as_dict()["representation_reachability"]
    assert reachability["canonical_is_diagnostic"]
    assert not reachability["canonical_is_compilation_gate"]


def test_full_family_still_rejects_the_same_active_failure():
    output = O3IrrepsSpec.from_cartesian("ij=ji")
    family = FullCovariance().compile(output)
    compiler = O3RepresentationCompiler(output)
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
            output.irreps,
            operator_family=family,
            canonical_reachability=failure,
            active_reachability=failure,
        )


def test_fidelity_executor_and_measured_cost_are_independent():
    signature = ShapeSignature(16, SEED.irreps.dim, "float32", "cuda")
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
