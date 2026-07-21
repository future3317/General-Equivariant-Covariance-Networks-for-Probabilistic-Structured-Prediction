"""Public staged compiler API, report, and safeguard regression tests."""

import pytest
import torch

from equivcompiler import (
    AutoBudget,
    ExactOnly,
    FeatureSpec,
    FullCovariance,
    GraphPrecision,
    LowRankCovariance,
    TruncatedMultiplicityRank,
    compile_readout,
    describe_output,
    plan_readout,
)
from representations import CompilationError, EquivariantOutputGraph


SEED = "4x0e + 2x1o + 2x2e"


def _certificate_codes(report):
    return {certificate["code"] for certificate in report["certificates"]}


def test_describe_output_is_semantic_and_explicitly_non_executable():
    semantics = describe_output("ijkl=jikl=ijlk=klij")
    assert semantics.output_representation == "2x0e+2x2e+1x4e"
    assert semantics.covariance_representation.endswith("1x8e")
    assert semantics.output_dimension == 21
    assert semantics.full_covariance_parameters == 231
    assert semantics.highest_covariance_angular_momentum == 8
    assert not semantics.executable
    assert semantics.reachability == "unknown_without_seed"


def test_declarative_readout_api_builds_executable_and_full_report():
    seed = FeatureSpec.from_irreps(SEED, scope="global")
    readout, report = compile_readout(
        seed,
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=ExactOnly(),
        distribution="student_t",
    )
    record = report.as_dict()
    assert record["output"]["output_representation"] == "1x0e+1x2e"
    assert record["covariance_representation"]["irreps"] == "2x0e+2x2e+1x4e"
    assert record["representation_reachability"]["canonical"]["reachable"]
    assert record["representation_reachability"]["active"]["reachable"]
    assert record["representation_reachability"]["multiplicity_coverage"] == {
        "canonical_deficit": {},
        "active_deficit": {},
    }
    assert record["family"]["relation_to_canonical"] == "canonical_full"
    assert record["execution_fidelity"]["exactness"] == "exact_for_active_family"
    assert record["probability"]["scale_semantics"] == "scatter"
    assert record["objective"]["proper"]
    assert record["complexity"]["parameter_counts"]["readout_trainable"] == sum(
        parameter.numel() for parameter in readout.parameters()
    )
    assert len(report.compatibility_hash) == 64

    features = torch.randn(3, readout.irreps_in.dim)
    target = torch.randn(3, readout.irreps_out.dim)
    result = readout(features, target=target, return_scale=True)
    assert result["mu"].shape == (3, 6)
    assert result["scale"].shape == (3, 6, 6)
    assert torch.isfinite(result["loss"])


def test_planning_is_separate_from_materialization():
    seed = FeatureSpec.from_irreps(SEED, scope="global")
    plan = plan_readout(
        seed,
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=ExactOnly(),
    )
    assert not isinstance(plan, torch.nn.Module)
    assert plan.report.compatibility_hash == plan.compatibility_hash
    readout = plan.build_readout(dtype=torch.float64)
    assert next(readout.parameters()).dtype == torch.float64


def test_budget_subfamily_is_not_reported_as_lowering_approximation():
    seed = FeatureSpec.from_irreps(SEED, scope="global")
    _, report = compile_readout(
        seed,
        output="ijkl=jikl=ijlk=klij",
        covariance=AutoBudget(
            max_parameters=192,
            candidates=(FullCovariance(), LowRankCovariance(8)),
        ),
        output_scope="global",
    )
    assert report.family["kind"] == "low_rank_plus_isotropic"
    assert report.family["relation_to_canonical"] == "strict_subfamily"
    assert report.execution_fidelity["exactness"] == "exact_for_active_family"
    assert report.execution_fidelity["approximation"] is None
    assert report.family["selection_reason"]["rule"] == "family_cost_model_under_budget"
    assert "structured_covariance_restriction" in _certificate_codes(report)
    restriction = next(
        certificate
        for certificate in report["certificates"]
        if certificate["code"] == "structured_covariance_restriction"
    )
    assert restriction["details"]["selected_by"] == "family_cost_model_under_budget"
    counts = report.complexity["parameter_counts"]
    assert counts["canonical_covariance_coordinates"] == 231
    assert counts["active_covariance_coordinates"] == 169


def test_parity_failure_is_machine_readable():
    with pytest.raises(CompilationError) as caught:
        plan_readout(
            FeatureSpec.from_irreps("2x0e + 1x2e", scope="global"),
            output="1o",
            covariance=FullCovariance(),
        )
    assert caught.value.certificate.code == "parity_unreachable"
    assert caught.value.certificate.details["parity_obstruction"]


def test_full_family_active_reachability_remains_a_hard_gate():
    with pytest.raises(CompilationError) as caught:
        plan_readout(
            FeatureSpec.from_irreps("2x0e + 1x2e", scope="global"),
            output="1o",
            covariance=FullCovariance(),
        )
    assert caught.value.certificate.code == "parity_unreachable"


@pytest.mark.parametrize(
    "feature,code",
    [
        (FeatureSpec.from_irreps(SEED, group="SO3"), "unsupported_group_contract"),
        (
            FeatureSpec.from_irreps(SEED, layout="compiler_native"),
            "unsupported_feature_layout",
        ),
        (
            FeatureSpec.from_irreps(SEED, scope="node", allow_pooling=False),
            "scope_pooling_forbidden",
        ),
    ],
)
def test_feature_contract_failures_are_structured(feature, code):
    with pytest.raises(CompilationError) as caught:
        plan_readout(feature, output="ij=ji", covariance=FullCovariance())
    assert caught.value.certificate.code == code


def test_truncated_execution_has_separate_approximation_certificate():
    seed = FeatureSpec.from_irreps(SEED, scope="global")
    _, report = compile_readout(
        seed,
        output="ij=ji",
        covariance=FullCovariance(),
        fidelity=TruncatedMultiplicityRank(rank=1),
    )
    assert report.family["relation_to_canonical"] == "canonical_full"
    assert report.execution_fidelity["exactness"] == "approximate_for_active_family"
    assert report.execution_fidelity["checkpoint_mapping"] == "not_available"
    assert report.execution_fidelity["approximation"]["kind"] == "truncated_multiplicity_rank"
    assert "truncated_contraction" in _certificate_codes(report)


def test_compatibility_hash_covers_scope_and_layout():
    common = dict(output="ij=ji", covariance=FullCovariance())
    global_plan = plan_readout(
        FeatureSpec.from_irreps(SEED, scope="global"), **common
    )
    node_plan = plan_readout(
        FeatureSpec.from_irreps(SEED, scope="node"), **common
    )
    assert global_plan.compatibility_hash != node_plan.compatibility_hash


def test_graph_precision_is_restricted_family_with_exact_lowering():
    graph = EquivariantOutputGraph(
        num_nodes=3,
        edges=((0, 1), (1, 2)),
        node_irrep="1o",
    )
    plan = plan_readout(
        FeatureSpec.from_irreps(SEED, scope="node"),
        output=graph.output_irreps,
        covariance=GraphPrecision(graph),
        output_scope="global",
    )
    assert plan.report.family["kind"] == "graph_precision"
    assert plan.report.family["relation_to_canonical"] == "strict_subfamily"
    assert plan.report.execution_fidelity["exactness"] == "exact_for_active_family"
    assert plan.report.execution_fidelity["approximation"] is None
    assert plan.report["pooling_required"]


def test_backbone_binding_checks_full_feature_fingerprint():
    class DummyBackbone(torch.nn.Module):
        feature_scope = "node"

        def __init__(self, irreps):
            super().__init__()
            self.irreps_out = irreps

    seed = FeatureSpec.from_irreps(SEED, scope="node")
    plan = plan_readout(seed, output="ij=ji", covariance=FullCovariance())
    assert isinstance(plan.bind(DummyBackbone(seed.irreps)), torch.nn.Module)

    incompatible = DummyBackbone(seed.irreps)
    incompatible.feature_basis_convention = "different_real_basis"
    with pytest.raises(CompilationError) as caught:
        plan.bind(incompatible)
    assert caught.value.certificate.code == "backbone_compatibility_mismatch"
