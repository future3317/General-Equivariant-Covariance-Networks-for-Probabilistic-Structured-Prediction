"""Pure planning between semantic analysis and module materialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

import torch
from compatibility.e3nn import o3

from equivcompiler.modules import CompiledProbabilisticReadout
from equivcompiler.policies import (
    AutoBudget,
    CovariancePolicy,
    ExactOnly,
    FullCovariance,
    GraphPrecision,
    IsotypicBlockCovariance,
    LowRankCovariance,
    LoweringPolicy,
    TruncatedMultiplicityRank,
)
from equivcompiler.specs import FeatureSpec, OutputSemantics, describe_output
from representations import (
    CompilationCertificate,
    CompilationError,
    CompilationReport,
    CompilerConfig,
    EquivariantOutputGraph,
    O3Compilation,
    O3RepresentationCompiler,
    direct_sum_irreps,
    irrep_multiplicities,
    plan_lifting_graph,
)


OutputScope = Literal["global", "node", "edge"]
COMPILER_VERSION = "0.2"


def _isotypic_parameters(irreps: o3.Irreps) -> int:
    return sum(
        multiplicity * (multiplicity + 1) // 2
        for multiplicity in irrep_multiplicities(irreps).values()
    )


def _graph_parameters(graph: EquivariantOutputGraph) -> int:
    local = graph.block_dim * (graph.block_dim + 1) // 2
    return local * graph.num_potentials


def _policy_record(policy: CovariancePolicy | LoweringPolicy) -> dict[str, Any]:
    if isinstance(policy, FullCovariance):
        return {"kind": "full_covariance"}
    if isinstance(policy, LowRankCovariance):
        return {"kind": "low_rank_plus_isotropic", "rank": policy.rank}
    if isinstance(policy, IsotypicBlockCovariance):
        return {"kind": "isotypic_block_covariance"}
    if isinstance(policy, GraphPrecision):
        return {"kind": "graph_precision", "graph": policy.graph.as_dict()}
    if isinstance(policy, AutoBudget):
        return {
            "kind": "auto_budget",
            "budget": policy.budget,
            "low_rank": policy.low_rank,
            "allowed_families": list(policy.allowed_families),
            "graph": policy.graph.as_dict() if policy.graph is not None else None,
        }
    if isinstance(policy, ExactOnly):
        return {"kind": "exact_only", "backend": policy.backend}
    if isinstance(policy, TruncatedMultiplicityRank):
        return {
            "kind": "truncated_multiplicity_rank",
            "backend": policy.backend,
            "rank": policy.rank,
        }
    raise TypeError(f"unsupported policy: {type(policy).__name__}")


def _select_family(
    policy: CovariancePolicy,
    semantics: OutputSemantics,
) -> tuple[str, int, EquivariantOutputGraph | None, dict[str, Any]]:
    dimension = semantics.output_dimension
    if isinstance(policy, FullCovariance):
        return "full", min(8, dimension), None, {
            "rule": "explicit_request",
            "requested_family": "full",
        }
    if isinstance(policy, LowRankCovariance):
        return "low_rank", min(policy.rank, dimension), None, {
            "rule": "explicit_request",
            "requested_family": "low_rank",
            "rank": min(policy.rank, dimension),
        }
    if isinstance(policy, IsotypicBlockCovariance):
        return "block", min(8, dimension), None, {
            "rule": "explicit_request",
            "requested_family": "block",
        }
    if isinstance(policy, GraphPrecision):
        return "graph", min(8, dimension), policy.graph, {
            "rule": "explicit_request",
            "requested_family": "graph",
        }
    if not isinstance(policy, AutoBudget):
        raise TypeError(f"unsupported covariance policy: {type(policy).__name__}")

    rank = min(policy.low_rank, dimension)
    counts: dict[str, int | None] = {
        "full": semantics.full_covariance_parameters,
        "graph": _graph_parameters(policy.graph) if policy.graph is not None else None,
        "low_rank": dimension * rank + 1,
        "block": _isotypic_parameters(semantics.output_spec.irreps),
    }
    considered = []
    selected = None
    for family in policy.allowed_families:
        parameters = counts[family]
        eligible = parameters is not None and parameters <= policy.budget
        considered.append(
            {
                "family": family,
                "parameters": parameters,
                "within_budget": eligible,
            }
        )
        if selected is None and eligible:
            selected = family
    if selected is None:
        raise CompilationError(
            CompilationCertificate(
                code="covariance_budget_unsatisfied",
                status="failure",
                message="no authorized covariance family satisfies the parameter budget",
                details={
                    "budget": policy.budget,
                    "allowed_families": list(policy.allowed_families),
                    "considered": considered,
                },
            )
        )
    return selected, rank, policy.graph if selected == "graph" else None, {
        "rule": "parameter_budget",
        "budget": policy.budget,
        "selected_family": selected,
        "selected_parameters": counts[selected],
        "full_parameters": counts["full"],
        "considered": considered,
    }


def _validate_contract(
    seed: FeatureSpec,
    output_scope: OutputScope,
) -> bool:
    if seed.group != "O3":
        raise CompilationError(
            CompilationCertificate(
                code="unsupported_group_contract",
                status="failure",
                message="the released executable backend supports O(3), not the declared group",
                details={"declared_group": seed.group, "supported": ["O3"]},
            )
        )
    if seed.layout != "e3nn" or seed.basis_convention != "e3nn_real_v1":
        raise CompilationError(
            CompilationCertificate(
                code="unsupported_feature_layout",
                status="failure",
                message="feature layout or real-basis convention has no registered lowering",
                details={
                    "layout": seed.layout,
                    "basis_convention": seed.basis_convention,
                    "supported": {
                        "layout": "e3nn",
                        "basis_convention": "e3nn_real_v1",
                    },
                },
            )
        )
    if output_scope == "global":
        pool_input = seed.scope != "global"
        if pool_input and not seed.allow_pooling:
            raise CompilationError(
                CompilationCertificate(
                    code="scope_pooling_forbidden",
                    status="failure",
                    message="global output requires pooling but the feature contract forbids it",
                    details={"seed_scope": seed.scope, "output_scope": output_scope},
                )
            )
        return pool_input
    if output_scope != seed.scope:
        raise CompilationError(
            CompilationCertificate(
                code="scope_incompatible",
                status="failure",
                message="non-global output scope must match the feature scope",
                details={"seed_scope": seed.scope, "output_scope": output_scope},
            )
        )
    return False


def _validate_graph(
    graph: EquivariantOutputGraph | None,
    semantics: OutputSemantics,
) -> None:
    if graph is not None and graph.output_irreps != semantics.output_spec.irreps:
        raise CompilationError(
            CompilationCertificate(
                code="graph_output_mismatch",
                status="failure",
                message="graph output representation does not match declared output semantics",
                details={
                    "graph_output": str(graph.output_irreps),
                    "declared_output": semantics.output_representation,
                },
            )
        )


def _compatibility_hash(
    seed: FeatureSpec,
    semantics: OutputSemantics,
    covariance: CovariancePolicy,
    lowering: LoweringPolicy,
    output_scope: OutputScope,
) -> str:
    payload = {
        "compiler_version": COMPILER_VERSION,
        "feature": seed.as_dict(),
        "output": semantics.as_dict(),
        "covariance_policy": _policy_record(covariance),
        "lowering_policy": _policy_record(lowering),
        "output_scope": output_scope,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decorate_report(
    report: CompilationReport,
    *,
    seed: FeatureSpec,
    semantics: OutputSemantics,
    covariance_policy: CovariancePolicy,
    lowering_policy: LoweringPolicy,
    output_scope: OutputScope,
    selection_reason: dict[str, Any],
    compatibility_hash: str,
) -> CompilationReport:
    record = report.as_dict()
    certificates = []
    for certificate in record["certificates"]:
        certificate = dict(certificate)
        if certificate["code"] == "structured_covariance_restriction":
            details = dict(certificate["details"])
            details["selected_by"] = selection_reason["rule"]
            certificate["details"] = details
        certificates.append(certificate)
    family = dict(record["family"])
    family["policy"] = _policy_record(covariance_policy)
    family["selection_reason"] = selection_reason
    lowering = dict(record["lowering"])
    lowering["policy"] = _policy_record(lowering_policy)
    record.update(
        {
            "seed": seed.as_dict(),
            "output": semantics.as_dict(),
            "family": family,
            "lowering": lowering,
            "output_scope": output_scope,
            "pooling_required": output_scope == "global" and seed.scope != "global",
            "compatibility_hash": compatibility_hash,
            "feature_fingerprint": seed.fingerprint,
            "certificates": certificates,
        }
    )
    return CompilationReport.from_dict(record)


@dataclass(frozen=True)
class CompilationPlan:
    """Immutable dry-run result; module construction is deliberately deferred."""

    seed: FeatureSpec
    semantics: OutputSemantics
    covariance_policy: CovariancePolicy
    lowering_policy: LoweringPolicy
    output_scope: OutputScope
    compilation: O3Compilation
    selection_reason: dict[str, Any]
    compatibility_hash: str
    report: CompilationReport

    def build_readout(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> CompiledProbabilisticReadout:
        readout = CompiledProbabilisticReadout(self.compilation)
        if device is not None or dtype is not None:
            readout = readout.to(device=device, dtype=dtype)
        return readout

    def bind(self, backbone: torch.nn.Module):
        actual = FeatureSpec.from_backbone(backbone)
        if actual.fingerprint != self.seed.fingerprint:
            raise CompilationError(
                CompilationCertificate(
                    code="backbone_compatibility_mismatch",
                    status="failure",
                    message="backbone feature contract does not match the compilation plan",
                    details={
                        "compiled_feature": self.seed.as_dict(),
                        "current_feature": actual.as_dict(),
                        "compiled_fingerprint": self.seed.fingerprint,
                        "current_fingerprint": actual.fingerprint,
                    },
                )
            )
        return self.compilation.build_model(backbone)

    def report_for(self, executable: torch.nn.Module) -> CompilationReport:
        return _decorate_report(
            self.compilation.report(executable),
            seed=self.seed,
            semantics=self.semantics,
            covariance_policy=self.covariance_policy,
            lowering_policy=self.lowering_policy,
            output_scope=self.output_scope,
            selection_reason=self.selection_reason,
            compatibility_hash=self.compatibility_hash,
        )


def plan_readout(
    seed: FeatureSpec,
    *,
    output,
    covariance: CovariancePolicy,
    lowering: LoweringPolicy = ExactOnly(),
    distribution: Literal["gaussian", "student_t"] = "gaussian",
    student_t_dof: float = 5.0,
    output_scope: OutputScope = "global",
) -> CompilationPlan:
    """Plan semantics, reachability, family, and lowering without modules."""
    semantics = describe_output(output)
    pool_input = _validate_contract(seed, output_scope)
    canonical_target = direct_sum_irreps(
        semantics.output_spec.irreps,
        semantics.output_spec.symmetric_square_irreps,
    )
    canonical_plan = plan_lifting_graph(seed.irreps, canonical_target)
    mode, rank, graph, selection_reason = _select_family(covariance, semantics)
    _validate_graph(graph, semantics)

    if isinstance(lowering, ExactOnly):
        backend = lowering.backend
        contraction_rank = None
    elif isinstance(lowering, TruncatedMultiplicityRank):
        backend = lowering.backend
        contraction_rank = lowering.rank
    else:
        raise TypeError(f"unsupported lowering policy: {type(lowering).__name__}")

    config = CompilerConfig(
        covariance=mode,
        output_scope="global" if pool_input else "dense",
        objective=distribution,
        parameter_budget=(
            covariance.budget if isinstance(covariance, AutoBudget) else 1
        ),
        low_rank=rank,
        student_t_dof=student_t_dof,
        backend=backend,
        stf_contraction_rank=contraction_rank,
    )
    compiler = (
        O3RepresentationCompiler.for_graph(graph, config)
        if graph is not None
        else O3RepresentationCompiler(semantics.output_spec, config)
    )
    compilation = compiler.compile(seed.irreps, canonical_plan=canonical_plan)
    if isinstance(lowering, ExactOnly) and not compilation.backend_exact:
        raise RuntimeError("ExactOnly materialized an approximate backend")

    compatibility_hash = _compatibility_hash(
        seed, semantics, covariance, lowering, output_scope
    )
    report = _decorate_report(
        compilation.report(),
        seed=seed,
        semantics=semantics,
        covariance_policy=covariance,
        lowering_policy=lowering,
        output_scope=output_scope,
        selection_reason=selection_reason,
        compatibility_hash=compatibility_hash,
    )
    return CompilationPlan(
        seed=seed,
        semantics=semantics,
        covariance_policy=covariance,
        lowering_policy=lowering,
        output_scope=output_scope,
        compilation=compilation,
        selection_reason=selection_reason,
        compatibility_hash=compatibility_hash,
        report=report,
    )
