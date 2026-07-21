"""Pure planning from semantic representations to executable lowerings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

import torch

from equivcompiler.distributions import (
    DistributionSpec,
    normalize_distribution,
)
from equivcompiler.executors import (
    CandidateEnumerator,
    DEFAULT_EXACT_LOWERINGS,
    ExecutorDecision,
    ExecutionContext,
)
from equivcompiler.modules import CompiledProbabilisticReadout
from equivcompiler.policies import (
    AutoBudget,
    CostPolicy,
    CovariancePolicy,
    ExactExecutorCandidates,
    ExactOnly,
    ExecutorPolicy,
    FidelityPolicy,
    FirstFeasible,
    MinimizeLatency,
    OperatorFamilySpec,
    PreferExecutor,
    SpecificExecutor,
    TruncatedMultiplicityRank,
)
from equivcompiler.specs import FeatureSpec, OutputSemantics, describe_output
from equivcompiler.signatures import plan_fingerprints
from representations import (
    CompilationCertificate,
    CompilationError,
    CompilationReport,
    LoweringConfig,
    O3Compilation,
    O3ProgramCompiler,
    analyze_lifting_graph,
)
from representations.operator_ir import OperatorFamilyPlan


OutputScope = Literal["global", "node", "edge"]
COMPILER_VERSION = "0.4"


def _policy_record(policy: object) -> dict[str, Any]:
    if hasattr(policy, "as_dict"):
        return policy.as_dict()  # type: ignore[no-any-return]
    if isinstance(policy, ExactOnly):
        return {"kind": "exact_only"}
    if isinstance(policy, TruncatedMultiplicityRank):
        return {"kind": "truncated_multiplicity_rank", "rank": policy.rank}
    if isinstance(policy, ExactExecutorCandidates):
        return {
            "kind": "exact_executor_candidates",
            "candidates": list(policy.candidates),
        }
    if isinstance(policy, SpecificExecutor):
        return {"kind": "specific_executor", "name": policy.name}
    if isinstance(policy, PreferExecutor):
        return {"kind": "prefer_executor", "priority": list(policy.priority)}
    if isinstance(policy, MinimizeLatency):
        return {
            "kind": "minimize_measured_latency",
            "signature": policy.signature.as_dict(),
            "measurements": [
                {
                    "executor": item.executor,
                    "median_ms": item.median_ms,
                    "iqr_ms": item.iqr_ms,
                }
                for item in policy.measurements
            ],
        }
    raise TypeError(f"unsupported policy: {type(policy).__name__}")


def _select_family(
    policy: CovariancePolicy,
    semantics: OutputSemantics,
) -> tuple[OperatorFamilyPlan, dict[str, Any]]:
    if isinstance(policy, OperatorFamilySpec):
        plan = policy.compile(semantics.output_spec)
        return plan, {
            "rule": "explicit_request",
            "selected_family": plan.kind,
            "selected_parameters": plan.parameter_count,
        }

    candidates = (
        policy.candidates if isinstance(policy, AutoBudget) else policy.priority
    )
    compiled = [candidate.compile(semantics.output_spec) for candidate in candidates]
    considered = [
        {
            "family": item.kind,
            "parameters": item.parameter_count,
            "within_budget": item.parameter_count <= policy.max_parameters,
            "relation_to_full": item.relation_to_full.value,
        }
        for item in compiled
    ]
    pairwise_relations = []
    for left_index, left in enumerate(compiled):
        for right in compiled[left_index + 1 :]:
            certificate = left.relation_to(right)
            pairwise_relations.append(
                {
                    "left": left.kind,
                    "right": right.kind,
                    **certificate.as_dict(),
                }
            )
    feasible = [
        (index, item)
        for index, item in enumerate(compiled)
        if item.parameter_count <= policy.max_parameters
    ]
    if not feasible:
        raise CompilationError(
            CompilationCertificate(
                code="covariance_budget_unsatisfied",
                status="failure",
                message="no authorized operator family satisfies the parameter budget",
                details={
                    "budget": policy.max_parameters,
                    "considered": considered,
                },
            )
        )

    if isinstance(policy, AutoBudget):
        selected_index, selected = min(
            feasible, key=lambda pair: (*policy.objective.score(pair[1]), pair[0])
        )
        rule = "family_cost_model_under_budget"
        selection_semantics = "cost_objective_with_declaration_order_tie_break"
    elif isinstance(policy, FirstFeasible):
        selected_index, selected = feasible[0]
        rule = "user_declared_priority"
        selection_semantics = "first_feasible_in_explicit_priority"
    else:  # pragma: no cover - guarded by the public policy union
        raise TypeError(f"unsupported covariance policy: {type(policy).__name__}")

    return selected, {
        "rule": rule,
        "selection_semantics": selection_semantics,
        "budget": policy.max_parameters,
        "selected_family": selected.kind,
        "selected_candidate_index": selected_index,
        "selected_parameters": selected.parameter_count,
        "cost_objective": (
            policy.objective.as_dict() if isinstance(policy, AutoBudget) else None
        ),
        "considered": considered,
        "pairwise_family_relations": pairwise_relations,
        "candidate_family_order_is_not_a_mathematical_inclusion_order": True,
    }


def _validate_contract(seed: FeatureSpec, output_scope: OutputScope) -> bool:
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
                message="feature coordinates have no registered executable lowering",
                details={
                    "layout": seed.layout,
                    "basis_convention": seed.basis_convention,
                    "supported": {"layout": "e3nn", "basis_convention": "e3nn_real_v1"},
                },
            )
        )
    if not seed.metric.is_orthonormal:
        raise CompilationError(
            CompilationCertificate(
                code="metric_lowering_unavailable",
                status="failure",
                message="the metric is typed, but this release has no Gram-whitening lowering",
                details={"metric": seed.metric.as_dict()},
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


def _select_executor(
    seed: FeatureSpec,
    semantics: OutputSemantics,
    family: OperatorFamilyPlan,
    active_plan,
    distribution: DistributionSpec,
    fidelity: FidelityPolicy,
    executor: ExecutorPolicy,
    cost: CostPolicy,
    lifting_backend: str = "e3nn",
    cueq_method: str = "naive",
) -> ExecutorDecision:
    context = ExecutionContext(
        feature=seed,
        output=semantics.output_spec,
        active_plan=active_plan,
        operator_family=family,
        fidelity=fidelity,
    )
    support = DEFAULT_EXACT_LOWERINGS.as_dict(context)
    requested = (
        executor.candidates
        if isinstance(executor, ExactExecutorCandidates)
        else (executor.name,)
    )
    certificates = CandidateEnumerator(DEFAULT_EXACT_LOWERINGS).enumerate(
        requested, context
    )
    available = tuple(item.executor for item in certificates)
    if not certificates:
        raise CompilationError(
            CompilationCertificate(
                code="backend_incompatible",
                status="failure",
                message="no requested executor supports the active representation program",
                details={
                    "requested": list(requested),
                    "support": support,
                    "fidelity": _policy_record(fidelity),
                    "operator_program_hash": family.assembly.fingerprint,
                    "active_plan": active_plan.as_dict(),
                },
            )
        )

    if isinstance(cost, PreferExecutor):
        selected = next((name for name in cost.priority if name in available), None)
        if selected is None:
            raise CompilationError(
                CompilationCertificate(
                    code="cost_policy_no_candidate",
                    status="failure",
                    message="the static cost priority contains no supported executor",
                    details={
                        "available": list(available),
                        "priority": list(cost.priority),
                    },
                )
            )
        basis = {
            "method": "explicit_static_priority",
            "performance_claim": "none",
            "available_candidates": list(available),
            "priority": list(cost.priority),
        }
    elif isinstance(cost, MinimizeLatency):
        expected = plan_fingerprints(
            feature_record=seed.as_dict(),
            output_record=semantics.as_dict(),
            active_plan_record=active_plan.as_dict(),
                operator_record=family.assembly.semantic_dict(),
                distribution_record=distribution.as_dict(),
                lowering_record={
                    "lifting_backend": lifting_backend,
                    "cueq_method": cueq_method,
                },
        )
        mismatches = {
            "feature_fingerprint": (
                cost.signature.feature_fingerprint,
                seed.fingerprint,
            ),
            "active_plan_hash": (
                cost.signature.active_plan_hash,
                expected["active_plan_hash"],
            ),
            "operator_program_hash": (
                cost.signature.operator_program_hash,
                expected["operator_program_hash"],
            ),
            "semantic_plan_hash": (
                cost.signature.semantic_plan_hash,
                expected["semantic_plan_hash"],
            ),
        }
        mismatches = {
            key: {"measured": pair[0], "planned": pair[1]}
            for key, pair in mismatches.items()
            if pair[0] != pair[1]
        }
        if mismatches:
            raise CompilationError(
                CompilationCertificate(
                    code="executor_measurement_signature_mismatch",
                    status="failure",
                    message="latency evidence does not describe the active semantic plan",
                    details={"mismatches": mismatches},
                )
            )
        measured = [item for item in cost.measurements if item.executor in available]
        if not measured:
            raise CompilationError(
                CompilationCertificate(
                    code="missing_executor_measurement",
                    status="failure",
                    message="no compatible latency measurement matches the requested signature",
                    details={
                        "available": list(available),
                        "signature": cost.signature.as_dict(),
                    },
                )
            )
        best = min(measured, key=lambda item: item.median_ms)
        selected = best.executor
        basis = {
            "method": "measured_autotune",
            "performance_claim": "measured_for_exact_shape_signature",
            "signature": cost.signature.as_dict(),
            "available_candidates": list(available),
            "selected_median_ms": best.median_ms,
            "measurements": [
                {
                    "executor": item.executor,
                    "median_ms": item.median_ms,
                    "iqr_ms": item.iqr_ms,
                }
                for item in measured
            ],
        }
    else:  # pragma: no cover
        raise TypeError(f"unsupported cost policy: {type(cost).__name__}")

    capability = next(item for item in certificates if item.executor == selected)
    basis["capability_certificate"] = capability.as_dict()
    return ExecutorDecision(selected, capability, basis)


def _compatibility_hash(
    seed: FeatureSpec,
    semantics: OutputSemantics,
    distribution: DistributionSpec,
    covariance: CovariancePolicy,
    fidelity: FidelityPolicy,
    executor: ExecutorPolicy,
    cost: CostPolicy,
    output_scope: OutputScope,
    lifting_backend: str,
    cueq_method: str,
) -> str:
    payload = {
        "compiler_version": COMPILER_VERSION,
        "feature": seed.as_dict(),
        "output": semantics.as_dict(),
        "distribution": distribution.as_dict(),
        "covariance_policy": _policy_record(covariance),
        "fidelity_policy": _policy_record(fidelity),
        "executor_policy": _policy_record(executor),
        "cost_policy": _policy_record(cost),
        "output_scope": output_scope,
        "lifting_backend": lifting_backend,
        "cueq_method": cueq_method,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decorate_report(
    report: CompilationReport,
    *,
    seed: FeatureSpec,
    semantics: OutputSemantics,
    distribution: DistributionSpec,
    covariance_policy: CovariancePolicy,
    fidelity_policy: FidelityPolicy,
    executor_policy: ExecutorPolicy,
    cost_policy: CostPolicy,
    output_scope: OutputScope,
    selection_reason: dict[str, Any],
    backend_selection_basis: dict[str, Any],
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
    execution_fidelity = dict(record["execution_fidelity"])
    execution_fidelity["policy"] = _policy_record(fidelity_policy)
    backend_selection = dict(record["backend_selection_basis"])
    backend_selection.update(backend_selection_basis)
    backend_selection["executor_policy"] = _policy_record(executor_policy)
    backend_selection["cost_policy"] = _policy_record(cost_policy)
    record.update(
        {
            "seed": seed.as_dict(),
            "output": semantics.as_dict(),
            "distribution_ir": distribution.as_dict(),
            "family": family,
            "execution_fidelity": execution_fidelity,
            "backend_selection_basis": backend_selection,
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
    distribution_spec: DistributionSpec
    covariance_policy: CovariancePolicy
    fidelity_policy: FidelityPolicy
    executor_policy: ExecutorPolicy
    cost_policy: CostPolicy
    output_scope: OutputScope
    compilation: O3Compilation
    selection_reason: dict[str, Any]
    backend_selection_basis: dict[str, Any]
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
            distribution=self.distribution_spec,
            covariance_policy=self.covariance_policy,
            fidelity_policy=self.fidelity_policy,
            executor_policy=self.executor_policy,
            cost_policy=self.cost_policy,
            output_scope=self.output_scope,
            selection_reason=self.selection_reason,
            backend_selection_basis=self.backend_selection_basis,
            compatibility_hash=self.compatibility_hash,
        )


def plan_readout(
    seed: FeatureSpec,
    *,
    output,
    covariance: CovariancePolicy,
    distribution: DistributionSpec | Literal["gaussian", "student_t"] = "gaussian",
    fidelity: FidelityPolicy | None = None,
    executor: ExecutorPolicy = ExactExecutorCandidates(),
    cost: CostPolicy = PreferExecutor(),
    student_t_dof: float = 5.0,
    output_scope: OutputScope = "global",
    lifting_backend: str = "e3nn",
    cueq_method: str = "naive",
) -> CompilationPlan:
    """Analyze semantics/reachability and select an independently costed lowering."""
    fidelity_policy = fidelity or ExactOnly()
    distribution_spec = normalize_distribution(
        distribution, student_t_dof=student_t_dof
    )
    semantics = describe_output(output)
    pool_input = _validate_contract(seed, output_scope)
    family, selection_reason = _select_family(covariance, semantics)

    canonical_expression = distribution_spec.canonical_reference(semantics.output_spec)
    active_expression = distribution_spec.active_parameter_rep(
        semantics.output_spec, family
    )
    canonical_target = canonical_expression.decompose_o3().irreps
    active_target = active_expression.decompose_o3().irreps
    canonical_reachability = analyze_lifting_graph(seed.irreps, canonical_target)
    active_reachability = (
        canonical_reachability
        if active_target == canonical_target
        else analyze_lifting_graph(seed.irreps, active_target)
    )
    if not active_reachability.reachable:
        # The core compiler converts this diagnostic into the public active-target error.
        backend_selection_basis = {"method": "not_selected_active_unreachable"}
    else:
        active_plan = active_reachability.plan
        assert active_plan is not None
        executor_decision = _select_executor(
            seed,
            semantics,
            family,
            active_plan,
            distribution_spec,
            fidelity_policy,
            executor,
            cost,
            lifting_backend,
            cueq_method,
        )
        backend_selection_basis = executor_decision.selection_basis

    backend_selection_basis = {
        **backend_selection_basis,
        "lifting_backend": lifting_backend,
        "cueq_method": cueq_method,
    }

    config = LoweringConfig(
        output_scope="global" if pool_input else "dense",
        parameter_budget=getattr(covariance, "max_parameters", family.parameter_count),
        lifting_backend=lifting_backend,
        cueq_method=cueq_method,
    )
    compiler = O3ProgramCompiler(
        semantics.output_spec,
        config,
    )
    compilation = compiler.compile(
        seed,
        operator_family=family,
        executor_decision=executor_decision if active_reachability.reachable else None,
        distribution_spec=distribution_spec,
        canonical_reachability=canonical_reachability,
        active_reachability=active_reachability,
    )
    if isinstance(fidelity_policy, ExactOnly) and not compilation.backend_exact:
        raise RuntimeError("ExactOnly materialized an approximate executor")

    compatibility_hash = _compatibility_hash(
        seed,
        semantics,
        distribution_spec,
        covariance,
        fidelity_policy,
        executor,
        cost,
        output_scope,
        lifting_backend,
        cueq_method,
    )
    report = _decorate_report(
        compilation.report(),
        seed=seed,
        semantics=semantics,
        distribution=distribution_spec,
        covariance_policy=covariance,
        fidelity_policy=fidelity_policy,
        executor_policy=executor,
        cost_policy=cost,
        output_scope=output_scope,
        selection_reason=selection_reason,
        backend_selection_basis=backend_selection_basis,
        compatibility_hash=compatibility_hash,
    )
    return CompilationPlan(
        seed=seed,
        semantics=semantics,
        distribution_spec=distribution_spec,
        covariance_policy=covariance,
        fidelity_policy=fidelity_policy,
        executor_policy=executor,
        cost_policy=cost,
        output_scope=output_scope,
        compilation=compilation,
        selection_reason=selection_reason,
        backend_selection_basis=backend_selection_basis,
        compatibility_hash=compatibility_hash,
        report=report,
    )
