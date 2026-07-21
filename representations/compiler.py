"""Representation compiler for probabilistic O(3)-equivariant outputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

import torch
from compatibility.e3nn import o3

from representations.adaptive_lifting import (
    O3AdaptiveLifting,
    O3LiftingPlan,
    O3ReachabilityAnalysis,
    analyze_lifting_graph,
    direct_sum_irreps,
    irrep_multiplicities,
)
from representations.o3_irreps import O3IrrepsSpec
from representations.graph_structure import EquivariantOutputGraph
from representations.diagnostics import (
    CompilationCertificate,
    CompilationError,
    UnreachableActiveTargetError,
)
from representations.operator_ir import OperatorFamilyPlan
from representations.operator_lowering import (
    install_parameter_projections,
    lower_operator_program,
    project_parameter_bindings,
)

if TYPE_CHECKING:
    from equivcompiler.executors import ExecutorDecision


CovarianceMode = Literal["full", "block", "low_rank", "graph"]
OutputScope = Literal["global", "dense"]


@dataclass(frozen=True)
class LoweringConfig:
    """Execution-neutral settings remaining after semantic planning."""

    output_scope: OutputScope = "global"
    parameter_budget: int | None = None

    def __post_init__(self):
        if self.output_scope not in {"global", "dense"}:
            raise ValueError(f"unknown output scope: {self.output_scope}")
        if self.parameter_budget is not None and self.parameter_budget < 1:
            raise ValueError("parameter_budget must be positive")


def _isotypic_parameter_count(output_irreps: o3.Irreps) -> int:
    return sum(
        multiplicity * (multiplicity + 1) // 2
        for multiplicity in irrep_multiplicities(output_irreps).values()
    )


@dataclass(frozen=True)
class O3Compilation:
    """Immutable result of compiling an output representation."""

    output_spec: O3IrrepsSpec
    feature_contract: Any
    seed_irreps: o3.Irreps
    operator_family: OperatorFamilyPlan
    canonical_target_irreps: o3.Irreps
    active_target_irreps: o3.Irreps
    canonical_reachability: O3ReachabilityAnalysis
    active_reachability: O3ReachabilityAnalysis
    executor_decision: "ExecutorDecision"
    distribution_spec: Any
    backend: Literal["spherical_cg", "cartesian_stf"]
    backend_exact: bool
    stf_contraction_rank: int | None
    config: LoweringConfig

    @property
    def covariance_mode(self) -> str:
        return self.operator_family.kind

    @property
    def covariance_rank(self) -> int | None:
        return self.operator_family.rank

    @property
    def covariance_parameter_count(self) -> int:
        return self.operator_family.parameter_count

    @property
    def graph_structure(self) -> EquivariantOutputGraph | None:
        return self.operator_family.graph

    @property
    def canonical_plan(self) -> O3LiftingPlan | None:
        return self.canonical_reachability.plan

    @property
    def active_plan(self) -> O3LiftingPlan:
        plan = self.active_reachability.plan
        if plan is None:
            raise RuntimeError(
                "a completed compilation must have an active lifting plan"
            )
        return plan

    @property
    def mean_irreps(self) -> o3.Irreps:
        return self.output_spec.irreps

    @property
    def covariance_irreps(self) -> o3.Irreps:
        return self.output_spec.symmetric_square_irreps

    def as_dict(self) -> dict:
        return self.report().as_dict()

    def report(self, executable: torch.nn.Module | None = None):
        """Return the stable, machine-readable compilation report."""
        from representations.report import build_compilation_report

        return build_compilation_report(self, executable)

    def build_head(self) -> "O3CompiledOutputHead":
        return O3CompiledOutputHead(self)

    def build_spd_map(self):
        return lower_operator_program(self)

    def build_distribution(self):
        return self.distribution_spec.materialize_log_prob()

    def build_model(self, backbone: torch.nn.Module):
        """Build a predictor driven entirely by this compilation result."""
        from models.structured_predictor import StructuredProbabilisticPredictor

        if o3.Irreps(backbone.irreps_out) != self.seed_irreps:
            raise ValueError(
                f"backbone irreps {backbone.irreps_out} do not match compiled "
                f"seed irreps {self.seed_irreps}"
            )
        return StructuredProbabilisticPredictor(
            backbone=backbone,
            output_spec=self.output_spec,
            joint_head=self.build_head(),
            spd_map=self.build_spd_map(),
            distribution=self.build_distribution(),
            compilation=self,
        )


class O3ProgramCompiler:
    r"""Lower typed distribution/operator plans to an executable O(3) model.

    ``V + Sym^2(V)`` is retained as the unrestricted diagnostic reference.
    The selected operator family supplies its own active parameter expression;
    only that active expression is a compilation gate.
    """

    def __init__(
        self,
        output: O3IrrepsSpec | o3.Irreps | str,
        config: LoweringConfig | None = None,
        *,
        cartesian_formula: str | None = None,
    ):
        if cartesian_formula is not None:
            if isinstance(output, O3IrrepsSpec):
                raise ValueError("pass either output or cartesian_formula, not both")
            self.output_spec = O3IrrepsSpec.from_cartesian(cartesian_formula)
        elif isinstance(output, O3IrrepsSpec):
            self.output_spec = output
        else:
            self.output_spec = O3IrrepsSpec(o3.Irreps(output))
        self.config = config or LoweringConfig()

    @classmethod
    def from_cartesian(
        cls, formula: str, config: LoweringConfig | None = None
    ) -> "O3ProgramCompiler":
        return cls(o3.Irreps("0e"), config, cartesian_formula=formula)

    def _output_expression(self):
        from representations.representation_ir import IrrepsExpr

        return IrrepsExpr(self.output_spec.irreps, "location")

    @staticmethod
    def _validate_reachability_contract(
        analysis: O3ReachabilityAnalysis,
        seed: o3.Irreps,
        target: o3.Irreps,
        role: str,
    ) -> None:
        if analysis.seed_irreps != seed or analysis.target_irreps != target:
            raise CompilationError(
                CompilationCertificate(
                    code="reachability_contract_mismatch",
                    status="failure",
                    message=f"the supplied {role} reachability analysis has the wrong contract",
                    details={
                        "target_role": role,
                        "expected_seed_irreps": str(seed),
                        "analyzed_seed_irreps": str(analysis.seed_irreps),
                        "expected_target_irreps": str(target),
                        "analyzed_target_irreps": str(analysis.target_irreps),
                    },
                )
            )

    def compile(
        self,
        feature_contract: Any,
        *,
        operator_family: OperatorFamilyPlan,
        executor_decision: "ExecutorDecision | None",
        distribution_spec: Any,
        canonical_reachability: O3ReachabilityAnalysis | None = None,
        active_reachability: O3ReachabilityAnalysis | None = None,
    ) -> O3Compilation:
        if not hasattr(feature_contract, "irreps") or not hasattr(
            feature_contract, "fingerprint"
        ):
            raise TypeError("core compiler requires a complete FeatureSpec contract")
        seed = o3.Irreps(feature_contract.irreps)
        output_irreps = self.output_spec.irreps
        covariance_irreps = self.output_spec.symmetric_square_irreps
        canonical_target = direct_sum_irreps(output_irreps, covariance_irreps)
        active_target = (
            operator_family.active_expression(self._output_expression())
            .decompose_o3()
            .irreps
        )

        if canonical_reachability is None:
            canonical_reachability = analyze_lifting_graph(seed, canonical_target)
        self._validate_reachability_contract(
            canonical_reachability, seed, canonical_target, "canonical"
        )

        if active_reachability is None:
            active_reachability = (
                canonical_reachability
                if active_target == canonical_target
                else analyze_lifting_graph(seed, active_target)
            )
        self._validate_reachability_contract(
            active_reachability, seed, active_target, "active"
        )
        if not active_reachability.reachable:
            assert active_reachability.failure is not None
            failure = active_reachability.failure
            raise UnreachableActiveTargetError(
                CompilationCertificate(
                    code=failure.code,
                    status="failure",
                    message=(
                        "the selected active parameter target is unreachable; "
                        + failure.message
                    ),
                    details={**failure.details, "target_role": "active"},
                )
            )

        active_plan = active_reachability.plan
        assert active_plan is not None
        if executor_decision is None:
            raise CompilationError(
                CompilationCertificate(
                    code="missing_executor_decision",
                    status="failure",
                    message="core lowering requires a planning-time executor decision",
                    details={"active_plan": active_plan.as_dict()},
                )
            )
        capability = executor_decision.capability
        if not capability.supported:
            raise CompilationError(
                CompilationCertificate(
                    code="invalid_executor_decision",
                    status="failure",
                    message="selected executor capability is not supported",
                    details=capability.as_dict(),
                )
            )
        backend = executor_decision.name
        active_plan_hash = hashlib.sha256(
            json.dumps(
                active_plan.as_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        decision_contract = {
            "feature_fingerprint": (
                capability.feature_fingerprint,
                feature_contract.fingerprint,
            ),
            "active_plan_hash": (capability.active_plan_hash, active_plan_hash),
            "operator_program_hash": (
                capability.operator_program_hash,
                operator_family.assembly.fingerprint,
            ),
        }
        mismatches = {
            key: {"decision": value[0], "compilation": value[1]}
            for key, value in decision_contract.items()
            if value[0] != value[1]
        }
        if mismatches:
            raise CompilationError(
                CompilationCertificate(
                    code="executor_decision_contract_mismatch",
                    status="failure",
                    message="executor decision does not certify this active program",
                    details={"mismatches": mismatches},
                )
            )
        contraction_rank = capability.fidelity.effective_contraction_rank
        backend_exact = capability.exact
        return O3Compilation(
            output_spec=self.output_spec,
            feature_contract=feature_contract,
            seed_irreps=seed,
            operator_family=operator_family,
            canonical_target_irreps=canonical_target,
            active_target_irreps=active_target,
            canonical_reachability=canonical_reachability,
            active_reachability=active_reachability,
            executor_decision=executor_decision,
            distribution_spec=distribution_spec,
            backend=backend,
            backend_exact=backend_exact,
            stf_contraction_rank=contraction_rank,
            config=self.config,
        )


class O3CompiledOutputHead(torch.nn.Module):
    """Shared lifting trunk with checkpoint-preserving backend lowering."""

    def __init__(self, compilation: O3Compilation):
        super().__init__()
        self.compilation = compilation
        self.output_spec = compilation.output_spec
        self.pool = compilation.config.output_scope == "global"
        active_irreps = compilation.active_target_irreps
        self.lifting = O3AdaptiveLifting(
            compilation.seed_irreps,
            active_irreps,
            plan=compilation.active_plan,
            tensor_product_backend=(
                "dense_projector"
                if compilation.backend == "cartesian_stf"
                else "spherical_cg"
            ),
            contraction_rank=compilation.stf_contraction_rank,
        )
        self.mean_projection = o3.Linear(active_irreps, self.output_spec.irreps)
        install_parameter_projections(self)

    def _compiled_features(
        self, node_features: torch.Tensor, batch: torch.Tensor | None
    ) -> torch.Tensor:
        hidden = node_features
        if self.pool:
            if batch is None:
                raise ValueError("batch is required for global output")
            from models.pooling import mean_pool

            hidden = mean_pool(hidden, batch)
        return self.lifting(hidden)

    def forward_parameters(
        self, node_features: torch.Tensor, batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project only certified operator parameters from input features."""
        compiled = self._compiled_features(node_features, batch)
        return project_parameter_bindings(self, compiled)

    def forward(
        self, node_features: torch.Tensor, batch: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compiled = self._compiled_features(node_features, batch)
        mean = self.mean_projection(compiled)
        parameters = project_parameter_bindings(self, compiled)
        return mean, parameters
