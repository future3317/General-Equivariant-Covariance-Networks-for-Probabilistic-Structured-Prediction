"""Representation compiler for probabilistic O(3)-equivariant outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
from representations.cartesian_stf import is_rank2_stf_output
from representations.diagnostics import (
    CompilationCertificate,
    CompilationError,
    UnreachableActiveTargetError,
)
from representations.operator_ir import OperatorFamilyPlan
from representations.operator_runtime import operator_runtime


CovarianceMode = Literal["auto", "full", "block", "low_rank", "graph"]
OutputScope = Literal["global", "dense"]
ObjectiveName = Literal["gaussian", "student_t"]
ExecutionBackend = Literal["auto", "spherical_cg", "cartesian_stf"]


@dataclass(frozen=True)
class CompilerConfig:
    """Architecture and probabilistic choices made by the compiler."""

    covariance: CovarianceMode = "auto"
    output_scope: OutputScope = "global"
    objective: ObjectiveName = "gaussian"
    parameter_budget: int = 192
    low_rank: int = 8
    student_t_dof: float = 5.0
    backend: ExecutionBackend = "auto"
    stf_contraction_rank: int | None = None

    def __post_init__(self):
        if self.covariance not in {"auto", "full", "block", "low_rank", "graph"}:
            raise CompilationError(
                CompilationCertificate(
                    code="unsupported_covariance_parameterization",
                    status="failure",
                    message=f"unknown covariance mode: {self.covariance}",
                    details={
                        "requested": self.covariance,
                        "supported": ["auto", "full", "block", "low_rank", "graph"],
                        "safeguard": "coordinate-wise Cholesky is intentionally unavailable because it is not conjugation equivariant",
                    },
                )
            )
        if self.output_scope not in {"global", "dense"}:
            raise ValueError(f"unknown output scope: {self.output_scope}")
        if self.objective not in {"gaussian", "student_t"}:
            raise ValueError(f"unknown proper objective: {self.objective}")
        if self.parameter_budget < 1:
            raise ValueError("parameter_budget must be positive")
        if self.low_rank < 1:
            raise ValueError("low_rank must be positive")
        if self.student_t_dof <= 0:
            raise ValueError("student_t_dof must be positive")
        if self.backend not in {"auto", "spherical_cg", "cartesian_stf"}:
            raise CompilationError(
                CompilationCertificate(
                    code="unsupported_execution_backend",
                    status="failure",
                    message=f"unknown execution backend: {self.backend}",
                    details={
                        "requested": self.backend,
                        "supported": ["auto", "spherical_cg", "cartesian_stf"],
                        "safeguard": "scalar Gaunt shortcuts are not accepted as a complete CG executor",
                    },
                )
            )
        if self.stf_contraction_rank is not None and self.stf_contraction_rank < 1:
            raise ValueError("stf_contraction_rank must be positive")


def _isotypic_parameter_count(output_irreps: o3.Irreps) -> int:
    return sum(
        multiplicity * (multiplicity + 1) // 2
        for multiplicity in irrep_multiplicities(output_irreps).values()
    )


@dataclass(frozen=True)
class O3Compilation:
    """Immutable result of compiling an output representation."""

    output_spec: O3IrrepsSpec
    seed_irreps: o3.Irreps
    operator_family: OperatorFamilyPlan
    canonical_target_irreps: o3.Irreps
    active_target_irreps: o3.Irreps
    canonical_reachability: O3ReachabilityAnalysis
    active_reachability: O3ReachabilityAnalysis
    backend: Literal["spherical_cg", "cartesian_stf"]
    backend_exact: bool
    stf_contraction_rank: int | None
    config: CompilerConfig

    @property
    def covariance_mode(self) -> Literal["full", "block", "low_rank", "graph"]:
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
            raise RuntimeError("a completed compilation must have an active lifting plan")
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
        return operator_runtime(self.operator_family.kind).build_spd_map(self)

    def build_distribution(self):
        if self.config.objective == "gaussian":
            from distributions import GaussianNLL

            return GaussianNLL()
        from distributions import StudentTNLL

        return StudentTNLL(nu=self.config.student_t_dof)

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


class O3RepresentationCompiler:
    r"""Lower typed distribution/operator plans to an executable O(3) model.

    ``V + Sym^2(V)`` is retained as the unrestricted diagnostic reference.
    The selected operator family supplies its own active parameter expression;
    only that active expression is a compilation gate.
    """

    def __init__(
        self,
        output: O3IrrepsSpec | o3.Irreps | str,
        config: CompilerConfig | None = None,
        *,
        cartesian_formula: str | None = None,
        graph_structure: EquivariantOutputGraph | None = None,
    ):
        if cartesian_formula is not None:
            if isinstance(output, O3IrrepsSpec):
                raise ValueError("pass either output or cartesian_formula, not both")
            self.output_spec = O3IrrepsSpec.from_cartesian(cartesian_formula)
        elif isinstance(output, O3IrrepsSpec):
            self.output_spec = output
        else:
            self.output_spec = O3IrrepsSpec(o3.Irreps(output))
        self.config = config or CompilerConfig()
        self.graph_structure = graph_structure
        if (
            graph_structure is not None
            and graph_structure.output_irreps != self.output_spec.irreps
        ):
            raise ValueError(
                f"graph output {graph_structure.output_irreps} does not match "
                f"compiled output {self.output_spec.irreps}"
            )

    @classmethod
    def from_cartesian(
        cls, formula: str, config: CompilerConfig | None = None
    ) -> "O3RepresentationCompiler":
        return cls(o3.Irreps("0e"), config, cartesian_formula=formula)

    @classmethod
    def for_graph(
        cls,
        graph_structure: EquivariantOutputGraph,
        config: CompilerConfig | None = None,
    ) -> "O3RepresentationCompiler":
        """Compile repeated node variables with graph-structured precision."""
        return cls(
            graph_structure.output_irreps,
            config or CompilerConfig(covariance="graph"),
            graph_structure=graph_structure,
        )

    def _output_expression(self):
        from representations.representation_ir import IrrepsExpr

        return IrrepsExpr(self.output_spec.irreps, "location")

    def _legacy_operator_family(self) -> OperatorFamilyPlan:
        """Adapt the pre-IR low-level config to an operator-family plugin."""
        from equivcompiler.policies import (
            FullCovariance,
            GraphPrecision,
            IsotypicBlockCovariance,
            LowRankCovariance,
        )

        mode = self.config.covariance
        if mode == "auto":
            full = FullCovariance().compile(self.output_spec)
            graph = (
                GraphPrecision(self.graph_structure).compile(self.output_spec)
                if self.graph_structure is not None
                else None
            )
            low_rank = LowRankCovariance(self.config.low_rank).compile(self.output_spec)
            if full.parameter_count <= self.config.parameter_budget:
                return full
            if graph is not None and graph.parameter_count <= self.config.parameter_budget:
                return graph
            if low_rank.parameter_count <= self.config.parameter_budget:
                return low_rank
            return IsotypicBlockCovariance().compile(self.output_spec)
        if mode == "full":
            return FullCovariance().compile(self.output_spec)
        if mode == "low_rank":
            return LowRankCovariance(self.config.low_rank).compile(self.output_spec)
        if mode == "block":
            return IsotypicBlockCovariance().compile(self.output_spec)
        if mode == "graph":
            if self.graph_structure is None:
                raise ValueError("graph covariance requires graph_structure")
            return GraphPrecision(self.graph_structure).compile(self.output_spec)
        raise AssertionError(f"unhandled covariance mode: {mode}")

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
        seed_irreps: o3.Irreps,
        *,
        operator_family: OperatorFamilyPlan | None = None,
        canonical_reachability: O3ReachabilityAnalysis | None = None,
        active_reachability: O3ReachabilityAnalysis | None = None,
    ) -> O3Compilation:
        seed = o3.Irreps(seed_irreps)
        output_irreps = self.output_spec.irreps
        covariance_irreps = self.output_spec.symmetric_square_irreps
        canonical_target = direct_sum_irreps(output_irreps, covariance_irreps)
        if operator_family is None:
            operator_family = self._legacy_operator_family()
        active_target = operator_family.active_expression(
            self._output_expression()
        ).decompose_o3().irreps

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

        stf_supported = (
            operator_family.kind == "full"
            and is_rank2_stf_output(output_irreps)
            and active_plan.depth == 1
        )
        backend = self.config.backend
        if backend == "auto":
            backend = "cartesian_stf" if stf_supported else "spherical_cg"
        elif backend == "cartesian_stf" and not stf_supported:
            raise CompilationError(
                CompilationCertificate(
                    code="backend_incompatible",
                    status="failure",
                    message=(
                        "cartesian_stf requires full covariance, V=0e+2e, and a "
                        "single direct lifting edge"
                    ),
                    details={
                        "requested_backend": "cartesian_stf",
                        "covariance_mode": operator_family.kind,
                        "output_irreps": str(output_irreps),
                        "active_depth": active_plan.depth,
                    },
                )
            )
        if self.config.stf_contraction_rank is not None and backend != "cartesian_stf":
            raise CompilationError(
                CompilationCertificate(
                    code="contraction_rank_without_lowering",
                    status="failure",
                    message="stf_contraction_rank is only valid when cartesian_stf is selected",
                    details={
                        "selected_backend": backend,
                        "contraction_rank": self.config.stf_contraction_rank,
                    },
                )
            )
        contraction_rank = self.config.stf_contraction_rank
        if backend == "cartesian_stf" and contraction_rank is not None:
            max_exact_rank = max(multiplicity for multiplicity, _ in seed)
            if contraction_rank >= max_exact_rank:
                contraction_rank = None
        backend_exact = backend == "spherical_cg" or contraction_rank is None
        return O3Compilation(
            output_spec=self.output_spec,
            seed_irreps=seed,
            operator_family=operator_family,
            canonical_target_irreps=canonical_target,
            active_target_irreps=active_target,
            canonical_reachability=canonical_reachability,
            active_reachability=active_reachability,
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
        self.operator_runtime = operator_runtime(compilation.operator_family.kind)
        self.operator_runtime.install(self)

    def forward(
        self, node_features: torch.Tensor, batch: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = node_features
        if self.pool:
            if batch is None:
                raise ValueError("batch is required for global output")
            from models.pooling import mean_pool

            hidden = mean_pool(hidden, batch)
        compiled = self.lifting(hidden)
        mean = self.mean_projection(compiled)
        parameters = self.operator_runtime.project(self, compiled)
        return mean, parameters
