"""Representation compiler for probabilistic O(3)-equivariant outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from compatibility.e3nn import o3

from representations.adaptive_lifting import (
    O3AdaptiveLifting,
    O3LiftingPlan,
    direct_sum_irreps,
    irrep_multiplicities,
    plan_lifting_graph,
)
from representations.o3_irreps import O3IrrepsSpec
from representations.graph_structure import EquivariantOutputGraph
from representations.irrep_layout import RepeatedIrrepLayout
from representations.cartesian_stf import (
    Rank2CartesianSTFOperatorBasis,
    is_rank2_stf_output,
)
from representations.diagnostics import CompilationCertificate, CompilationError


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
    covariance_mode: Literal["full", "block", "low_rank", "graph"]
    covariance_rank: int | None
    covariance_parameter_count: int
    canonical_target_irreps: o3.Irreps
    active_target_irreps: o3.Irreps
    canonical_plan: O3LiftingPlan
    active_plan: O3LiftingPlan
    backend: Literal["spherical_cg", "cartesian_stf"]
    backend_exact: bool
    stf_contraction_rank: int | None
    config: CompilerConfig
    graph_structure: EquivariantOutputGraph | None = None

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
        if self.covariance_mode == "full":
            from spd_maps import MatrixExponentialMap

            return MatrixExponentialMap()
        if self.covariance_mode == "block":
            from spd_maps import IsotypicBlockMap

            return IsotypicBlockMap(self.output_spec.irreps)
        if self.covariance_mode == "graph":
            from spd_maps import GraphStructuredPrecisionMap

            if self.graph_structure is None:
                raise RuntimeError("graph compilation is missing its graph structure")
            return GraphStructuredPrecisionMap(self.graph_structure)
        from spd_maps import LowRankPlusIsotropicMap

        return LowRankPlusIsotropicMap(
            dim=self.output_spec.dim, rank=int(self.covariance_rank)
        )

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
    r"""Compile ``T(V)=Irreps(V + Sym^2(V))`` and an executable model.

    The canonical plan always covers the complete mean/covariance target.  An
    optional structured covariance choice materializes a cheaper active graph:
    invariant isotypic blocks or an equivariant low-rank factor.  This makes the
    approximation explicit in the compilation report rather than hiding it in
    a hand-written head.
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

    def compile(
        self,
        seed_irreps: o3.Irreps,
        *,
        canonical_plan: O3LiftingPlan | None = None,
    ) -> O3Compilation:
        seed = o3.Irreps(seed_irreps)
        output_irreps = self.output_spec.irreps
        covariance_irreps = self.output_spec.symmetric_square_irreps
        canonical_target = direct_sum_irreps(output_irreps, covariance_irreps)
        if canonical_plan is None:
            canonical_plan = plan_lifting_graph(seed, canonical_target)
        elif (
            canonical_plan.seed_irreps != seed
            or canonical_plan.target_irreps != canonical_target
        ):
            raise CompilationError(
                CompilationCertificate(
                    code="canonical_plan_contract_mismatch",
                    status="failure",
                    message="the supplied canonical lifting plan does not match this compiler contract",
                    details={
                        "expected_seed_irreps": str(seed),
                        "planned_seed_irreps": str(canonical_plan.seed_irreps),
                        "expected_target_irreps": str(canonical_target),
                        "planned_target_irreps": str(canonical_plan.target_irreps),
                    },
                )
            )

        dim = self.output_spec.dim
        full_parameters = dim * (dim + 1) // 2
        rank = min(self.config.low_rank, dim)
        low_rank_parameters = dim * rank + 1
        block_parameters = _isotypic_parameter_count(output_irreps)
        graph_parameters: int | None = None
        graph_operator_irreps: o3.Irreps | None = None
        if self.graph_structure is not None:
            local_spec = O3IrrepsSpec(o3.Irreps([(1, self.graph_structure.node_irrep)]))
            graph_operator_irreps = local_spec.symmetric_square_irreps
            graph_parameters = (
                local_spec.dim * (local_spec.dim + 1) // 2
                * self.graph_structure.num_potentials
            )

        mode = self.config.covariance
        if mode == "auto":
            if full_parameters <= self.config.parameter_budget:
                mode = "full"
            elif (
                graph_parameters is not None
                and graph_parameters <= self.config.parameter_budget
            ):
                mode = "graph"
            elif low_rank_parameters <= self.config.parameter_budget:
                mode = "low_rank"
            else:
                mode = "block"

        if mode == "full":
            active_target = canonical_target
            parameter_count = full_parameters
            selected_rank = None
        elif mode == "low_rank":
            factors = RepeatedIrrepLayout(output_irreps, rank).expanded_irreps
            active_target = direct_sum_irreps(output_irreps, factors, o3.Irreps("1x0e"))
            parameter_count = low_rank_parameters
            selected_rank = rank
        elif mode == "graph":
            if self.graph_structure is None or graph_parameters is None:
                raise ValueError("graph covariance requires graph_structure")
            assert graph_operator_irreps is not None
            potential_irreps = RepeatedIrrepLayout(
                graph_operator_irreps, self.graph_structure.num_potentials
            ).expanded_irreps
            active_target = direct_sum_irreps(output_irreps, potential_irreps)
            parameter_count = graph_parameters
            selected_rank = None
        else:
            active_target = direct_sum_irreps(
                output_irreps, o3.Irreps(f"{block_parameters}x0e")
            )
            parameter_count = block_parameters
            selected_rank = None

        active_plan = (
            canonical_plan
            if active_target == canonical_target
            else plan_lifting_graph(seed, active_target)
        )

        stf_supported = (
            mode == "full"
            and is_rank2_stf_output(output_irreps)
            and canonical_plan.depth == 1
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
                        "covariance_mode": mode,
                        "output_irreps": str(output_irreps),
                        "canonical_depth": canonical_plan.depth,
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
            covariance_mode=mode,
            covariance_rank=selected_rank,
            covariance_parameter_count=parameter_count,
            canonical_target_irreps=canonical_target,
            active_target_irreps=active_target,
            canonical_plan=canonical_plan,
            active_plan=active_plan,
            backend=backend,
            backend_exact=backend_exact,
            stf_contraction_rank=contraction_rank,
            config=self.config,
            graph_structure=self.graph_structure if mode == "graph" else None,
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

        if compilation.covariance_mode == "full":
            self.operator_basis = (
                Rank2CartesianSTFOperatorBasis()
                if compilation.backend == "cartesian_stf"
                else self.output_spec.symmetric_square()
            )
            self.covariance_projection = o3.Linear(
                active_irreps, self.operator_basis.operator_irreps
            )
        elif compilation.covariance_mode == "low_rank":
            self.factor_layout = RepeatedIrrepLayout(
                self.output_spec.irreps, int(compilation.covariance_rank)
            )
            self.factor_irreps = self.factor_layout.expanded_irreps
            self.covariance_projection = o3.Linear(active_irreps, self.factor_irreps)
            self.scale_projection = o3.Linear(active_irreps, o3.Irreps("1x0e"))
        elif compilation.covariance_mode == "graph":
            if compilation.graph_structure is None:
                raise RuntimeError("graph compilation is missing its graph structure")
            local_irreps = o3.Irreps([(1, compilation.graph_structure.node_irrep)])
            self.operator_basis = O3IrrepsSpec(local_irreps).symmetric_square()
            self.potential_layout = RepeatedIrrepLayout(
                self.operator_basis.operator_irreps,
                compilation.graph_structure.num_potentials,
            )
            self.potential_irreps = self.potential_layout.expanded_irreps
            self.covariance_projection = o3.Linear(active_irreps, self.potential_irreps)
        else:
            count = compilation.covariance_parameter_count
            self.covariance_projection = o3.Linear(
                active_irreps, o3.Irreps(f"{count}x0e")
            )

    def _pack_factors(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.factor_layout.pack(coefficients).transpose(-1, -2)

    def _pack_repeated_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.potential_layout.pack(coefficients)

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

        if self.compilation.covariance_mode == "full":
            coefficients = self.covariance_projection(compiled)
            parameters = self.operator_basis.assemble(coefficients)
        elif self.compilation.covariance_mode == "low_rank":
            factors = self._pack_factors(self.covariance_projection(compiled))
            log_sigma2 = self.scale_projection(compiled)
            parameters = torch.cat([factors.flatten(start_dim=-2), log_sigma2], dim=-1)
        elif self.compilation.covariance_mode == "graph":
            coefficients = self.covariance_projection(compiled)
            parameters = self.operator_basis.assemble(
                self._pack_repeated_coefficients(coefficients)
            )
        else:
            parameters = self.covariance_projection(compiled)
        return mean, parameters
