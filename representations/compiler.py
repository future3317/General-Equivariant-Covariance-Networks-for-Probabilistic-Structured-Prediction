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


CovarianceMode = Literal["auto", "full", "block", "low_rank", "graph"]
OutputScope = Literal["global", "dense"]
ObjectiveName = Literal["gaussian", "student_t"]


@dataclass(frozen=True)
class CompilerConfig:
    """Architecture and probabilistic choices made by the compiler."""

    covariance: CovarianceMode = "auto"
    output_scope: OutputScope = "global"
    objective: ObjectiveName = "gaussian"
    parameter_budget: int = 192
    low_rank: int = 8
    student_t_dof: float = 5.0

    def __post_init__(self):
        if self.covariance not in {"auto", "full", "block", "low_rank", "graph"}:
            raise ValueError(f"unknown covariance mode: {self.covariance}")
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


def _factor_irreps(output_irreps: o3.Irreps, rank: int) -> o3.Irreps:
    return o3.Irreps(
        [(multiplicity * rank, irrep) for multiplicity, irrep in output_irreps]
    )


def _repeat_irreps(irreps: o3.Irreps, copies: int) -> o3.Irreps:
    return o3.Irreps(
        [(multiplicity * copies, irrep) for multiplicity, irrep in irreps]
    )


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
    config: CompilerConfig
    graph_structure: EquivariantOutputGraph | None = None

    @property
    def mean_irreps(self) -> o3.Irreps:
        return self.output_spec.irreps

    @property
    def covariance_irreps(self) -> o3.Irreps:
        return self.output_spec.symmetric_square().operator_irreps

    def as_dict(self) -> dict:
        return {
            "output_irreps": str(self.output_spec.irreps),
            "cartesian_formula": self.output_spec.cartesian_formula,
            "seed_irreps": str(self.seed_irreps),
            "canonical_target_irreps": str(self.canonical_target_irreps),
            "active_target_irreps": str(self.active_target_irreps),
            "covariance_mode": self.covariance_mode,
            "covariance_rank": self.covariance_rank,
            "covariance_parameter_count": self.covariance_parameter_count,
            "canonical_covariance_parameter_count": (
                self.output_spec.dim * (self.output_spec.dim + 1) // 2
            ),
            "active_covariance_parameter_count": self.covariance_parameter_count,
            "output_scope": self.config.output_scope,
            "objective": self.config.objective,
            "canonical_lifting": self.canonical_plan.as_dict(),
            "active_lifting": self.active_plan.as_dict(),
            "graph_structure": (
                self.graph_structure.as_dict()
                if self.graph_structure is not None
                else None
            ),
        }

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

    def compile(self, seed_irreps: o3.Irreps) -> O3Compilation:
        seed = o3.Irreps(seed_irreps)
        output_irreps = self.output_spec.irreps
        covariance_irreps = self.output_spec.symmetric_square().operator_irreps
        canonical_target = direct_sum_irreps(output_irreps, covariance_irreps)
        canonical_plan = plan_lifting_graph(seed, canonical_target)

        dim = self.output_spec.dim
        full_parameters = dim * (dim + 1) // 2
        rank = min(self.config.low_rank, dim)
        low_rank_parameters = dim * rank + 1
        block_parameters = _isotypic_parameter_count(output_irreps)
        graph_parameters: int | None = None
        graph_operator_irreps: o3.Irreps | None = None
        if self.graph_structure is not None:
            local_spec = O3IrrepsSpec(o3.Irreps([(1, self.graph_structure.node_irrep)]))
            graph_operator_irreps = local_spec.symmetric_square().operator_irreps
            graph_parameters = (
                local_spec.symmetric_square().operator_dim
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
            factors = _factor_irreps(output_irreps, rank)
            active_target = direct_sum_irreps(
                output_irreps, factors, o3.Irreps("1x0e")
            )
            parameter_count = low_rank_parameters
            selected_rank = rank
        elif mode == "graph":
            if self.graph_structure is None or graph_parameters is None:
                raise ValueError("graph covariance requires graph_structure")
            assert graph_operator_irreps is not None
            potential_irreps = _repeat_irreps(
                graph_operator_irreps,
                self.graph_structure.num_potentials,
            )
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
            config=self.config,
            graph_structure=self.graph_structure if mode == "graph" else None,
        )


class O3CompiledOutputHead(torch.nn.Module):
    """Shared lifting trunk with compiled mean and covariance projections."""

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
        )
        self.mean_projection = o3.Linear(active_irreps, self.output_spec.irreps)

        if compilation.covariance_mode == "full":
            self.operator_basis = self.output_spec.symmetric_square()
            self.covariance_projection = o3.Linear(
                active_irreps, self.operator_basis.operator_irreps
            )
        elif compilation.covariance_mode == "low_rank":
            self.factor_irreps = _factor_irreps(
                self.output_spec.irreps, int(compilation.covariance_rank)
            )
            self.covariance_projection = o3.Linear(active_irreps, self.factor_irreps)
            self.scale_projection = o3.Linear(active_irreps, o3.Irreps("1x0e"))
            self._factor_slices = self._build_factor_slices(
                self.output_spec.irreps, int(compilation.covariance_rank)
            )
        elif compilation.covariance_mode == "graph":
            if compilation.graph_structure is None:
                raise RuntimeError("graph compilation is missing its graph structure")
            local_irreps = o3.Irreps([(1, compilation.graph_structure.node_irrep)])
            self.operator_basis = O3IrrepsSpec(local_irreps).symmetric_square()
            self.potential_irreps = _repeat_irreps(
                self.operator_basis.operator_irreps,
                compilation.graph_structure.num_potentials,
            )
            self.covariance_projection = o3.Linear(
                active_irreps, self.potential_irreps
            )
            self._potential_slices = self._build_repeated_slices(
                self.operator_basis.operator_irreps,
                compilation.graph_structure.num_potentials,
            )
        else:
            count = compilation.covariance_parameter_count
            self.covariance_projection = o3.Linear(
                active_irreps, o3.Irreps(f"{count}x0e")
            )

    @staticmethod
    def _build_factor_slices(
        output_irreps: o3.Irreps, rank: int
    ) -> list[list[tuple[int, int]]]:
        slices = [[] for _ in range(rank)]
        cursor = 0
        for multiplicity, irrep in output_irreps:
            for copy in range(multiplicity * rank):
                rank_slot = copy % rank
                start = cursor + copy * irrep.dim
                slices[rank_slot].append((start, start + irrep.dim))
            cursor += multiplicity * rank * irrep.dim
        return slices

    def _pack_factors(self, coefficients: torch.Tensor) -> torch.Tensor:
        rank = int(self.compilation.covariance_rank)
        factor = coefficients.new_empty(
            (*coefficients.shape[:-1], self.output_spec.dim, rank)
        )
        for rank_slot, source_slices in enumerate(self._factor_slices):
            row = 0
            for start, end in source_slices:
                width = end - start
                factor[..., row : row + width, rank_slot] = coefficients[..., start:end]
                row += width
        return factor

    @staticmethod
    def _build_repeated_slices(
        irreps: o3.Irreps, copies: int
    ) -> list[list[tuple[int, int]]]:
        slices = [[] for _ in range(copies)]
        cursor = 0
        for multiplicity, irrep in irreps:
            for repeated_copy in range(multiplicity * copies):
                copy_slot = repeated_copy % copies
                start = cursor + repeated_copy * irrep.dim
                slices[copy_slot].append((start, start + irrep.dim))
            cursor += multiplicity * copies * irrep.dim
        return slices

    def _pack_repeated_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        packed = coefficients.new_empty(
            (
                *coefficients.shape[:-1],
                len(self._potential_slices),
                self.operator_basis.operator_dim,
            )
        )
        for copy_slot, source_slices in enumerate(self._potential_slices):
            target_cursor = 0
            for start, end in source_slices:
                width = end - start
                packed[..., copy_slot, target_cursor : target_cursor + width] = (
                    coefficients[..., start:end]
                )
                target_cursor += width
        return packed

    def forward(
        self, node_features: torch.Tensor, batch: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = node_features
        if self.pool:
            if batch is None:
                raise ValueError("batch is required for global output")
            if batch.numel() == 0:
                raise ValueError("batch must not be empty")
            graph_count = int(batch.max().item()) + 1
            pooled = hidden.new_zeros((graph_count, hidden.shape[-1]))
            pooled.index_add_(0, batch, hidden)
            counts = torch.bincount(batch, minlength=graph_count).to(hidden.dtype)
            hidden = pooled / counts.clamp_min(1).unsqueeze(-1)
        compiled = self.lifting(hidden)
        mean = self.mean_projection(compiled)

        if self.compilation.covariance_mode == "full":
            coefficients = self.covariance_projection(compiled)
            parameters = self.operator_basis.assemble(coefficients)
        elif self.compilation.covariance_mode == "low_rank":
            factors = self._pack_factors(self.covariance_projection(compiled))
            log_sigma2 = self.scale_projection(compiled)
            parameters = torch.cat(
                [factors.flatten(start_dim=-2), log_sigma2], dim=-1
            )
        elif self.compilation.covariance_mode == "graph":
            coefficients = self.covariance_projection(compiled)
            parameters = self.operator_basis.assemble(
                self._pack_repeated_coefficients(coefficients)
            )
        else:
            parameters = self.covariance_projection(compiled)
        return mean, parameters
