"""Independent statistical-family, fidelity, executor, and cost policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from representations import EquivariantOutputGraph, O3IrrepsSpec, irrep_multiplicities
from representations.operator_ir import FamilyRelation, OperatorFamilyPlan, OperatorIR
from representations.representation_ir import (
    DirectSumExpr,
    IrrepsExpr,
    RepeatedExpr,
    SymmetricSquareExpr,
    TrivialScalarsExpr,
)


ExecutorName = Literal["spherical_cg", "cartesian_stf"]


class OperatorFamilySpec(ABC):
    """Plugin interface from an output representation to an operator program."""

    @abstractmethod
    def compile(self, output: O3IrrepsSpec) -> OperatorFamilyPlan:
        """Return parameter representation, assembly IR, and family certificate."""

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return a stable policy record."""


@dataclass(frozen=True)
class FullCovariance(OperatorFamilySpec):
    """Canonical unrestricted SPD scatter family."""

    def compile(self, output: O3IrrepsSpec) -> OperatorFamilyPlan:
        base = IrrepsExpr(output.irreps, "output")
        parameter = SymmetricSquareExpr(base)
        return OperatorFamilyPlan(
            kind="full",
            parameter_expression=parameter,
            parameter_count=output.dim * (output.dim + 1) // 2,
            domain="scatter",
            assembly=OperatorIR.node(
                "spectral_positive",
                OperatorIR.node("symmetric_operator", positivity="unspecified"),
                map="matrix_exponential",
                positivity="spd",
            ),
            relation_to_full=FamilyRelation.EQUAL_TO_FULL,
        )

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "full_covariance"}


@dataclass(frozen=True)
class LowRankCovariance(OperatorFamilySpec):
    """Exact low-rank-plus-isotropic scatter subfamily."""

    rank: int = 8

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")

    def compile(self, output: O3IrrepsSpec) -> OperatorFamilyPlan:
        rank = min(self.rank, output.dim)
        base = IrrepsExpr(output.irreps, "output")
        parameter = DirectSumExpr((RepeatedExpr(base, rank), TrivialScalarsExpr(1)))
        isotropic = OperatorIR.node("positive_scalar_identity", positivity="spd")
        gram = OperatorIR.node(
            "gram",
            OperatorIR.node("equivariant_factor", rank=rank),
            positivity="psd",
        )
        return OperatorFamilyPlan(
            kind="low_rank",
            parameter_expression=parameter,
            parameter_count=output.dim * rank + 1,
            domain="scatter",
            assembly=OperatorIR.node("add", isotropic, gram, positivity="spd"),
            relation_to_full=(
                FamilyRelation.EQUAL_TO_FULL
                if rank == output.dim
                else FamilyRelation.STRICT_SUBSET
            ),
            rank=rank,
            restriction=(
                None if rank == output.dim else "rank_r_plus_isotropic_scatter"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "low_rank_plus_isotropic", "rank": self.rank}


@dataclass(frozen=True)
class IsotypicBlockCovariance(OperatorFamilySpec):
    """Invariant multiplicity-space SPD block subfamily."""

    def compile(self, output: O3IrrepsSpec) -> OperatorFamilyPlan:
        multiplicities = irrep_multiplicities(output.irreps)
        count = sum(value * (value + 1) // 2 for value in multiplicities.values())
        equals_full = (
            len(multiplicities) == 1
            and next(iter(multiplicities)).dim == 1
        )
        blocks = tuple(
            OperatorIR.node(
                "kronecker_identity",
                OperatorIR.node(
                    "spectral_positive",
                    map="multiplicity_cholesky",
                    positivity="spd",
                ),
                irrep=str(irrep),
                irrep_dimension=irrep.dim,
                multiplicity=multiplicity,
                positivity="spd",
            )
            for irrep, multiplicity in multiplicities.items()
        )
        return OperatorFamilyPlan(
            kind="block",
            parameter_expression=TrivialScalarsExpr(count),
            parameter_count=count,
            domain="scatter",
            assembly=OperatorIR.node("direct_sum", *blocks, positivity="spd"),
            relation_to_full=(
                FamilyRelation.EQUAL_TO_FULL
                if equals_full
                else FamilyRelation.STRICT_SUBSET
            ),
            restriction=None if equals_full else "isotypic_multiplicity_blocks",
        )

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "isotypic_block_covariance"}


@dataclass(frozen=True)
class GraphPrecision(OperatorFamilySpec):
    """Fixed homogeneous factor-graph precision subfamily."""

    graph: EquivariantOutputGraph

    def compile(self, output: O3IrrepsSpec) -> OperatorFamilyPlan:
        if self.graph.output_irreps != output.irreps:
            raise ValueError(
                f"graph output {self.graph.output_irreps} does not match {output.irreps}"
            )
        local = IrrepsExpr(f"1x{self.graph.node_irrep}", "graph_residual")
        parameter = RepeatedExpr(
            SymmetricSquareExpr(local), self.graph.num_potentials
        )
        local_count = self.graph.block_dim * (self.graph.block_dim + 1) // 2
        unary = OperatorIR.node(
            "direct_sum",
            OperatorIR.node("local_spectral_positive", role="unary", positivity="spd"),
            copies=self.graph.num_nodes,
            positivity="spd",
        )
        factor = OperatorIR.node(
            "direct_sum",
            OperatorIR.node("local_spectral_positive", role="factor", positivity="spd"),
            copies=self.graph.num_edges,
            positivity="spd",
        )
        pullback = OperatorIR.node(
            "pullback",
            factor,
            map="homogeneous_graph_coboundary",
            positivity="psd",
        )
        return OperatorFamilyPlan(
            kind="graph",
            parameter_expression=parameter,
            parameter_count=local_count * self.graph.num_potentials,
            domain="precision",
            assembly=OperatorIR.node("add", unary, pullback, positivity="spd"),
            relation_to_full=(
                FamilyRelation.EQUAL_TO_FULL
                if self.graph.num_nodes == 1
                else FamilyRelation.STRICT_SUBSET
            ),
            graph=self.graph,
            restriction=(
                None
                if self.graph.num_nodes == 1
                else "fixed_skeleton_precision_cone"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "graph_precision", "graph": self.graph.as_dict()}


@dataclass(frozen=True)
class MinParameterCount:
    """Choose the feasible candidate with the fewest emitted coordinates."""


@dataclass(frozen=True)
class AutoBudget:
    """Cost-based budget selection over explicitly supplied candidates."""

    max_parameters: int
    candidates: tuple[OperatorFamilySpec, ...]
    objective: MinParameterCount = MinParameterCount()

    def __post_init__(self) -> None:
        if self.max_parameters < 1:
            raise ValueError("max_parameters must be positive")
        if not self.candidates:
            raise ValueError("candidates must not be empty")

    @property
    def budget(self) -> int:
        return self.max_parameters

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "auto_budget",
            "max_parameters": self.max_parameters,
            "objective": "min_parameter_count",
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class FirstFeasible:
    """User-declared priority order among mutually incomparable families."""

    max_parameters: int
    priority: tuple[OperatorFamilySpec, ...]

    def __post_init__(self) -> None:
        if self.max_parameters < 1:
            raise ValueError("max_parameters must be positive")
        if not self.priority:
            raise ValueError("priority must not be empty")

    @property
    def budget(self) -> int:
        return self.max_parameters

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "first_feasible",
            "max_parameters": self.max_parameters,
            "selection_semantics": "user_declared_priority",
            "priority": [candidate.as_dict() for candidate in self.priority],
        }


CovariancePolicy = OperatorFamilySpec | AutoBudget | FirstFeasible


@dataclass(frozen=True)
class ExactOnly:
    """Allow only algebraically exact realization of the active family."""


@dataclass(frozen=True)
class TruncatedMultiplicityRank:
    """Explicitly authorize approximate multiplicity-rank contraction."""

    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")


FidelityPolicy = ExactOnly | TruncatedMultiplicityRank


@dataclass(frozen=True)
class ExactExecutorCandidates:
    """Enumerate exact executors without asserting a performance order."""

    candidates: tuple[ExecutorName, ...] = ("spherical_cg", "cartesian_stf")

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("executor candidates must not be empty")
        invalid = set(self.candidates) - {"spherical_cg", "cartesian_stf"}
        if invalid:
            raise ValueError(f"unknown executor candidates: {sorted(invalid)}")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("executor candidates must not contain duplicates")


@dataclass(frozen=True)
class SpecificExecutor:
    """Require one executor by name."""

    name: ExecutorName

    def __post_init__(self) -> None:
        if self.name not in {"spherical_cg", "cartesian_stf"}:
            raise ValueError(f"unknown executor: {self.name}")


ExecutorPolicy = ExactExecutorCandidates | SpecificExecutor


@dataclass(frozen=True)
class ShapeSignature:
    """Performance signature for measured executor selection."""

    batch_size: int
    feature_dimension: int
    dtype: str
    device: str
    phase: Literal["forward", "forward_backward"] = "forward_backward"

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.feature_dimension < 1:
            raise ValueError("shape dimensions must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "feature_dimension": self.feature_dimension,
            "dtype": self.dtype,
            "device": self.device,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class ExecutorMeasurement:
    """One measured latency used by a cost policy."""

    executor: ExecutorName
    signature: ShapeSignature
    median_ms: float
    iqr_ms: float | None = None

    def __post_init__(self) -> None:
        if self.median_ms < 0 or (self.iqr_ms is not None and self.iqr_ms < 0):
            raise ValueError("latency measurements must be nonnegative")


@dataclass(frozen=True)
class PreferExecutor:
    """Explicit static priority; makes no speed-optimality claim."""

    priority: tuple[ExecutorName, ...] = ("spherical_cg", "cartesian_stf")

    def __post_init__(self) -> None:
        if not self.priority:
            raise ValueError("executor priority must not be empty")
        invalid = set(self.priority) - {"spherical_cg", "cartesian_stf"}
        if invalid:
            raise ValueError(f"unknown executor priority: {sorted(invalid)}")


@dataclass(frozen=True)
class MinimizeLatency:
    """Choose the lowest measured latency for one exact shape signature."""

    signature: ShapeSignature
    measurements: tuple[ExecutorMeasurement, ...]

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError("latency selection requires measurements")
        if any(item.signature != self.signature for item in self.measurements):
            raise ValueError("all measurements must match the requested signature")


CostPolicy = PreferExecutor | MinimizeLatency


# Compatibility name retained for callers that used the old combined policy.
LoweringPolicy = FidelityPolicy
