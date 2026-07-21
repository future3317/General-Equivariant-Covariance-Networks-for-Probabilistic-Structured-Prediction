"""Independent statistical-family, fidelity, executor, and cost policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from representations import EquivariantOutputGraph, O3IrrepsSpec, irrep_multiplicities
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
        parameter_ref = OperatorIR.parameter("operator")
        symmetric = OperatorIR.symmetric_operator(
            parameter=parameter_ref,
            coordinate_space="output_representation",
            output_irreps=str(output.irreps),
        )
        return OperatorFamilyPlan(
            kind="full",
            parameter_bindings=(
                ParameterBinding("operator", parameter, "covariance_projection"),
            ),
            parameter_count=output.dim * (output.dim + 1) // 2,
            domain="scatter",
            assembly=OperatorIR.spectral_positive(
                symmetric, map="matrix_exponential"
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
        rank = self.rank
        base = IrrepsExpr(output.irreps, "output")
        factor_expression = RepeatedExpr(base, rank)
        scale_expression = TrivialScalarsExpr(1)
        isotropic = OperatorIR.positive_scalar_identity(
            OperatorIR.parameter("scale"), dimension=output.dim
        )
        gram = OperatorIR.gram(
            OperatorIR.equivariant_factor(
                OperatorIR.parameter("factor"),
                rank=rank,
                output_irreps=str(output.irreps),
            )
        )
        return OperatorFamilyPlan(
            kind="low_rank",
            parameter_bindings=(
                ParameterBinding(
                    "factor", factor_expression, "covariance_projection"
                ),
                ParameterBinding("scale", scale_expression, "scale_projection"),
            ),
            parameter_count=output.dim * rank + 1,
            domain="scatter",
            assembly=OperatorIR.add(isotropic, gram),
            relation_to_full=(
                FamilyRelation.EQUAL_TO_FULL
                if rank >= output.dim
                else FamilyRelation.STRICT_SUBSET
            ),
            rank=rank,
            restriction=(
                None if rank >= output.dim else "rank_r_plus_isotropic_scatter"
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
        blocks = []
        cursor = 0
        for irrep, multiplicity in multiplicities.items():
            block_count = multiplicity * (multiplicity + 1) // 2
            blocks.append(
                OperatorIR.kronecker_identity(
                    OperatorIR.cholesky_positive(
                        OperatorIR.parameter(
                            "blocks", start=cursor, stop=cursor + block_count
                        ),
                        dimension=multiplicity,
                    ),
                    irrep=str(irrep),
                    irrep_dimension=irrep.dim,
                    multiplicity=multiplicity,
                )
            )
            cursor += block_count
        return OperatorFamilyPlan(
            kind="block",
            parameter_bindings=(
                ParameterBinding(
                    "blocks", TrivialScalarsExpr(count), "covariance_projection"
                ),
            ),
            parameter_count=count,
            domain="scatter",
            assembly=OperatorIR.direct_sum(*blocks),
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
        local_operator = SymmetricSquareExpr(local)
        local_operator_irreps = str(local_operator.decompose_o3().irreps)
        parameter = RepeatedExpr(local_operator, self.graph.num_potentials)
        local_count = self.graph.block_dim * (self.graph.block_dim + 1) // 2
        unary_stop = self.graph.num_nodes * local_count
        local_unary = OperatorIR.spectral_positive(
            OperatorIR.symmetric_operator(
                parameter=OperatorIR.parameter(
                    "potentials",
                    start=0,
                    stop=unary_stop,
                    coordinate_layout="repeated_irrep",
                    unit_irreps=local_operator_irreps,
                    copies=self.graph.num_potentials,
                ),
                coordinate_space="graph_local",
                role="unary",
                irrep=str(self.graph.node_irrep),
                copies=self.graph.num_nodes,
            ),
            map="matrix_exponential",
        )
        unary = OperatorIR.direct_sum(
            local_unary,
            copies=self.graph.num_nodes,
        )
        local_factor = OperatorIR.spectral_positive(
            OperatorIR.symmetric_operator(
                parameter=OperatorIR.parameter(
                        "potentials",
                        start=unary_stop,
                        stop=self.graph.num_potentials * local_count,
                        coordinate_layout="repeated_irrep",
                        unit_irreps=local_operator_irreps,
                        copies=self.graph.num_potentials,
                    ),
                coordinate_space="graph_local",
                role="factor",
                irrep=str(self.graph.node_irrep),
                copies=self.graph.num_edges,
            ),
            map="matrix_exponential",
        )
        factor = OperatorIR.direct_sum(
            local_factor,
            copies=self.graph.num_edges,
        )
        pullback = OperatorIR.pullback(
            factor, intertwiner="homogeneous_graph_coboundary"
        )
        return OperatorFamilyPlan(
            kind="graph",
            parameter_bindings=(
                ParameterBinding(
                    "potentials", parameter, "covariance_projection"
                ),
            ),
            parameter_count=local_count * self.graph.num_potentials,
            domain="precision",
            assembly=OperatorIR.add(unary, pullback),
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


class FamilyCostModel(ABC):
    """Explicit cost objective used to compare feasible family plans."""

    @abstractmethod
    def score(self, family: OperatorFamilyPlan) -> tuple[float, ...]:
        """Return a lexicographically minimized score."""

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return stable cost-model semantics."""


@dataclass(frozen=True)
class MinParameterCount(FamilyCostModel):
    """Choose the feasible candidate with the fewest emitted coordinates."""

    def score(self, family: OperatorFamilyPlan) -> tuple[float, ...]:
        return (float(family.parameter_count),)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "min_parameter_count"}


@dataclass(frozen=True)
class AutoBudget:
    """Cost-based budget selection over explicitly supplied candidates."""

    max_parameters: int
    candidates: tuple[OperatorFamilySpec, ...]
    objective: FamilyCostModel = MinParameterCount()

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
            "objective": self.objective.as_dict(),
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
class ExecutionSignature:
    """Exact semantic, software, device, and tensor contract for autotuning."""

    semantic_plan_hash: str
    feature_fingerprint: str
    active_plan_hash: str
    operator_program_hash: str
    batch_shape: tuple[int, ...]
    dtype: str
    device: str
    device_uuid: str
    phase: Literal["forward", "forward_backward"]
    compilation_mode: Literal["eager", "torch_compile"]
    software_fingerprint: str
    tensor_layout: str = "contiguous"
    requires_input_grad: bool = True
    requires_parameter_grad: bool = True

    def __post_init__(self) -> None:
        hashes = (
            self.semantic_plan_hash,
            self.feature_fingerprint,
            self.active_plan_hash,
            self.operator_program_hash,
            self.device_uuid,
            self.software_fingerprint,
        )
        if any(not item for item in hashes):
            raise ValueError("execution signature fingerprints must not be empty")
        if not self.batch_shape or any(size < 1 for size in self.batch_shape):
            raise ValueError("batch_shape must contain positive dimensions")

    @property
    def batch_size(self) -> int:
        return self.batch_shape[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_plan_hash": self.semantic_plan_hash,
            "feature_fingerprint": self.feature_fingerprint,
            "active_plan_hash": self.active_plan_hash,
            "operator_program_hash": self.operator_program_hash,
            "batch_shape": list(self.batch_shape),
            "dtype": self.dtype,
            "device": self.device,
            "device_uuid": self.device_uuid,
            "phase": self.phase,
            "compilation_mode": self.compilation_mode,
            "software_fingerprint": self.software_fingerprint,
            "tensor_layout": self.tensor_layout,
            "requires_input_grad": self.requires_input_grad,
            "requires_parameter_grad": self.requires_parameter_grad,
        }


@dataclass(frozen=True)
class ExecutorMeasurement:
    """One measured latency used by a cost policy."""

    executor: ExecutorName
    signature: ExecutionSignature
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

    signature: ExecutionSignature
    measurements: tuple[ExecutorMeasurement, ...]

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError("latency selection requires measurements")
        if any(item.signature != self.signature for item in self.measurements):
            raise ValueError("all measurements must match the requested signature")


CostPolicy = PreferExecutor | MinimizeLatency
