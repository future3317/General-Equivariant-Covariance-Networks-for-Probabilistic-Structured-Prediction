"""Orthogonal statistical-family and execution-lowering policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from representations import EquivariantOutputGraph


FamilyName = Literal["full", "graph", "low_rank", "block"]


@dataclass(frozen=True)
class FullCovariance:
    """Request the canonical unrestricted SPD family."""


@dataclass(frozen=True)
class LowRankCovariance:
    """Request the exact low-rank-plus-isotropic subfamily."""

    rank: int = 8

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")


@dataclass(frozen=True)
class IsotypicBlockCovariance:
    """Request invariant multiplicity-space SPD blocks."""


@dataclass(frozen=True)
class GraphPrecision:
    """Request a fixed-graph precision subfamily."""

    graph: EquivariantOutputGraph


@dataclass(frozen=True)
class AutoBudget:
    """Authorize budget-based selection among explicitly allowed families."""

    budget: int = 192
    low_rank: int = 8
    graph: EquivariantOutputGraph | None = None
    allowed_families: tuple[FamilyName, ...] = (
        "full",
        "graph",
        "low_rank",
        "block",
    )

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be positive")
        if self.low_rank < 1:
            raise ValueError("low_rank must be positive")
        if not self.allowed_families:
            raise ValueError("allowed_families must not be empty")
        invalid = set(self.allowed_families) - {"full", "graph", "low_rank", "block"}
        if invalid:
            raise ValueError(f"unknown covariance families: {sorted(invalid)}")
        if len(set(self.allowed_families)) != len(self.allowed_families):
            raise ValueError("allowed_families must not contain duplicates")


CovariancePolicy = (
    FullCovariance
    | LowRankCovariance
    | IsotypicBlockCovariance
    | GraphPrecision
    | AutoBudget
)


@dataclass(frozen=True)
class ExactOnly:
    """Allow only algebraically exact lowering of the selected family."""

    backend: Literal["auto", "spherical_cg", "cartesian_stf"] = "auto"


@dataclass(frozen=True)
class TruncatedMultiplicityRank:
    """Explicitly authorize approximate multiplicity-rank contraction."""

    rank: int
    backend: Literal["cartesian_stf"] = "cartesian_stf"

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")


LoweringPolicy = ExactOnly | TruncatedMultiplicityRank
