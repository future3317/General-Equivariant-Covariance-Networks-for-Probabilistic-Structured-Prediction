"""Coordinate-independent representation expressions and typed layouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from compatibility.e3nn import o3

from representations.adaptive_lifting import direct_sum_irreps
from representations.irrep_layout import RepeatedIrrepLayout
from representations.o3_irreps import O3IrrepsSpec


class RepExpr(ABC):
    """Symbolic representation expression before backend decomposition."""

    @abstractmethod
    def decompose_o3(self) -> "DecomposedRep":
        """Decompose this expression with the released O(3) backend."""

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return a stable, serializable semantic record."""


@dataclass(frozen=True)
class DecomposedRep:
    """Irrep/multiplicity decomposition, independent of a concrete layout."""

    group: str
    irreps: o3.Irreps
    multiplicity_basis: str = "abstract_multiplicity_space"

    def __post_init__(self) -> None:
        object.__setattr__(self, "irreps", o3.Irreps(self.irreps))

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "irreps": str(self.irreps),
            "dimension": self.irreps.dim,
            "multiplicity_basis": self.multiplicity_basis,
        }


@dataclass(frozen=True)
class IrrepsExpr(RepExpr):
    """Atomic O(3) representation used by the current backend."""

    irreps: o3.Irreps
    semantic_name: str = "representation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "irreps", o3.Irreps(self.irreps))

    def decompose_o3(self) -> DecomposedRep:
        return DecomposedRep("O3", self.irreps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "atomic_representation",
            "semantic_name": self.semantic_name,
            "declared_irreps": str(self.irreps),
        }


@dataclass(frozen=True)
class DirectSumExpr(RepExpr):
    """Symbolic direct sum; decomposition is a separate backend pass."""

    terms: tuple[RepExpr, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("a direct sum must contain at least one term")

    def decompose_o3(self) -> DecomposedRep:
        irreps = direct_sum_irreps(*(term.decompose_o3().irreps for term in self.terms))
        return DecomposedRep("O3", irreps)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "direct_sum", "terms": [term.as_dict() for term in self.terms]}


@dataclass(frozen=True)
class SymmetricSquareExpr(RepExpr):
    """Symbolic symmetric square of an inner-product representation."""

    operand: RepExpr

    def decompose_o3(self) -> DecomposedRep:
        operand = self.operand.decompose_o3().irreps
        return DecomposedRep("O3", O3IrrepsSpec(operand).symmetric_square_irreps)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "symmetric_square", "operand": self.operand.as_dict()}


@dataclass(frozen=True)
class RepeatedExpr(RepExpr):
    """Ordered copies of a representation, retaining multiplicity semantics."""

    operand: RepExpr
    copies: int

    def __post_init__(self) -> None:
        if self.copies < 1:
            raise ValueError("copies must be positive")

    def decompose_o3(self) -> DecomposedRep:
        base = self.operand.decompose_o3().irreps
        return DecomposedRep(
            "O3", RepeatedIrrepLayout(base, self.copies).expanded_irreps
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "repeated_representation",
            "copies": self.copies,
            "operand": self.operand.as_dict(),
        }


@dataclass(frozen=True)
class TrivialScalarsExpr(RepExpr):
    """Invariant scalar coordinates such as multiplicity-space SPD blocks."""

    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be positive")

    def decompose_o3(self) -> DecomposedRep:
        return DecomposedRep("O3", o3.Irreps(f"{self.count}x0e"))

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "trivial_scalars", "count": self.count}


@dataclass(frozen=True)
class InvariantMetricSpec:
    """Invariant inner product carried by a representation type."""

    kind: str = "orthonormal_identity"
    gram_matrix_id: str | None = None

    @property
    def is_orthonormal(self) -> bool:
        return self.kind == "orthonormal_identity"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "is_orthonormal": self.is_orthonormal,
            "gram_matrix_id": self.gram_matrix_id,
        }


@dataclass(frozen=True)
class CoordinateSpec:
    """Concrete basis and memory layout used by an executor."""

    basis_id: str
    layout: str
    metric: InvariantMetricSpec = InvariantMetricSpec()

    def as_dict(self) -> dict[str, Any]:
        return {
            "basis_id": self.basis_id,
            "layout": self.layout,
            "metric": self.metric.as_dict(),
        }


@dataclass(frozen=True)
class InnerProductRep:
    """A representation together with its invariant metric."""

    group: str
    representation: RepExpr
    metric: InvariantMetricSpec = InvariantMetricSpec()

    def decompose_o3(self) -> DecomposedRep:
        if self.group != "O3":
            raise ValueError("the released decomposition backend currently supports O3")
        return self.representation.decompose_o3()

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "representation": self.representation.as_dict(),
            "metric": self.metric.as_dict(),
        }
