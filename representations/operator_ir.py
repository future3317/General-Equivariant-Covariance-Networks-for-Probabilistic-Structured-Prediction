"""Typed structured-operator intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from compatibility.e3nn import o3

from representations.graph_structure import EquivariantOutputGraph
from representations.representation_ir import DirectSumExpr, RepExpr


class FamilyRelation(str, Enum):
    """Set-theoretic relation between statistical operator families."""

    EQUAL_TO_FULL = "equal_to_full"
    STRICT_SUBSET = "strict_subset"
    INCOMPARABLE = "incomparable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OperatorIR:
    """Composable, certificate-carrying structured operator program."""

    kind: str
    inputs: tuple["OperatorIR", ...] = ()
    attributes: tuple[tuple[str, Any], ...] = ()
    positivity: Literal["spd", "psd", "unspecified"] = "unspecified"
    equivariance: Literal["certified", "required"] = "certified"

    @classmethod
    def node(
        cls,
        kind: str,
        *inputs: "OperatorIR",
        positivity: Literal["spd", "psd", "unspecified"] = "unspecified",
        equivariance: Literal["certified", "required"] = "certified",
        **attributes: Any,
    ) -> "OperatorIR":
        return cls(
            kind=kind,
            inputs=tuple(inputs),
            attributes=tuple(sorted(attributes.items())),
            positivity=positivity,
            equivariance=equivariance,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "inputs": [item.as_dict() for item in self.inputs],
            "attributes": dict(self.attributes),
            "positivity": self.positivity,
            "equivariance": self.equivariance,
        }


@dataclass(frozen=True)
class OperatorFamilyPlan:
    """Semantic operator family before reachability and executable lowering."""

    kind: Literal["full", "block", "low_rank", "graph"]
    parameter_expression: RepExpr
    parameter_count: int
    domain: Literal["scatter", "precision"]
    assembly: OperatorIR
    relation_to_full: FamilyRelation
    rank: int | None = None
    graph: EquivariantOutputGraph | None = None
    restriction: str | None = None
    computational_oracles: tuple[str, ...] = (
        "logdet",
        "quadratic_form",
        "sampling",
    )

    @property
    def parameter_irreps(self) -> o3.Irreps:
        return self.parameter_expression.decompose_o3().irreps

    def active_expression(self, location: RepExpr) -> RepExpr:
        return DirectSumExpr((location, self.parameter_expression))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameter_expression": self.parameter_expression.as_dict(),
            "parameter_representation": self.parameter_expression.decompose_o3().as_dict(),
            "parameter_count": self.parameter_count,
            "domain": self.domain,
            "assembly_ir": self.assembly.as_dict(),
            "relation_to_full": self.relation_to_full.value,
            "rank": self.rank,
            "graph": self.graph.as_dict() if self.graph is not None else None,
            "restriction": self.restriction,
            "computational_oracles": list(self.computational_oracles),
            "certificates": {
                "equivariance": "compositional",
                "positivity": self.assembly.positivity,
            },
        }
