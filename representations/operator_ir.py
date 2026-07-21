"""Verified semantic IR for structured positive operators.

The IR is deliberately semantic: released operator families still use optimized
runtime plugins.  Positivity and equivariance are nevertheless derived here
from a closed rule set; callers cannot attach their own certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Literal

from compatibility.e3nn import o3

from representations.graph_structure import EquivariantOutputGraph
from representations.representation_ir import DirectSumExpr, RepExpr


class Positivity(str, Enum):
    """Verified order property of a symmetric operator program."""

    UNKNOWN = "unknown"
    PSD = "psd"
    SPD = "spd"


class Equivariance(str, Enum):
    """Whether equivariance follows from registered typed primitives."""

    UNKNOWN = "unknown"
    VERIFIED = "verified"


class FamilyRelation(str, Enum):
    """Set-theoretic relation between statistical operator families."""

    EQUAL_TO_FULL = "equal_to_full"
    STRICT_SUBSET = "strict_subset"
    STRICT_SUPERSET = "strict_superset"
    EQUAL = "equal"
    INCOMPARABLE = "incomparable"
    UNKNOWN = "unknown"


_POSITIVE_SPECTRAL_MAPS = {"matrix_exponential", "multiplicity_cholesky"}
_VERIFIED_INTERTWINERS = {"homogeneous_graph_coboundary"}


@dataclass(frozen=True)
class OperatorVerification:
    """Certificate derived by the verifier, never supplied by an IR node."""

    positivity: Positivity
    equivariance: Equivariance
    instructions: tuple[str, ...]
    unknown_instructions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.unknown_instructions and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "positivity": self.positivity.value,
            "equivariance": self.equivariance.value,
            "instructions": list(self.instructions),
            "unknown_instructions": list(self.unknown_instructions),
            "errors": list(self.errors),
            "derivation": "closed_primitive_rule_set",
        }


@dataclass(frozen=True)
class OperatorIR:
    """Composable semantic operator program with verifier-derived properties."""

    kind: str
    inputs: tuple["OperatorIR", ...] = ()
    attributes: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def node(cls, kind: str, *inputs: "OperatorIR", **attributes: Any) -> "OperatorIR":
        """Build a node without accepting claimed proof conclusions."""
        forbidden = {"positivity", "equivariance"}.intersection(attributes)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"operator proof fields are verifier-derived: {names}")
        return cls(
            kind=kind,
            inputs=tuple(inputs),
            attributes=tuple(sorted(attributes.items())),
        )

    @classmethod
    def symmetric_operator(cls, **attributes: Any) -> "OperatorIR":
        parameter = attributes.pop("parameter", None)
        inputs = () if parameter is None else (parameter,)
        return cls.node("symmetric_operator", *inputs, **attributes)

    @classmethod
    def parameter(
        cls,
        binding: str,
        *,
        start: int = 0,
        stop: int | None = None,
        coordinate_layout: str = "native",
        unit_irreps: str | None = None,
        copies: int | None = None,
    ) -> "OperatorIR":
        attributes: dict[str, Any] = {"binding": binding, "start": start}
        if stop is not None:
            attributes["stop"] = stop
        if coordinate_layout != "native":
            attributes["coordinate_layout"] = coordinate_layout
            attributes["unit_irreps"] = unit_irreps
            attributes["copies"] = copies
        return cls.node("parameter", **attributes)

    @classmethod
    def spectral_positive(
        cls, operator: "OperatorIR", *, map: str
    ) -> "OperatorIR":
        return cls.node("spectral_positive", operator, map=map)

    @classmethod
    def equivariant_factor(
        cls, parameter: "OperatorIR", *, rank: int, output_irreps: str
    ) -> "OperatorIR":
        return cls.node(
            "equivariant_factor",
            parameter,
            rank=rank,
            output_irreps=output_irreps,
        )

    @classmethod
    def gram(cls, factor: "OperatorIR") -> "OperatorIR":
        return cls.node("gram", factor)

    @classmethod
    def positive_scalar_identity(
        cls, parameter: "OperatorIR", *, dimension: int, minimum: float = 1e-4
    ) -> "OperatorIR":
        return cls.node(
            "positive_scalar_identity",
            parameter,
            dimension=dimension,
            minimum=minimum,
        )

    @classmethod
    def cholesky_positive(
        cls, parameter: "OperatorIR", *, dimension: int, minimum: float = 1e-4
    ) -> "OperatorIR":
        return cls.node(
            "cholesky_positive",
            parameter,
            dimension=dimension,
            minimum=minimum,
        )

    @classmethod
    def add(cls, *operators: "OperatorIR") -> "OperatorIR":
        return cls.node("add", *operators)

    @classmethod
    def direct_sum(cls, *operators: "OperatorIR", **attributes: Any) -> "OperatorIR":
        return cls.node("direct_sum", *operators, **attributes)

    @classmethod
    def kronecker_identity(
        cls, operator: "OperatorIR", **attributes: Any
    ) -> "OperatorIR":
        return cls.node("kronecker_identity", operator, **attributes)

    @classmethod
    def pullback(
        cls, operator: "OperatorIR", *, intertwiner: str
    ) -> "OperatorIR":
        return cls.node("pullback", operator, intertwiner=intertwiner)

    def attribute_dict(self) -> dict[str, Any]:
        return dict(self.attributes)

    def verify(self) -> OperatorVerification:
        return verify_operator(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.semantic_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "inputs": [item.semantic_dict() for item in self.inputs],
            "attributes": dict(self.attributes),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "verification": self.verify().as_dict()}


def _merge_children(children: tuple[OperatorVerification, ...]) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    instructions = tuple(item for child in children for item in child.instructions)
    unknown = tuple(item for child in children for item in child.unknown_instructions)
    errors = tuple(item for child in children for item in child.errors)
    return instructions, unknown, errors


def _all_equivariant(children: tuple[OperatorVerification, ...]) -> Equivariance:
    return (
        Equivariance.VERIFIED
        if all(child.equivariance is Equivariance.VERIFIED for child in children)
        else Equivariance.UNKNOWN
    )


def verify_operator(node: OperatorIR) -> OperatorVerification:
    """Derive operator certificates from typed primitive rules."""
    children = tuple(verify_operator(child) for child in node.inputs)
    nested_instructions, nested_unknown, nested_errors = _merge_children(children)
    instructions = nested_instructions + (node.kind,)
    attributes = node.attribute_dict()
    errors = list(nested_errors)
    unknown = list(nested_unknown)

    if node.kind == "parameter":
        if node.inputs or not attributes.get("binding"):
            errors.append("parameter must be a named leaf")
        coordinate_layout = attributes.get("coordinate_layout", "native")
        if coordinate_layout not in {"native", "repeated_irrep"}:
            errors.append(f"unregistered parameter coordinate layout: {coordinate_layout!r}")
        if coordinate_layout == "repeated_irrep" and (
            not attributes.get("unit_irreps")
            or int(attributes.get("copies") or 0) < 1
        ):
            errors.append(
                "repeated_irrep parameter layout requires unit_irreps and positive copies"
            )
        positivity = Positivity.UNKNOWN
        equivariance = Equivariance.VERIFIED
    elif node.kind == "symmetric_operator":
        if len(children) != 1 or node.inputs[0].kind != "parameter":
            errors.append("symmetric_operator requires one parameter input")
        positivity = Positivity.UNKNOWN
        equivariance = _all_equivariant(children)
    elif node.kind == "equivariant_factor":
        if (
            len(children) != 1
            or node.inputs[0].kind != "parameter"
            or int(attributes.get("rank", 0)) < 1
        ):
            errors.append("equivariant_factor requires a parameter and positive rank")
        positivity = Positivity.UNKNOWN
        equivariance = _all_equivariant(children)
    elif node.kind == "positive_scalar_identity":
        if len(children) != 1 or node.inputs[0].kind != "parameter":
            errors.append("positive_scalar_identity requires one parameter input")
        positivity = Positivity.SPD
        equivariance = _all_equivariant(children)
    elif node.kind == "cholesky_positive":
        if len(children) != 1 or node.inputs[0].kind != "parameter":
            errors.append("cholesky_positive requires one parameter input")
        if int(attributes.get("dimension", 0)) < 1:
            errors.append("cholesky_positive requires a positive dimension")
        positivity = Positivity.SPD if not errors else Positivity.UNKNOWN
        equivariance = _all_equivariant(children)
    elif node.kind == "spectral_positive":
        spectral_map = attributes.get("map")
        if len(children) != 1 or node.inputs[0].kind != "symmetric_operator":
            errors.append("spectral_positive requires one symmetric_operator input")
        if spectral_map not in _POSITIVE_SPECTRAL_MAPS:
            errors.append(f"unregistered positive spectral map: {spectral_map!r}")
        positivity = Positivity.SPD if not errors else Positivity.UNKNOWN
        equivariance = _all_equivariant(children)
    elif node.kind == "gram":
        if len(children) != 1 or node.inputs[0].kind != "equivariant_factor":
            errors.append("gram requires one equivariant_factor input")
        positivity = Positivity.PSD if not errors else Positivity.UNKNOWN
        equivariance = _all_equivariant(children)
    elif node.kind in {"add", "direct_sum"}:
        if not children:
            errors.append(f"{node.kind} requires at least one input")
        if children and all(
            child.positivity in {Positivity.PSD, Positivity.SPD}
            for child in children
        ):
            positivity = (
                Positivity.SPD
                if any(child.positivity is Positivity.SPD for child in children)
                else Positivity.PSD
            )
        else:
            positivity = Positivity.UNKNOWN
        equivariance = _all_equivariant(children)
    elif node.kind == "kronecker_identity":
        if len(children) != 1:
            errors.append("kronecker_identity requires one operator input")
        positivity = children[0].positivity if len(children) == 1 else Positivity.UNKNOWN
        equivariance = _all_equivariant(children)
    elif node.kind == "pullback":
        intertwiner = attributes.get("intertwiner")
        if len(children) != 1:
            errors.append("pullback requires one operator input")
        if intertwiner not in _VERIFIED_INTERTWINERS:
            errors.append(f"unregistered intertwiner: {intertwiner!r}")
        positivity = (
            Positivity.PSD
            if len(children) == 1
            and children[0].positivity in {Positivity.PSD, Positivity.SPD}
            and not errors
            else Positivity.UNKNOWN
        )
        equivariance = (
            _all_equivariant(children)
            if intertwiner in _VERIFIED_INTERTWINERS
            else Equivariance.UNKNOWN
        )
    else:
        positivity = Positivity.UNKNOWN
        equivariance = Equivariance.UNKNOWN
        unknown.append(node.kind)

    return OperatorVerification(
        positivity=positivity,
        equivariance=equivariance,
        instructions=instructions,
        unknown_instructions=tuple(dict.fromkeys(unknown)),
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class FamilyRelationCertificate:
    """Proven or deliberately unknown relation between two family plans."""

    relation: FamilyRelation
    evidence: str

    def as_dict(self) -> dict[str, str]:
        return {"relation": self.relation.value, "evidence": self.evidence}


@dataclass(frozen=True)
class ParameterBinding:
    """One independently projected representation consumed by an IR program."""

    name: str
    expression: RepExpr
    projection_name: str

    def __post_init__(self) -> None:
        if not self.name or not self.projection_name:
            raise ValueError("parameter binding names must not be empty")

    @property
    def irreps(self) -> o3.Irreps:
        return self.expression.decompose_o3().irreps

    @property
    def dimension(self) -> int:
        return self.irreps.dim

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression.as_dict(),
            "representation": self.expression.decompose_o3().as_dict(),
            "projection_name": self.projection_name,
        }


@dataclass(frozen=True)
class OperatorFamilyPlan:
    """Semantic operator family before reachability and executable lowering."""

    kind: str
    parameter_bindings: tuple[ParameterBinding, ...]
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

    def __post_init__(self) -> None:
        if not self.parameter_bindings:
            raise ValueError("operator family requires at least one parameter binding")
        names = [binding.name for binding in self.parameter_bindings]
        if len(set(names)) != len(names):
            raise ValueError("operator parameter binding names must be unique")
        references = [
            dict(node.attributes)["binding"]
            for node in _walk_operator(self.assembly)
            if node.kind == "parameter"
        ]
        if set(references) != set(names):
            raise ValueError(
                "operator parameter references do not match bindings: "
                f"references={sorted(set(references))}, bindings={sorted(names)}"
            )
        if self.parameter_count != sum(binding.dimension for binding in self.parameter_bindings):
            raise ValueError("parameter_count does not match bound representation dimensions")
        verification = self.assembly.verify()
        if not verification.valid:
            raise ValueError(
                "operator program failed verification: "
                f"unknown={verification.unknown_instructions}, errors={verification.errors}"
            )
        if verification.positivity is not Positivity.SPD:
            raise ValueError("operator family root must be verifier-proven SPD")
        if verification.equivariance is not Equivariance.VERIFIED:
            raise ValueError("operator family root must be verifier-proven equivariant")

    @property
    def parameter_expression(self) -> RepExpr:
        return DirectSumExpr(tuple(binding.expression for binding in self.parameter_bindings))

    @property
    def parameter_irreps(self) -> o3.Irreps:
        return self.parameter_expression.decompose_o3().irreps

    def active_expression(self, location: RepExpr) -> RepExpr:
        return DirectSumExpr((location, self.parameter_expression))

    def relation_to(self, other: "OperatorFamilyPlan") -> FamilyRelationCertificate:
        if self.assembly.fingerprint == other.assembly.fingerprint and (
            self.parameter_expression.as_dict() == other.parameter_expression.as_dict()
        ):
            return FamilyRelationCertificate(FamilyRelation.EQUAL, "identical_semantic_ir")
        if self.kind == "full":
            relation = (
                FamilyRelation.EQUAL
                if other.relation_to_full is FamilyRelation.EQUAL_TO_FULL
                else FamilyRelation.STRICT_SUPERSET
            )
            return FamilyRelationCertificate(relation, "full_family_reference")
        if other.kind == "full":
            relation = (
                FamilyRelation.EQUAL
                if self.relation_to_full is FamilyRelation.EQUAL_TO_FULL
                else FamilyRelation.STRICT_SUBSET
            )
            return FamilyRelationCertificate(relation, "full_family_reference")
        if self.kind == other.kind == "low_rank" and self.rank and other.rank:
            if self.rank == other.rank:
                return FamilyRelationCertificate(FamilyRelation.EQUAL, "equal_factor_rank")
            relation = (
                FamilyRelation.STRICT_SUBSET
                if self.rank < other.rank
                else FamilyRelation.STRICT_SUPERSET
            )
            return FamilyRelationCertificate(relation, "nested_factor_column_spaces")
        if self.kind == other.kind == "graph":
            return FamilyRelationCertificate(
                FamilyRelation.UNKNOWN,
                "different_graph_precision_cones_have_no_registered_relation_proof",
            )
        return FamilyRelationCertificate(
            FamilyRelation.UNKNOWN,
            "no_registered_family_relation_proof",
        )

    def as_dict(self) -> dict[str, Any]:
        verification = self.assembly.verify()
        return {
            "kind": self.kind,
            "parameter_bindings": [binding.as_dict() for binding in self.parameter_bindings],
            "parameter_expression": self.parameter_expression.as_dict(),
            "parameter_representation": self.parameter_expression.decompose_o3().as_dict(),
            "parameter_count": self.parameter_count,
            "domain": self.domain,
            "assembly_ir": self.assembly.as_dict(),
            "operator_program_hash": self.assembly.fingerprint,
            "relation_to_full": self.relation_to_full.value,
            "rank": self.rank,
            "graph": self.graph.as_dict() if self.graph is not None else None,
            "restriction": self.restriction,
            "computational_oracles": list(self.computational_oracles),
            "certificates": verification.as_dict(),
        }


def _walk_operator(node: OperatorIR) -> tuple[OperatorIR, ...]:
    return (node,) + tuple(
        descendant
        for child in node.inputs
        for descendant in _walk_operator(child)
    )
