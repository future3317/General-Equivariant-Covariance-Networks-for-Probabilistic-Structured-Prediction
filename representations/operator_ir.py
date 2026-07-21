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
from typing import Any, Literal, Mapping

from compatibility.e3nn import o3

from representations.graph_structure import EquivariantOutputGraph
from representations.irrep_layout import RepeatedIrrepLayout
from representations.representation_ir import (
    DirectSumExpr,
    IrrepsExpr,
    RepExpr,
    RepeatedExpr,
    SymmetricSquareExpr,
)


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


_POSITIVE_SPECTRAL_MAPS = {"matrix_exponential"}
_VERIFIED_INTERTWINERS = {"homogeneous_graph_coboundary"}


@dataclass(frozen=True)
class OperatorVerificationContext:
    """Typed environment required to verify an operator program."""

    bindings: Mapping[str, RepExpr]
    output_irreps: o3.Irreps
    graph: EquivariantOutputGraph | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", dict(self.bindings))
        object.__setattr__(self, "output_irreps", o3.Irreps(self.output_irreps))

    def as_dict(self) -> dict[str, Any]:
        return {
            "bindings": {
                name: expression.as_dict()
                for name, expression in sorted(self.bindings.items())
            },
            "output_irreps": str(self.output_irreps),
            "graph": self.graph.as_dict() if self.graph is not None else None,
        }


@dataclass(frozen=True)
class OperatorVerification:
    """Certificate derived by the verifier, never supplied by an IR node."""

    positivity: Positivity
    equivariance: Equivariance
    instructions: tuple[str, ...]
    unknown_instructions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    result_dimension: int | None = None
    result_irreps: str | None = None
    environment_typed: bool = False

    @property
    def valid(self) -> bool:
        # A certificate is valid only when the closed rule set established both
        # semantic properties needed by an operator lowering.  A well-typed
        # parameter leaf intentionally has unknown positivity, so it must not
        # be mistaken for a proof merely because no syntax error was found.
        return (
            not self.unknown_instructions
            and not self.errors
            and self.positivity is not Positivity.UNKNOWN
            and self.equivariance is Equivariance.VERIFIED
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "positivity": self.positivity.value,
            "equivariance": self.equivariance.value,
            "instructions": list(self.instructions),
            "unknown_instructions": list(self.unknown_instructions),
            "errors": list(self.errors),
            "result_dimension": self.result_dimension,
            "result_irreps": self.result_irreps,
            "environment_typed": self.environment_typed,
            "derivation": "closed_typed_primitive_rule_set",
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
    def spectral_positive(cls, operator: "OperatorIR", *, map: str) -> "OperatorIR":
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
    def pullback(cls, operator: "OperatorIR", *, intertwiner: str) -> "OperatorIR":
        return cls.node("pullback", operator, intertwiner=intertwiner)

    def attribute_dict(self) -> dict[str, Any]:
        return dict(self.attributes)

    def verify(
        self, context: OperatorVerificationContext | None = None
    ) -> OperatorVerification:
        return verify_operator(self, context)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.semantic_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "inputs": [item.semantic_dict() for item in self.inputs],
            "attributes": dict(self.attributes),
        }

    def as_dict(
        self, context: OperatorVerificationContext | None = None
    ) -> dict[str, Any]:
        return {**self.semantic_dict(), "verification": self.verify(context).as_dict()}


@dataclass(frozen=True)
class _TypedNode:
    certificate: OperatorVerification
    irreps: o3.Irreps | None = None
    binding_expression: RepExpr | None = None
    trivial_coordinates: bool = False
    coordinate_layout: str | None = None


def _same_irreps(left: o3.Irreps, right: o3.Irreps) -> bool:
    return o3.Irreps(left) == o3.Irreps(right)


def _all_trivial(irreps: o3.Irreps) -> bool:
    return bool(irreps) and all(irrep.l == 0 and irrep.p == 1 for _, irrep in irreps)


def _verification(
    positivity: Positivity,
    equivariance: Equivariance,
    instructions: tuple[str, ...],
    unknown: list[str],
    errors: list[str],
    dimension: int | None,
    irreps: o3.Irreps | None,
    context: OperatorVerificationContext | None,
) -> OperatorVerification:
    return OperatorVerification(
        positivity=positivity,
        equivariance=equivariance,
        instructions=instructions,
        unknown_instructions=tuple(dict.fromkeys(unknown)),
        errors=tuple(dict.fromkeys(errors)),
        result_dimension=dimension,
        result_irreps=str(irreps) if irreps is not None else None,
        environment_typed=context is not None,
    )


def _verify_node(
    node: OperatorIR, context: OperatorVerificationContext | None
) -> _TypedNode:
    children = tuple(_verify_node(child, context) for child in node.inputs)
    certificates = tuple(child.certificate for child in children)
    instructions = tuple(
        instruction for child in certificates for instruction in child.instructions
    ) + (node.kind,)
    unknown = [
        instruction
        for child in certificates
        for instruction in child.unknown_instructions
    ]
    errors = [error for child in certificates for error in child.errors]
    attributes = node.attribute_dict()
    dimension: int | None = None
    result_irreps: o3.Irreps | None = None
    expression: RepExpr | None = None
    trivial = False
    coordinate_layout: str | None = None

    def all_equivariant() -> Equivariance:
        return (
            Equivariance.VERIFIED
            if all(
                child.equivariance is Equivariance.VERIFIED for child in certificates
            )
            else Equivariance.UNKNOWN
        )

    if node.kind == "parameter":
        if node.inputs or not attributes.get("binding"):
            errors.append("parameter must be a named leaf")
        binding_name = str(attributes.get("binding", ""))
        coordinate_layout = str(attributes.get("coordinate_layout", "native"))
        binding_irreps: o3.Irreps | None = None
        if context is None:
            errors.append("parameter verification requires a typed binding environment")
        elif binding_name not in context.bindings:
            errors.append(f"unknown parameter binding: {binding_name!r}")
        else:
            expression = context.bindings[binding_name]
            binding_irreps = expression.decompose_o3().irreps

        if coordinate_layout not in {"native", "repeated_irrep"}:
            errors.append(
                f"unregistered parameter coordinate layout: {coordinate_layout!r}"
            )
        if coordinate_layout == "repeated_irrep" and binding_irreps is not None:
            unit_irreps = attributes.get("unit_irreps")
            copies = int(attributes.get("copies") or 0)
            if not unit_irreps or copies < 1:
                errors.append(
                    "repeated_irrep parameter layout requires unit_irreps and positive copies"
                )
            else:
                layout = RepeatedIrrepLayout(str(unit_irreps), copies)
                if not _same_irreps(layout.expanded_irreps, binding_irreps):
                    errors.append(
                        "repeated_irrep layout does not match the binding decomposition"
                    )
        total = binding_irreps.dim if binding_irreps is not None else 0
        try:
            start = int(attributes.get("start", 0))
            stop = int(attributes.get("stop", total))
        except (TypeError, ValueError):
            start, stop = 0, -1
            errors.append("parameter start/stop must be integers")
        if start < 0 or stop < start or stop > total:
            errors.append(
                f"parameter slice [{start}:{stop}] is outside transformed dimension {total}"
            )
        dimension = max(stop - start, 0)
        if binding_irreps is not None:
            trivial = _all_trivial(binding_irreps)
            if (
                coordinate_layout == "native"
                and not trivial
                and (start != 0 or stop != total)
            ):
                errors.append(
                    "native nontrivial bindings may only be consumed as a whole"
                )
            if coordinate_layout == "repeated_irrep" and attributes.get("unit_irreps"):
                unit_dim = o3.Irreps(str(attributes["unit_irreps"])).dim
                if start % unit_dim or stop % unit_dim:
                    errors.append(
                        "repeated_irrep slices must align to complete representation copies"
                    )
        positivity = Positivity.UNKNOWN
        equivariance = (
            Equivariance.VERIFIED
            if context is not None and not errors
            else Equivariance.UNKNOWN
        )

    elif node.kind == "symmetric_operator":
        if len(children) != 1 or node.inputs[0].kind != "parameter":
            errors.append("symmetric_operator requires one parameter input")
        space = attributes.get("coordinate_space")
        if context is None:
            errors.append(
                "symmetric_operator verification requires an output environment"
            )
        elif len(children) == 1:
            child = children[0]
            if space == "output_representation":
                expected = SymmetricSquareExpr(
                    IrrepsExpr(context.output_irreps, "output")
                )
                expected_dim = expected.decompose_o3().irreps.dim
                if not isinstance(
                    child.binding_expression, SymmetricSquareExpr
                ) or not _same_irreps(
                    child.binding_expression.operand.decompose_o3().irreps,
                    context.output_irreps,
                ):
                    errors.append("output symmetric_operator binding must be Sym^2(V)")
                if child.certificate.result_dimension != expected_dim:
                    errors.append(
                        "output symmetric_operator consumes the wrong coordinate count"
                    )
                if attributes.get("output_irreps") != str(context.output_irreps):
                    errors.append("symmetric_operator output_irreps does not match V")
                dimension = context.output_irreps.dim
                result_irreps = context.output_irreps
            elif space == "graph_local":
                graph = context.graph
                if graph is None:
                    errors.append("graph_local symmetric_operator requires a graph")
                else:
                    local = IrrepsExpr(f"1x{graph.node_irrep}", "graph_residual")
                    local_symmetric = SymmetricSquareExpr(local)
                    local_count = local_symmetric.decompose_o3().irreps.dim
                    role = attributes.get("role")
                    expected_copies = {
                        "unary": graph.num_nodes,
                        "factor": graph.num_edges,
                    }.get(role)
                    copies = int(attributes.get("copies", -1))
                    if expected_copies is None or copies != expected_copies:
                        errors.append("graph_local role/copies do not match the graph")
                    binding = child.binding_expression
                    if not (
                        isinstance(binding, RepeatedExpr)
                        and binding.copies == graph.num_potentials
                        and isinstance(binding.operand, SymmetricSquareExpr)
                        and _same_irreps(
                            binding.operand.operand.decompose_o3().irreps,
                            local.decompose_o3().irreps,
                        )
                    ):
                        errors.append(
                            "graph-local binding must be repeated local Sym^2(V0) potentials"
                        )
                    if child.coordinate_layout != "repeated_irrep":
                        errors.append(
                            "graph-local potentials require repeated_irrep layout"
                        )
                    if (
                        child.certificate.result_dimension
                        != max(copies, 0) * local_count
                    ):
                        errors.append(
                            "graph-local parameter slice has the wrong coordinate count"
                        )
                    if attributes.get("irrep") != str(graph.node_irrep):
                        errors.append(
                            "graph-local irrep does not match the graph node irrep"
                        )
                    dimension = max(copies, 0) * graph.block_dim
                    result_irreps = o3.Irreps([(max(copies, 0), graph.node_irrep)])
            else:
                errors.append(f"unknown symmetric coordinate space: {space!r}")
        positivity = Positivity.UNKNOWN
        equivariance = all_equivariant()

    elif node.kind == "equivariant_factor":
        rank = int(attributes.get("rank", 0))
        if len(children) != 1 or node.inputs[0].kind != "parameter" or rank < 1:
            errors.append("equivariant_factor requires a parameter and positive rank")
        if context is None:
            errors.append(
                "equivariant_factor verification requires an output environment"
            )
        elif len(children) == 1 and rank > 0:
            child = children[0]
            binding = child.binding_expression
            if not (
                isinstance(binding, RepeatedExpr)
                and binding.copies == rank
                and _same_irreps(
                    binding.operand.decompose_o3().irreps, context.output_irreps
                )
            ):
                errors.append("equivariant_factor binding must be r copies of V")
            if child.certificate.result_dimension != rank * context.output_irreps.dim:
                errors.append("equivariant_factor consumes the wrong coordinate count")
            if attributes.get("output_irreps") != str(context.output_irreps):
                errors.append("equivariant_factor output_irreps does not match V")
            dimension = context.output_irreps.dim
            result_irreps = context.output_irreps
        positivity = Positivity.UNKNOWN
        equivariance = all_equivariant()

    elif node.kind == "positive_scalar_identity":
        if len(children) != 1 or node.inputs[0].kind != "parameter":
            errors.append("positive_scalar_identity requires one parameter input")
        operator_dim = int(attributes.get("dimension", 0))
        if operator_dim < 1:
            errors.append("positive_scalar_identity requires a positive dimension")
        if float(attributes.get("minimum", 1e-4)) < 0:
            errors.append("positive_scalar_identity minimum must be nonnegative")
        if len(children) == 1 and (
            children[0].certificate.result_dimension != 1
            or not children[0].trivial_coordinates
        ):
            errors.append(
                "positive_scalar_identity requires exactly one trivial scalar"
            )
        dimension = operator_dim if operator_dim > 0 else None
        if context is not None and dimension == context.output_irreps.dim:
            result_irreps = context.output_irreps
        positivity = Positivity.SPD if not errors else Positivity.UNKNOWN
        equivariance = all_equivariant()

    elif node.kind == "cholesky_positive":
        if len(children) != 1 or node.inputs[0].kind != "parameter":
            errors.append("cholesky_positive requires one parameter input")
        operator_dim = int(attributes.get("dimension", 0))
        if operator_dim < 1:
            errors.append("cholesky_positive requires a positive dimension")
        if float(attributes.get("minimum", 1e-4)) < 0:
            errors.append("cholesky_positive minimum must be nonnegative")
        expected = operator_dim * (operator_dim + 1) // 2
        if len(children) == 1 and (
            children[0].certificate.result_dimension != expected
            or not children[0].trivial_coordinates
        ):
            errors.append(
                f"cholesky_positive({operator_dim}) requires {expected} trivial scalars"
            )
        dimension = operator_dim if operator_dim > 0 else None
        positivity = Positivity.SPD if not errors else Positivity.UNKNOWN
        equivariance = all_equivariant()

    elif node.kind == "spectral_positive":
        spectral_map = attributes.get("map")
        if len(children) != 1 or node.inputs[0].kind != "symmetric_operator":
            errors.append("spectral_positive requires one symmetric_operator input")
        if spectral_map not in _POSITIVE_SPECTRAL_MAPS:
            errors.append(f"unregistered positive spectral map: {spectral_map!r}")
        if children:
            dimension = children[0].certificate.result_dimension
            result_irreps = children[0].irreps
        positivity = Positivity.SPD if not errors else Positivity.UNKNOWN
        equivariance = all_equivariant()

    elif node.kind == "gram":
        if len(children) != 1 or node.inputs[0].kind != "equivariant_factor":
            errors.append("gram requires one equivariant_factor input")
        if children:
            dimension = children[0].certificate.result_dimension
            result_irreps = children[0].irreps
        positivity = Positivity.PSD if not errors else Positivity.UNKNOWN
        equivariance = all_equivariant()

    elif node.kind == "add":
        if not children:
            errors.append("add requires at least one input")
        dimensions = {child.certificate.result_dimension for child in children}
        if len(dimensions) > 1:
            errors.append("add inputs act on different operator dimensions")
        dimension = next(iter(dimensions)) if len(dimensions) == 1 else None
        known_irreps = [child.irreps for child in children if child.irreps is not None]
        if known_irreps and any(
            not _same_irreps(item, known_irreps[0]) for item in known_irreps[1:]
        ):
            errors.append("add inputs act on different representation spaces")
        result_irreps = known_irreps[0] if known_irreps else None
        positivity = (
            Positivity.SPD
            if children
            and all(
                child.certificate.positivity in {Positivity.PSD, Positivity.SPD}
                for child in children
            )
            and any(
                child.certificate.positivity is Positivity.SPD for child in children
            )
            and not errors
            else Positivity.UNKNOWN
        )
        equivariance = all_equivariant()

    elif node.kind == "direct_sum":
        if not children:
            errors.append("direct_sum requires at least one input")
        copies_attribute = attributes.get("copies")
        if copies_attribute is not None:
            copies = int(copies_attribute)
            if len(children) != 1 or copies < 0:
                errors.append(
                    "copied direct_sum requires one input and nonnegative copies"
                )
            if len(children) == 1:
                repeated = node.inputs[0]
                if repeated.kind == "spectral_positive" and repeated.inputs:
                    repeated = repeated.inputs[0]
                child_copies = int(repeated.attribute_dict().get("copies", -1))
                if child_copies != copies:
                    errors.append(
                        "direct_sum copies do not match its repeated operator"
                    )
                dimension = children[0].certificate.result_dimension
                result_irreps = children[0].irreps
        else:
            if children and all(
                child.certificate.result_dimension is not None for child in children
            ):
                dimension = sum(
                    int(child.certificate.result_dimension) for child in children
                )
            if children and all(child.irreps is not None for child in children):
                result_irreps = o3.Irreps(
                    [term for child in children for term in child.irreps]  # type: ignore[union-attr]
                )
        positivity = (
            Positivity.SPD
            if children
            and all(
                child.certificate.positivity is Positivity.SPD for child in children
            )
            and not errors
            else Positivity.UNKNOWN
        )
        equivariance = all_equivariant()

    elif node.kind == "kronecker_identity":
        if len(children) != 1:
            errors.append("kronecker_identity requires one operator input")
        try:
            irrep = o3.Irrep(str(attributes.get("irrep")))
        except ValueError:
            irrep = None
            errors.append("kronecker_identity requires a valid irrep")
        multiplicity = int(attributes.get("multiplicity", 0))
        if irrep is not None:
            if int(attributes.get("irrep_dimension", 0)) != irrep.dim:
                errors.append("kronecker_identity irrep dimension is inconsistent")
            if multiplicity < 1 or (
                children and children[0].certificate.result_dimension != multiplicity
            ):
                errors.append(
                    "kronecker_identity multiplicity does not match its block"
                )
            dimension = multiplicity * irrep.dim
            result_irreps = o3.Irreps([(multiplicity, irrep)])
        positivity = (
            children[0].certificate.positivity
            if len(children) == 1 and not errors
            else Positivity.UNKNOWN
        )
        equivariance = all_equivariant()

    elif node.kind == "pullback":
        intertwiner = attributes.get("intertwiner")
        if len(children) != 1:
            errors.append("pullback requires one operator input")
        if intertwiner not in _VERIFIED_INTERTWINERS:
            errors.append(f"unregistered intertwiner: {intertwiner!r}")
        graph = context.graph if context is not None else None
        if graph is None:
            errors.append("homogeneous_graph_coboundary requires a graph environment")
        elif len(children) == 1:
            expected_domain = graph.num_edges * graph.block_dim
            if children[0].certificate.result_dimension != expected_domain:
                errors.append("pullback child does not act on the graph edge domain")
            expected_irreps = o3.Irreps([(graph.num_edges, graph.node_irrep)])
            if children[0].irreps is None or not _same_irreps(
                children[0].irreps, expected_irreps
            ):
                errors.append(
                    "pullback child representation does not match graph edges"
                )
            dimension = graph.output_dim
            result_irreps = graph.output_irreps
        positivity = (
            Positivity.PSD
            if len(children) == 1
            and children[0].certificate.positivity in {Positivity.PSD, Positivity.SPD}
            and not errors
            else Positivity.UNKNOWN
        )
        equivariance = all_equivariant()

    else:
        positivity = Positivity.UNKNOWN
        equivariance = Equivariance.UNKNOWN
        unknown.append(node.kind)

    certificate = _verification(
        positivity,
        equivariance,
        instructions,
        unknown,
        errors,
        dimension,
        result_irreps,
        context,
    )
    return _TypedNode(
        certificate,
        result_irreps,
        expression,
        trivial,
        coordinate_layout,
    )


def verify_operator(
    node: OperatorIR, context: OperatorVerificationContext | None = None
) -> OperatorVerification:
    """Derive a certificate from registered primitives and their typed environment."""
    state = _verify_node(node, context)
    certificate = state.certificate
    if context is None:
        return certificate
    errors = list(certificate.errors)
    if certificate.result_dimension != context.output_irreps.dim:
        errors.append(
            "root operator dimension does not match the output representation"
        )
    if state.irreps is None or not _same_irreps(state.irreps, context.output_irreps):
        errors.append("root operator space is not the declared output representation")
    if errors == list(certificate.errors):
        return certificate
    return OperatorVerification(
        positivity=Positivity.UNKNOWN,
        equivariance=Equivariance.UNKNOWN,
        instructions=certificate.instructions,
        unknown_instructions=certificate.unknown_instructions,
        errors=tuple(dict.fromkeys(errors)),
        result_dimension=certificate.result_dimension,
        result_irreps=certificate.result_irreps,
        environment_typed=True,
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
    output_irreps: o3.Irreps
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
        object.__setattr__(self, "output_irreps", o3.Irreps(self.output_irreps))
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
        if self.parameter_count != sum(
            binding.dimension for binding in self.parameter_bindings
        ):
            raise ValueError(
                "parameter_count does not match bound representation dimensions"
            )
        verification = self.verification
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
    def verification_context(self) -> OperatorVerificationContext:
        return OperatorVerificationContext(
            bindings={
                binding.name: binding.expression for binding in self.parameter_bindings
            },
            output_irreps=self.output_irreps,
            graph=self.graph,
        )

    @property
    def verification(self) -> OperatorVerification:
        return self.assembly.verify(self.verification_context)

    @property
    def parameter_expression(self) -> RepExpr:
        return DirectSumExpr(
            tuple(binding.expression for binding in self.parameter_bindings)
        )

    @property
    def parameter_irreps(self) -> o3.Irreps:
        return self.parameter_expression.decompose_o3().irreps

    def active_expression(self, location: RepExpr) -> RepExpr:
        return DirectSumExpr((location, self.parameter_expression))

    def relation_to(self, other: "OperatorFamilyPlan") -> FamilyRelationCertificate:
        if self.assembly.fingerprint == other.assembly.fingerprint and (
            self.parameter_expression.as_dict() == other.parameter_expression.as_dict()
        ):
            return FamilyRelationCertificate(
                FamilyRelation.EQUAL, "identical_semantic_ir"
            )
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
                return FamilyRelationCertificate(
                    FamilyRelation.EQUAL, "equal_factor_rank"
                )
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
        verification = self.verification
        return {
            "kind": self.kind,
            "output_irreps": str(self.output_irreps),
            "parameter_bindings": [
                binding.as_dict() for binding in self.parameter_bindings
            ],
            "parameter_expression": self.parameter_expression.as_dict(),
            "parameter_representation": self.parameter_expression.decompose_o3().as_dict(),
            "parameter_count": self.parameter_count,
            "domain": self.domain,
            "assembly_ir": self.assembly.as_dict(self.verification_context),
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
        descendant for child in node.inputs for descendant in _walk_operator(child)
    )
