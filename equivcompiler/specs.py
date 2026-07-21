"""Semantic output and feature contracts for the public compiler API."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from compatibility.e3nn import o3

from representations import O3IrrepsSpec, irrep_multiplicities
from representations.representation_ir import (
    CoordinateSpec,
    DirectSumExpr,
    InnerProductRep,
    InvariantMetricSpec,
    IrrepsExpr,
    RepExpr,
    SymmetricSquareExpr,
)


FeatureScope = Literal["global", "node", "edge"]
FeatureLayout = Literal["e3nn", "compiler_native"]
GroupName = Literal["O3", "SO3"]


def _components(irreps: o3.Irreps) -> tuple[dict[str, Any], ...]:
    counts = irrep_multiplicities(irreps)
    return tuple(
        {
            "irrep": str(irrep),
            "angular_momentum": irrep.l,
            "parity": "even" if irrep.p == 1 else "odd",
            "multiplicity": multiplicity,
        }
        for irrep, multiplicity in sorted(
            counts.items(), key=lambda item: (item[0].l, -item[0].p)
        )
    )


@dataclass(frozen=True)
class FeatureSpec:
    """Complete contract for the equivariant features consumed by a readout."""

    irreps: o3.Irreps
    group: GroupName = "O3"
    scope: FeatureScope = "global"
    layout: FeatureLayout = "e3nn"
    basis_convention: str = "e3nn_real_v1"
    parity_convention: str = "e3nn_o3_v1"
    allow_pooling: bool = True
    metric_kind: str = "orthonormal_identity"
    gram_matrix_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "irreps", o3.Irreps(self.irreps))
        if self.group not in {"O3", "SO3"}:
            raise ValueError(f"unsupported feature group declaration: {self.group}")
        if self.scope not in {"global", "node", "edge"}:
            raise ValueError(f"unsupported feature scope: {self.scope}")
        if self.layout not in {"e3nn", "compiler_native"}:
            raise ValueError(f"unsupported feature layout: {self.layout}")
        if not self.basis_convention:
            raise ValueError("basis_convention must not be empty")
        if self.metric_kind != "orthonormal_identity" and not self.gram_matrix_id:
            raise ValueError(
                "a non-orthonormal coordinate contract requires gram_matrix_id"
            )

    @classmethod
    def from_irreps(
        cls,
        irreps: o3.Irreps | str,
        *,
        group: GroupName = "O3",
        scope: FeatureScope = "global",
        layout: FeatureLayout = "e3nn",
        basis_convention: str = "e3nn_real_v1",
        parity_convention: str = "e3nn_o3_v1",
        allow_pooling: bool = True,
        metric_kind: str = "orthonormal_identity",
        gram_matrix_id: str | None = None,
    ) -> "FeatureSpec":
        return cls(
            o3.Irreps(irreps),
            group=group,
            scope=scope,
            layout=layout,
            basis_convention=basis_convention,
            parity_convention=parity_convention,
            allow_pooling=allow_pooling,
            metric_kind=metric_kind,
            gram_matrix_id=gram_matrix_id,
        )

    @classmethod
    def from_backbone(
        cls,
        backbone,
        *,
        scope: FeatureScope | None = None,
    ) -> "FeatureSpec":
        declared = getattr(backbone, "feature_spec", None)
        if isinstance(declared, cls):
            if scope is not None and scope != declared.scope:
                raise ValueError("scope override conflicts with backbone.feature_spec")
            return declared
        if not hasattr(backbone, "irreps_out"):
            raise ValueError("backbone must expose irreps_out or feature_spec")
        return cls.from_irreps(
            backbone.irreps_out,
            group=getattr(backbone, "feature_group", "O3"),
            scope=scope or getattr(backbone, "feature_scope", "node"),
            layout=getattr(backbone, "feature_layout", "e3nn"),
            basis_convention=getattr(
                backbone, "feature_basis_convention", "e3nn_real_v1"
            ),
            parity_convention=getattr(
                backbone, "feature_parity_convention", "e3nn_o3_v1"
            ),
            allow_pooling=getattr(backbone, "allow_output_pooling", True),
            metric_kind=getattr(
                backbone, "feature_metric_kind", "orthonormal_identity"
            ),
            gram_matrix_id=getattr(backbone, "feature_gram_matrix_id", None),
        )

    @property
    def metric(self) -> InvariantMetricSpec:
        return InvariantMetricSpec(self.metric_kind, self.gram_matrix_id)

    @property
    def fiber(self) -> InnerProductRep:
        return InnerProductRep(
            self.group,
            IrrepsExpr(self.irreps, "feature_fiber"),
            self.metric,
        )

    @property
    def coordinates(self) -> CoordinateSpec:
        return CoordinateSpec(self.basis_convention, self.layout, self.metric)

    def as_dict(self) -> dict[str, Any]:
        return {
            "irreps": str(self.irreps),
            "dimension": self.irreps.dim,
            "components": list(_components(self.irreps)),
            "group": self.group,
            "scope": self.scope,
            "layout": self.layout,
            "basis_convention": self.basis_convention,
            "parity_convention": self.parity_convention,
            "allow_pooling": self.allow_pooling,
            "last_dimension_layout": "contiguous_irrep_terms_in_declared_order",
            "inner_product_rep": self.fiber.as_dict(),
            "coordinates": self.coordinates.as_dict(),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OutputSemantics:
    """Canonical probabilistic semantics derivable from ``V`` alone."""

    output_spec: O3IrrepsSpec
    output_expression: RepExpr
    full_reference_expression: RepExpr
    output_representation: str
    covariance_representation: str
    canonical_target: str
    output_dimension: int
    full_covariance_parameters: int
    highest_output_angular_momentum: int
    highest_covariance_angular_momentum: int
    components: tuple[dict[str, Any], ...]
    covariance_components: tuple[dict[str, Any], ...]
    executable: bool = False
    reachability: str = "unknown_without_seed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_representation": self.output_representation,
            "covariance_representation": self.covariance_representation,
            "canonical_target": self.canonical_target,
            "cartesian_formula": self.output_spec.cartesian_formula,
            "output_dimension": self.output_dimension,
            "full_covariance_parameters": self.full_covariance_parameters,
            "highest_output_angular_momentum": self.highest_output_angular_momentum,
            "highest_covariance_angular_momentum": self.highest_covariance_angular_momentum,
            "components": list(self.components),
            "covariance_components": list(self.covariance_components),
            "executable": self.executable,
            "reachability": self.reachability,
            "representation_ir": {
                "output_expression": self.output_expression.as_dict(),
                "full_reference_expression": self.full_reference_expression.as_dict(),
                "full_reference_decomposition": self.full_reference_expression.decompose_o3().as_dict(),
            },
        }


def _output_spec(output: O3IrrepsSpec | o3.Irreps | str) -> O3IrrepsSpec:
    if isinstance(output, O3IrrepsSpec):
        return output
    if isinstance(output, str) and "=" in output:
        return O3IrrepsSpec.from_cartesian(output)
    return O3IrrepsSpec(o3.Irreps(output))


def describe_output(
    output: O3IrrepsSpec | o3.Irreps | str,
) -> OutputSemantics:
    """Analyze ``V``, ``Sym^2(V)``, and ``T(V)`` without planning execution."""
    spec = _output_spec(output)
    output_expression = IrrepsExpr(spec.irreps, "output")
    full_reference_expression = DirectSumExpr(
        (output_expression, SymmetricSquareExpr(output_expression))
    )
    covariance = spec.symmetric_square_irreps
    canonical = full_reference_expression.decompose_o3().irreps
    return OutputSemantics(
        output_spec=spec,
        output_expression=output_expression,
        full_reference_expression=full_reference_expression,
        output_representation=str(spec.irreps),
        covariance_representation=str(covariance),
        canonical_target=str(canonical),
        output_dimension=spec.dim,
        full_covariance_parameters=spec.dim * (spec.dim + 1) // 2,
        highest_output_angular_momentum=max(irrep.l for _, irrep in spec.irreps),
        highest_covariance_angular_momentum=max(irrep.l for _, irrep in covariance),
        components=_components(spec.irreps),
        covariance_components=_components(covariance),
    )
