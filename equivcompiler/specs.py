"""Semantic output and feature contracts for the public compiler API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch

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


@dataclass(frozen=True)
class TargetTransform:
    """Typed affine map at the boundary of the compiled output contract.

    ``linear`` is represented in column-vector convention.  The compiler only
    accepts maps that commute with the declared output representation and whose
    bias is invariant.  Dataset preprocessing that does not satisfy this
    contract must be audited outside the exact compiler path.
    """

    output_irreps: o3.Irreps
    name: str = "identity"
    linear: tuple[tuple[float, ...], ...] | None = None
    bias: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        irreps = o3.Irreps(self.output_irreps)
        object.__setattr__(self, "output_irreps", irreps)
        dimension = irreps.dim
        if self.linear is not None:
            matrix = tuple(tuple(float(value) for value in row) for row in self.linear)
            if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
                raise ValueError("target-transform linear map has the wrong dimension")
            object.__setattr__(self, "linear", matrix)
        if self.bias is not None:
            vector = tuple(float(value) for value in self.bias)
            if len(vector) != dimension:
                raise ValueError("target-transform bias has the wrong dimension")
            object.__setattr__(self, "bias", vector)

    @classmethod
    def identity(cls, output_irreps: o3.Irreps | str, *, name: str = "identity"):
        return cls(o3.Irreps(output_irreps), name=name)

    @classmethod
    def affine(
        cls,
        output_irreps: o3.Irreps | str,
        linear: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        name: str = "affine",
    ):
        matrix = torch.as_tensor(linear, dtype=torch.float64)
        if matrix.ndim != 2:
            raise ValueError("target-transform linear map must be a matrix")
        vector = (
            torch.zeros(matrix.shape[0], dtype=torch.float64)
            if bias is None
            else torch.as_tensor(bias, dtype=torch.float64)
        )
        return cls(
            o3.Irreps(output_irreps),
            name=name,
            linear=tuple(tuple(float(value) for value in row) for row in matrix.tolist()),
            bias=tuple(float(value) for value in vector.tolist()),
        )

    @classmethod
    def from_multiplicity_blocks(
        cls,
        output_irreps: o3.Irreps | str,
        blocks: Mapping[str, torch.Tensor],
        bias: torch.Tensor | None = None,
        *,
        name: str = "multiplicity_block_affine",
    ) -> TargetTransform:
        """Construct an admissible linear map from isotypic blocks.

        A block for an irrep type acts on its multiplicity space and is
        repeated as the identity on the irrep coordinates.  Dense affine
        inputs remain supported through :meth:`affine`, but are checked
        against this same commutant structure by :meth:`verify`.
        """
        irreps = o3.Irreps(output_irreps)
        layout = _target_coordinate_layout(irreps)
        multiplicities = {
            str(irrep): max(slot for _, slot, _ in entries) + 1
            for irrep, entries in _layout_by_irrep(layout).items()
        }
        normalized: dict[str, torch.Tensor] = {}
        for irrep_name, multiplicity in multiplicities.items():
            block = torch.as_tensor(
                blocks.get(irrep_name, torch.eye(multiplicity, dtype=torch.float64)),
                dtype=torch.float64,
            )
            if block.shape != (multiplicity, multiplicity):
                raise ValueError(
                    f"multiplicity block for {irrep_name} must have shape "
                    f"({multiplicity}, {multiplicity}), got {tuple(block.shape)}"
                )
            normalized[irrep_name] = block
        linear = torch.zeros(irreps.dim, irreps.dim, dtype=torch.float64)
        slots = _layout_by_irrep(layout)
        for irrep_name, entries in slots.items():
            block = normalized[irrep_name]
            for row_index, row_slot, row_component in entries:
                for col_index, col_slot, col_component in entries:
                    if row_component == col_component:
                        linear[row_index, col_index] = block[row_slot, col_slot]
        return cls.affine(irreps, linear, bias, name=name)

    @property
    def dimension(self) -> int:
        return self.output_irreps.dim

    def linear_matrix(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        if self.linear is None:
            return torch.eye(self.dimension, dtype=dtype)
        return torch.tensor(self.linear, dtype=dtype)

    def bias_vector(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        if self.bias is None:
            return torch.zeros(self.dimension, dtype=dtype)
        return torch.tensor(self.bias, dtype=dtype)

    def verify(self, output_spec, *, tolerance: float = 1e-8) -> dict[str, Any]:
        """Verify the affine map against the exact irrep commutant structure.

        The structural check is the contract verifier.  The finite set of
        group elements is retained only as a numerical conformance audit and
        is never used as evidence for arbitrary O(3) equivariance.
        """
        if o3.Irreps(output_spec.irreps) != self.output_irreps:
            return {
                "status": "rejected",
                "reason": "output_irreps_mismatch",
                "expected": str(self.output_irreps),
                "actual": str(output_spec.irreps),
            }
        transformations = (
            torch.eye(3, dtype=torch.float64),
            torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
        )
        linear = self.linear_matrix()
        bias = self.bias_vector()
        projected_linear = _project_to_irrep_commutant(self.output_irreps, linear)
        linear_residual = float((linear - projected_linear).abs().max())
        invariant_coordinates = _invariant_coordinate_indices(self.output_irreps)
        projected_bias = torch.zeros_like(bias)
        if invariant_coordinates:
            projected_bias[list(invariant_coordinates)] = bias[list(invariant_coordinates)]
        bias_residual = float((bias - projected_bias).abs().max())
        numerical_linear_residual = 0.0
        numerical_bias_residual = 0.0
        for transformation in transformations:
            representation = output_spec.representation_matrix(transformation).double()
            numerical_linear_residual = max(
                numerical_linear_residual,
                float((linear @ representation - representation @ linear).abs().max()),
            )
            numerical_bias_residual = max(
                numerical_bias_residual,
                float((representation @ bias - bias).abs().max()),
            )
        verified = linear_residual <= tolerance and bias_residual <= tolerance
        return {
            "status": "verified" if verified else "rejected",
            "verification_kind": "irrep_commutant_structure",
            "tolerance": tolerance,
            "max_linear_intertwiner_residual": linear_residual,
            "max_bias_invariance_residual": bias_residual,
            "numerical_audit": {
                "kind": "finite_group_element_conformance",
                "elements": 3,
                "max_linear_intertwiner_residual": numerical_linear_residual,
                "max_bias_invariance_residual": numerical_bias_residual,
            },
            "admissible_form": "direct_sum(A_lambda tensor I_dim(lambda))",
            "invariant_bias_irrep": "0e",
            "conditions": ["A rho(g) = rho(g) A", "rho(g) b = b"],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_irreps": str(self.output_irreps),
            "dimension": self.dimension,
            "linear": self.linear_matrix().tolist(),
            "bias": self.bias_vector().tolist(),
            "coordinate_convention": "column_vector",
            "semantic_boundary": "target_transform_to_compiled_output",
        }


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


def _target_coordinate_layout(
    irreps: o3.Irreps,
) -> tuple[tuple[int, o3.Irrep, int, int], ...]:
    """Return ``(coordinate, irrep, multiplicity_slot, component)`` entries."""
    layout: list[tuple[int, o3.Irrep, int, int]] = []
    cursor = 0
    next_slot: dict[str, int] = {}
    for multiplicity, irrep in irreps:
        name = str(irrep)
        first_slot = next_slot.get(name, 0)
        for local_multiplicity in range(int(multiplicity)):
            for component in range(irrep.dim):
                layout.append((cursor, irrep, first_slot + local_multiplicity, component))
                cursor += 1
        next_slot[name] = first_slot + int(multiplicity)
    return tuple(layout)


def _layout_by_irrep(
    layout: tuple[tuple[int, o3.Irrep, int, int], ...],
) -> dict[str, tuple[tuple[int, int, int], ...]]:
    grouped: dict[str, list[tuple[int, int, int]]] = {}
    for coordinate, irrep, local_multiplicity, component in layout:
        name = str(irrep)
        grouped.setdefault(name, []).append((coordinate, local_multiplicity, component))
    return {name: tuple(entries) for name, entries in grouped.items()}


def _project_to_irrep_commutant(irreps: o3.Irreps, matrix: torch.Tensor) -> torch.Tensor:
    projected = torch.zeros_like(matrix)
    grouped = _layout_by_irrep(_target_coordinate_layout(irreps))
    for entries in grouped.values():
        by_slot_component: dict[tuple[int, int], list[int]] = {}
        for coordinate, slot, component in entries:
            by_slot_component.setdefault((slot, component), []).append(coordinate)
        slots = sorted({slot for _, slot, _ in entries})
        components = sorted({component for _, _, component in entries})
        for row_slot in slots:
            for col_slot in slots:
                values = []
                for component in components:
                    rows = by_slot_component[(row_slot, component)]
                    cols = by_slot_component[(col_slot, component)]
                    values.append(matrix[rows[0], cols[0]])
                value = torch.stack(values).mean()
                for component in components:
                    row = by_slot_component[(row_slot, component)][0]
                    col = by_slot_component[(col_slot, component)][0]
                    projected[row, col] = value
    return projected


def _invariant_coordinate_indices(irreps: o3.Irreps) -> tuple[int, ...]:
    return tuple(
        coordinate
        for coordinate, irrep, _, _ in _target_coordinate_layout(irreps)
        if irrep.l == 0 and irrep.p == 1
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
    ) -> FeatureSpec:
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
    ) -> FeatureSpec:
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
