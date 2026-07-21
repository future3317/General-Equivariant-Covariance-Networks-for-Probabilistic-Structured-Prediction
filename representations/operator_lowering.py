"""Recursive lowering of verified Operator IR to executable PyTorch modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
from compatibility.e3nn import o3

from representations.cartesian_stf import Rank2CartesianSTFOperatorBasis
from representations.irrep_layout import RepeatedIrrepLayout
from representations.operator_ir import OperatorFamilyPlan, OperatorIR
from representations.symmetric_square import O3SymmetricOperatorBasis
from spd_maps.base import SPDMap, symmetrize

if TYPE_CHECKING:
    from representations.compiler import O3Compilation, O3CompiledOutputHead


def _batched_block_diag(blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not blocks:
        raise ValueError("direct_sum requires at least one runtime block")
    leading = blocks[0].shape[:-2]
    if any(block.shape[:-2] != leading for block in blocks):
        raise ValueError("direct_sum blocks must have identical batch dimensions")
    size = sum(block.shape[-1] for block in blocks)
    result = blocks[0].new_zeros((*leading, size, size))
    cursor = 0
    for block in blocks:
        width = block.shape[-1]
        result[..., cursor : cursor + width, cursor : cursor + width] = block
        cursor += width
    return result


class RecursiveOperatorMap(SPDMap):
    """Generic interpreter for any verified composition of registered primitives."""

    def __init__(self, compilation: "O3Compilation"):
        super().__init__()
        self.program = compilation.operator_family.assembly
        self.domain = compilation.operator_family.domain
        self.output_dim = compilation.output_spec.dim
        self.graph = compilation.operator_family.graph
        self._binding_slices: dict[str, slice] = {}
        cursor = 0
        for binding in compilation.operator_family.parameter_bindings:
            self._binding_slices[binding.name] = slice(cursor, cursor + binding.dimension)
            cursor += binding.dimension
        self.parameter_count = cursor

        output_basis = O3SymmetricOperatorBasis(compilation.output_spec.irreps).basis
        self.register_buffer("_output_basis", output_basis, persistent=False)
        if self.graph is not None:
            local_irreps = o3.Irreps([(1, self.graph.node_irrep)])
            local_basis = O3SymmetricOperatorBasis(local_irreps).basis
            self.register_buffer("_local_basis", local_basis, persistent=False)
            incidence = self.graph.incidence_matrix()
            identity = torch.eye(self.graph.block_dim, dtype=incidence.dtype)
            self.register_buffer(
                "_graph_coboundary", torch.kron(incidence, identity), persistent=False
            )

    def _parameter(self, node: OperatorIR, params: torch.Tensor) -> torch.Tensor:
        attributes = node.attribute_dict()
        value = params[..., self._binding_slices[str(attributes["binding"])]]
        if attributes.get("coordinate_layout", "native") == "repeated_irrep":
            layout = RepeatedIrrepLayout(
                str(attributes["unit_irreps"]), int(attributes["copies"])
            )
            value = layout.pack(value).flatten(start_dim=-2)
        start = int(attributes.get("start", 0))
        stop = attributes.get("stop")
        return value[..., start : int(stop) if stop is not None else None]

    def _evaluate(self, node: OperatorIR, params: torch.Tensor) -> torch.Tensor:
        attributes = node.attribute_dict()
        children = tuple(self._evaluate(child, params) for child in node.inputs)
        if node.kind == "parameter":
            return self._parameter(node, params)
        if node.kind == "symmetric_operator":
            coefficients = children[0]
            space = attributes.get("coordinate_space")
            if space == "output_representation":
                return symmetrize(
                    torch.einsum("...q,qij->...ij", coefficients, self._output_basis)
                )
            if space == "graph_local":
                copies = int(attributes.get("copies", 1))
                if copies == 0:
                    return coefficients.new_zeros(
                        (*coefficients.shape[:-1], 0, self.graph.block_dim, self.graph.block_dim)
                    )
                local_count = self._local_basis.shape[0]
                reshaped = coefficients.reshape(
                    *coefficients.shape[:-1], copies, local_count
                )
                return symmetrize(
                    torch.einsum("...nq,qij->...nij", reshaped, self._local_basis)
                )
            raise ValueError(f"unknown symmetric coordinate space: {space!r}")
        if node.kind == "equivariant_factor":
            rank = int(attributes["rank"])
            layout = RepeatedIrrepLayout(attributes["output_irreps"], rank)
            return layout.pack(children[0]).transpose(-1, -2)
        if node.kind == "positive_scalar_identity":
            scalar = torch.nn.functional.softplus(children[0][..., 0]) + float(
                attributes.get("minimum", 1e-4)
            )
            identity = torch.eye(
                int(attributes["dimension"]),
                dtype=scalar.dtype,
                device=scalar.device,
            )
            return scalar[..., None, None] * identity
        if node.kind == "cholesky_positive":
            dimension = int(attributes["dimension"])
            lower = children[0].new_zeros(
                (*children[0].shape[:-1], dimension, dimension)
            )
            rows, cols = torch.tril_indices(
                dimension, dimension, device=children[0].device
            )
            lower[..., rows, cols] = children[0]
            diagonal = torch.arange(dimension, device=children[0].device)
            lower[..., diagonal, diagonal] = (
                torch.nn.functional.softplus(lower[..., diagonal, diagonal])
                + float(attributes.get("minimum", 1e-4))
            )
            return lower @ lower.transpose(-1, -2)
        if node.kind == "spectral_positive":
            if attributes["map"] != "matrix_exponential":
                raise ValueError(
                    "multiplicity_cholesky is represented by cholesky_positive"
                )
            return torch.linalg.matrix_exp(symmetrize(children[0]))
        if node.kind == "gram":
            return children[0] @ children[0].transpose(-1, -2)
        if node.kind == "kronecker_identity":
            identity = torch.eye(
                int(attributes["irrep_dimension"]),
                dtype=children[0].dtype,
                device=children[0].device,
            )
            return torch.einsum("...ab,ij->...aibj", children[0], identity).reshape(
                *children[0].shape[:-2],
                children[0].shape[-1] * identity.shape[0],
                children[0].shape[-1] * identity.shape[0],
            )
        if node.kind == "direct_sum":
            copies = attributes.get("copies")
            if copies is not None:
                matrices = children[0]
                return _batched_block_diag(
                    tuple(matrices[..., index, :, :] for index in range(int(copies)))
                ) if int(copies) else matrices.new_zeros(
                    (*matrices.shape[:-3], 0, 0)
                )
            return _batched_block_diag(children)
        if node.kind == "pullback":
            if attributes["intertwiner"] != "homogeneous_graph_coboundary":
                raise ValueError("unregistered runtime intertwiner")
            coboundary = self._graph_coboundary.to(
                dtype=children[0].dtype, device=children[0].device
            )
            return coboundary.transpose(-1, -2) @ children[0] @ coboundary
        if node.kind == "add":
            return sum(children[1:], children[0])
        raise ValueError(f"no primitive lowering registered for {node.kind!r}")

    def _operator(self, params: torch.Tensor) -> torch.Tensor:
        if params.shape[-1] != self.parameter_count:
            raise ValueError(
                f"params last dim {params.shape[-1]} != {self.parameter_count}"
            )
        return symmetrize(self._evaluate(self.program, params))

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        operator = self._operator(params)
        if self.domain == "scatter":
            return operator
        return torch.cholesky_inverse(torch.linalg.cholesky(operator))

    def precision(self, params: torch.Tensor) -> torch.Tensor:
        operator = self._operator(params)
        if self.domain == "precision":
            return operator
        return torch.cholesky_inverse(torch.linalg.cholesky(operator))

    def statistics(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        operator = self._operator(params)
        cholesky = torch.linalg.cholesky(operator)
        logdet_operator = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(-1)
        if self.domain == "precision":
            quadratic = torch.einsum("...i,...ij,...j->...", residual, operator, residual)
            return -logdet_operator, quadratic
        solved = torch.cholesky_solve(residual.unsqueeze(-1), cholesky).squeeze(-1)
        return logdet_operator, torch.sum(residual * solved, dim=-1)

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        dummy = params.new_zeros((*params.shape[:-1], self.output_dim))
        return self.statistics(params, dummy)[0]

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return self.statistics(params, residual)[1]

    def sample(
        self, mean: torch.Tensor, params: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        covariance = self(params)
        cholesky = torch.linalg.cholesky(covariance)
        noise = torch.randn(
            *mean.shape[:-1],
            self.output_dim,
            num_samples,
            dtype=mean.dtype,
            device=mean.device,
        )
        return mean.unsqueeze(-1) + cholesky @ noise


class OptimizedProgramMap(SPDMap):
    """IR-pattern optimization preserving the generic parameter contract."""

    def __init__(
        self,
        compilation: "O3Compilation",
        delegate: SPDMap,
        transform: Callable[[torch.Tensor], torch.Tensor],
        optimization_name: str,
    ):
        super().__init__()
        self.delegate = delegate
        self._transform = transform
        self.optimization_name = optimization_name
        self.output_dim = compilation.output_spec.dim

    def _transform_parameters(self, params: torch.Tensor) -> torch.Tensor:
        return self._transform(params)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.delegate(self._transform_parameters(params))

    def precision(self, params: torch.Tensor) -> torch.Tensor:
        return self.delegate.precision(self._transform_parameters(params))

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        return self.delegate.logdet(self._transform_parameters(params))

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return self.delegate.precision_action(
            self._transform_parameters(params), residual
        )

    def statistics(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.delegate.statistics(self._transform_parameters(params), residual)

    def sample(
        self, mean: torch.Tensor, params: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        transformed = self._transform_parameters(params)
        sampler = getattr(self.delegate, "sample", None)
        if sampler is not None:
            return sampler(mean, transformed, num_samples)
        covariance = self.delegate(transformed)
        cholesky = torch.linalg.cholesky(covariance)
        noise = torch.randn(
            *mean.shape[:-1],
            self.output_dim,
            num_samples,
            dtype=mean.dtype,
            device=mean.device,
        )
        return mean.unsqueeze(-1) + cholesky @ noise


def _binding_slices(family: OperatorFamilyPlan) -> dict[str, slice]:
    result: dict[str, slice] = {}
    cursor = 0
    for binding in family.parameter_bindings:
        result[binding.name] = slice(cursor, cursor + binding.dimension)
        cursor += binding.dimension
    return result


def _try_optimized_map(compilation: "O3Compilation") -> OptimizedProgramMap | None:
    from spd_maps import (
        GraphStructuredPrecisionMap,
        IsotypicBlockMap,
        LowRankPlusIsotropicMap,
        MatrixExponentialMap,
    )

    family = compilation.operator_family
    root = family.assembly
    slices = _binding_slices(family)
    if (
        root.kind == "spectral_positive"
        and len(root.inputs) == 1
        and root.inputs[0].kind == "symmetric_operator"
        and root.attribute_dict().get("map") == "matrix_exponential"
    ):
        basis = O3SymmetricOperatorBasis(compilation.output_spec.irreps).basis

        def full_transform(params: torch.Tensor) -> torch.Tensor:
            coefficients = params[..., slices["operator"]]
            return symmetrize(torch.einsum("...q,qij->...ij", coefficients, basis.to(params)))

        return OptimizedProgramMap(
            compilation,
            MatrixExponentialMap(),
            full_transform,
            "spectral_trace_and_exponential_oracle",
        )

    child_kinds = {child.kind for child in root.inputs}
    if root.kind == "add" and child_kinds == {"gram", "positive_scalar_identity"}:
        rank = int(family.rank or 0)
        factor_layout = RepeatedIrrepLayout(compilation.output_spec.irreps, rank)

        def low_rank_transform(params: torch.Tensor) -> torch.Tensor:
            factors = factor_layout.pack(params[..., slices["factor"]]).transpose(-1, -2)
            scale = params[..., slices["scale"]]
            return torch.cat([factors.flatten(start_dim=-2), scale], dim=-1)

        return OptimizedProgramMap(
            compilation,
            LowRankPlusIsotropicMap(compilation.output_spec.dim, rank),
            low_rank_transform,
            "woodbury_and_determinant_lemma_oracle",
        )

    if root.kind == "direct_sum" and all(
        child.kind == "kronecker_identity" for child in root.inputs
    ):
        return OptimizedProgramMap(
            compilation,
            IsotypicBlockMap(compilation.output_spec.irreps),
            lambda params: params[..., slices["blocks"]],
            "multiplicity_block_oracle",
        )

    if family.graph is not None and "pullback" in root.verify().instructions:
        graph = family.graph
        local_irreps = o3.Irreps([(1, graph.node_irrep)])
        basis = O3SymmetricOperatorBasis(local_irreps).basis
        layout = RepeatedIrrepLayout(
            O3SymmetricOperatorBasis(local_irreps).operator_irreps,
            graph.num_potentials,
        )

        def graph_transform(params: torch.Tensor) -> torch.Tensor:
            coefficients = layout.pack(params[..., slices["potentials"]])
            return symmetrize(
                torch.einsum("...nq,qij->...nij", coefficients, basis.to(params))
            )

        return OptimizedProgramMap(
            compilation,
            GraphStructuredPrecisionMap(graph),
            graph_transform,
            "graph_elimination_or_dense_precision_oracle",
        )
    return None


@dataclass(frozen=True)
class PrimitiveLowering:
    """One registered recursive-lowering rule."""

    op: str
    validate: Callable[[OperatorIR], None]


class PrimitiveLoweringRegistry:
    """Closed registry used both for coverage analysis and materialization."""

    def __init__(self) -> None:
        self._rules: dict[str, PrimitiveLowering] = {}

    def register(self, op: str, validate: Callable[[OperatorIR], None]) -> None:
        if op in self._rules:
            raise ValueError(f"primitive lowering {op!r} is already registered")
        self._rules[op] = PrimitiveLowering(op, validate)

    def analyze(self, node: OperatorIR) -> tuple[str, ...]:
        covered: list[str] = []

        def visit(current: OperatorIR) -> None:
            try:
                rule = self._rules[current.kind]
            except KeyError as error:
                raise ValueError(
                    f"no primitive lowering registered for {current.kind!r}"
                ) from error
            for child in current.inputs:
                visit(child)
            rule.validate(current)
            covered.append(current.kind)

        visit(node)
        return tuple(covered)

    def materialize(self, compilation: "O3Compilation") -> RecursiveOperatorMap:
        self.analyze(compilation.operator_family.assembly)
        optimized = _try_optimized_map(compilation)
        return optimized if optimized is not None else RecursiveOperatorMap(compilation)


def _verified_node(node: OperatorIR) -> None:
    certificate = node.verify()
    if not certificate.valid:
        raise ValueError(f"invalid operator instruction {node.kind}: {certificate.errors}")


DEFAULT_PRIMITIVE_LOWERINGS = PrimitiveLoweringRegistry()
for _op in (
    "parameter",
    "symmetric_operator",
    "equivariant_factor",
    "positive_scalar_identity",
    "cholesky_positive",
    "spectral_positive",
    "gram",
    "kronecker_identity",
    "direct_sum",
    "pullback",
    "add",
):
    DEFAULT_PRIMITIVE_LOWERINGS.register(_op, _verified_node)


def install_parameter_projections(head: "O3CompiledOutputHead") -> None:
    compilation = head.compilation
    for binding in compilation.operator_family.parameter_bindings:
        if hasattr(head, binding.projection_name):
            raise ValueError(f"duplicate projection module name {binding.projection_name!r}")
        setattr(
            head,
            binding.projection_name,
            o3.Linear(compilation.active_target_irreps, binding.irreps),
        )

    # Preserve the deterministic basis key used by exact spherical/STF head
    # checkpoint conversion even though assembly now lives in the IR map.
    root = compilation.operator_family.assembly
    if root.kind == "spectral_positive" and root.inputs[0].kind == "symmetric_operator":
        head.operator_basis = (
            Rank2CartesianSTFOperatorBasis()
            if compilation.backend == "cartesian_stf"
            else compilation.output_spec.symmetric_square()
        )


def project_parameter_bindings(
    head: "O3CompiledOutputHead", compiled: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        [
            getattr(head, binding.projection_name)(compiled)
            for binding in head.compilation.operator_family.parameter_bindings
        ],
        dim=-1,
    )


def lower_operator_program(compilation: "O3Compilation") -> SPDMap:
    """Recursively verify and materialize a family-independent operator program."""
    return DEFAULT_PRIMITIVE_LOWERINGS.materialize(compilation)
