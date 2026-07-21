"""Runtime plugins that lower structured operator IR to PyTorch modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from compatibility.e3nn import o3

from representations.cartesian_stf import Rank2CartesianSTFOperatorBasis
from representations.irrep_layout import RepeatedIrrepLayout
from representations.o3_irreps import O3IrrepsSpec

if TYPE_CHECKING:
    from representations.compiler import O3Compilation, O3CompiledOutputHead


class OperatorRuntime(ABC):
    """Executable lowering plugin for one semantic operator family."""

    @abstractmethod
    def install(self, head: "O3CompiledOutputHead") -> None:
        """Install projection modules while preserving stable checkpoint names."""

    @abstractmethod
    def project(
        self, head: "O3CompiledOutputHead", compiled: torch.Tensor
    ) -> torch.Tensor:
        """Project compiled features to raw operator parameters."""

    @abstractmethod
    def build_spd_map(self, compilation: "O3Compilation") -> torch.nn.Module:
        """Build the family-specific scatter or precision oracle."""


class FullScatterRuntime(OperatorRuntime):
    def install(self, head: "O3CompiledOutputHead") -> None:
        compilation = head.compilation
        head.operator_basis = (
            Rank2CartesianSTFOperatorBasis()
            if compilation.backend == "cartesian_stf"
            else head.output_spec.symmetric_square()
        )
        head.covariance_projection = o3.Linear(
            compilation.active_target_irreps, head.operator_basis.operator_irreps
        )

    def project(
        self, head: "O3CompiledOutputHead", compiled: torch.Tensor
    ) -> torch.Tensor:
        return head.operator_basis.assemble(head.covariance_projection(compiled))

    def build_spd_map(self, compilation: "O3Compilation") -> torch.nn.Module:
        from spd_maps import MatrixExponentialMap

        return MatrixExponentialMap()


class LowRankScatterRuntime(OperatorRuntime):
    def install(self, head: "O3CompiledOutputHead") -> None:
        compilation = head.compilation
        head.factor_layout = RepeatedIrrepLayout(
            head.output_spec.irreps, int(compilation.covariance_rank)
        )
        head.factor_irreps = head.factor_layout.expanded_irreps
        head.covariance_projection = o3.Linear(
            compilation.active_target_irreps, head.factor_irreps
        )
        head.scale_projection = o3.Linear(
            compilation.active_target_irreps, o3.Irreps("1x0e")
        )

    def project(
        self, head: "O3CompiledOutputHead", compiled: torch.Tensor
    ) -> torch.Tensor:
        factors = head.factor_layout.pack(
            head.covariance_projection(compiled)
        ).transpose(-1, -2)
        log_sigma2 = head.scale_projection(compiled)
        return torch.cat([factors.flatten(start_dim=-2), log_sigma2], dim=-1)

    def build_spd_map(self, compilation: "O3Compilation") -> torch.nn.Module:
        from spd_maps import LowRankPlusIsotropicMap

        return LowRankPlusIsotropicMap(
            dim=compilation.output_spec.dim,
            rank=int(compilation.covariance_rank),
        )


class IsotypicBlockRuntime(OperatorRuntime):
    def install(self, head: "O3CompiledOutputHead") -> None:
        count = head.compilation.covariance_parameter_count
        head.covariance_projection = o3.Linear(
            head.compilation.active_target_irreps, o3.Irreps(f"{count}x0e")
        )

    def project(
        self, head: "O3CompiledOutputHead", compiled: torch.Tensor
    ) -> torch.Tensor:
        return head.covariance_projection(compiled)

    def build_spd_map(self, compilation: "O3Compilation") -> torch.nn.Module:
        from spd_maps import IsotypicBlockMap

        return IsotypicBlockMap(compilation.output_spec.irreps)


class GraphPrecisionRuntime(OperatorRuntime):
    def install(self, head: "O3CompiledOutputHead") -> None:
        graph = head.compilation.operator_family.graph
        if graph is None:
            raise RuntimeError("graph operator plan is missing its graph structure")
        local_irreps = o3.Irreps([(1, graph.node_irrep)])
        head.operator_basis = O3IrrepsSpec(local_irreps).symmetric_square()
        head.potential_layout = RepeatedIrrepLayout(
            head.operator_basis.operator_irreps, graph.num_potentials
        )
        head.potential_irreps = head.potential_layout.expanded_irreps
        head.covariance_projection = o3.Linear(
            head.compilation.active_target_irreps, head.potential_irreps
        )

    def project(
        self, head: "O3CompiledOutputHead", compiled: torch.Tensor
    ) -> torch.Tensor:
        coefficients = head.covariance_projection(compiled)
        return head.operator_basis.assemble(head.potential_layout.pack(coefficients))

    def build_spd_map(self, compilation: "O3Compilation") -> torch.nn.Module:
        from spd_maps import GraphStructuredPrecisionMap

        graph = compilation.operator_family.graph
        if graph is None:
            raise RuntimeError("graph operator plan is missing its graph structure")
        return GraphStructuredPrecisionMap(graph)


_RUNTIMES: dict[str, OperatorRuntime] = {
    "full": FullScatterRuntime(),
    "low_rank": LowRankScatterRuntime(),
    "block": IsotypicBlockRuntime(),
    "graph": GraphPrecisionRuntime(),
}


def operator_runtime(kind: str) -> OperatorRuntime:
    try:
        return _RUNTIMES[kind]
    except KeyError as error:
        raise ValueError(f"no registered operator runtime for family {kind!r}") from error
