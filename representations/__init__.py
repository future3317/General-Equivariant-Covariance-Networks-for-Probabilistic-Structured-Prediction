"""Representation layer: orthogonal representations and symmetric squares."""

from representations.base import OrthogonalRepresentationSpec, SymmetricSquareSpec
from representations.symmetric_square import O3SymmetricOperatorBasis, symmetric_square_irreps
from representations.o3_irreps import O3IrrepsSpec
from representations.cartesian_outputs import (
    rank2_symmetric_irreps,
    rank4_elasticity_irreps,
)
from representations.adaptive_lifting import (
    O3AdaptiveLifting,
    O3LiftingPlan,
    LiftingStage,
    coverage_deficit,
    direct_sum_irreps,
    irrep_multiplicities,
    plan_lifting_graph,
    required_lifting_depth,
)
from representations.compiler import (
    CompilerConfig,
    O3Compilation,
    O3CompiledOutputHead,
    O3RepresentationCompiler,
)
from representations.graph_structure import EquivariantOutputGraph

__all__ = [
    "OrthogonalRepresentationSpec",
    "SymmetricSquareSpec",
    "O3IrrepsSpec",
    "O3SymmetricOperatorBasis",
    "symmetric_square_irreps",
    "rank2_symmetric_irreps",
    "rank4_elasticity_irreps",
    "O3AdaptiveLifting",
    "O3LiftingPlan",
    "LiftingStage",
    "coverage_deficit",
    "direct_sum_irreps",
    "irrep_multiplicities",
    "plan_lifting_graph",
    "required_lifting_depth",
    "CompilerConfig",
    "O3Compilation",
    "O3CompiledOutputHead",
    "O3RepresentationCompiler",
    "EquivariantOutputGraph",
]
