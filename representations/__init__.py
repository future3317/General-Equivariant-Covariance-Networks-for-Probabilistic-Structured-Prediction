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
    O3ReachabilityAnalysis,
    LiftingStage,
    analyze_lifting_graph,
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
from representations.cartesian_stf import (
    MultiplicityFirstCartesianTensorSquare,
    Rank2CartesianSTFOperatorBasis,
    is_rank2_stf_output,
    supports_cartesian_stf_seed,
)
from representations.dense_projector import MultiplicityFirstDenseTensorProduct
from representations.diagnostics import (
    CompilationCertificate,
    CompilationError,
    UnreachableActiveTargetError,
    UnreachableTargetError,
)
from representations.representation_ir import (
    CoordinateSpec,
    DecomposedRep,
    DirectSumExpr,
    InnerProductRep,
    InvariantMetricSpec,
    IrrepsExpr,
    RepExpr,
    RepeatedExpr,
    SymmetricSquareExpr,
    TrivialScalarsExpr,
)
from representations.operator_ir import (
    FamilyRelation,
    OperatorFamilyPlan,
    OperatorIR,
)
from representations.report import CompilationReport, build_compilation_report

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
    "O3ReachabilityAnalysis",
    "LiftingStage",
    "analyze_lifting_graph",
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
    "MultiplicityFirstCartesianTensorSquare",
    "MultiplicityFirstDenseTensorProduct",
    "Rank2CartesianSTFOperatorBasis",
    "is_rank2_stf_output",
    "supports_cartesian_stf_seed",
    "CompilationCertificate",
    "CompilationError",
    "UnreachableActiveTargetError",
    "UnreachableTargetError",
    "RepExpr",
    "DecomposedRep",
    "IrrepsExpr",
    "DirectSumExpr",
    "SymmetricSquareExpr",
    "RepeatedExpr",
    "TrivialScalarsExpr",
    "InvariantMetricSpec",
    "CoordinateSpec",
    "InnerProductRep",
    "FamilyRelation",
    "OperatorIR",
    "OperatorFamilyPlan",
    "CompilationReport",
    "build_compilation_report",
]
