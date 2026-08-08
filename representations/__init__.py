"""Representation layer: orthogonal representations and symmetric squares."""

from representations.adaptive_lifting import (
    LiftingStage,
    O3AdaptiveLifting,
    O3LiftingPlan,
    O3ReachabilityAnalysis,
    analyze_lifting_graph,
    coverage_deficit,
    direct_sum_irreps,
    irrep_multiplicities,
    plan_lifting_graph,
    required_lifting_depth,
)
from representations.base import OrthogonalRepresentationSpec, SymmetricSquareSpec
from representations.cartesian_outputs import (
    rank2_symmetric_irreps,
    rank4_elasticity_irreps,
)
from representations.cartesian_stf import (
    MultiplicityFirstCartesianTensorSquare,
    Rank2CartesianSTFOperatorBasis,
    is_rank2_stf_output,
    supports_cartesian_stf_seed,
)
from representations.compiler import (
    LoweringConfig,
    O3Compilation,
    O3CompiledOutputHead,
    O3ProgramCompiler,
)
from representations.dense_projector import MultiplicityFirstDenseTensorProduct
from representations.diagnostics import (
    CompilationCertificate,
    CompilationError,
    UnreachableActiveTargetError,
    UnreachableTargetError,
)
from representations.exterior_square import O3SkewOperatorBasis, exterior_square_irreps
from representations.graph_structure import EquivariantOutputGraph
from representations.o3_irreps import O3IrrepsSpec
from representations.operator_ir import (
    Equivariance,
    FamilyRelation,
    OperatorFamilyPlan,
    OperatorIR,
    OperatorVerificationContext,
    ParameterBinding,
    Positivity,
)
from representations.finite_precision import (
    NumericalSPDCertificate,
    certify_numerical_spd,
)
from representations.operator_lowering import OptimizationCertificate
from representations.report import CompilationReport, build_compilation_report
from representations.representation_ir import (
    CoordinateSpec,
    DecomposedRep,
    DirectSumExpr,
    ExteriorSquareExpr,
    InnerProductRep,
    InvariantMetricSpec,
    IrrepsExpr,
    RepeatedExpr,
    RepExpr,
    SymmetricSquareExpr,
    TrivialScalarsExpr,
)
from representations.symmetric_square import (
    O3SymmetricOperatorBasis,
    symmetric_square_irreps,
)

__all__ = [
    "CompilationCertificate",
    "CompilationError",
    "CompilationReport",
    "CoordinateSpec",
    "DecomposedRep",
    "DirectSumExpr",
    "Equivariance",
    "EquivariantOutputGraph",
    "ExteriorSquareExpr",
    "FamilyRelation",
    "NumericalSPDCertificate",
    "certify_numerical_spd",
    "InnerProductRep",
    "InvariantMetricSpec",
    "IrrepsExpr",
    "LiftingStage",
    "LoweringConfig",
    "MultiplicityFirstCartesianTensorSquare",
    "MultiplicityFirstDenseTensorProduct",
    "O3AdaptiveLifting",
    "O3Compilation",
    "O3CompiledOutputHead",
    "O3IrrepsSpec",
    "O3LiftingPlan",
    "O3ProgramCompiler",
    "O3ReachabilityAnalysis",
    "O3SkewOperatorBasis",
    "O3SymmetricOperatorBasis",
    "OperatorFamilyPlan",
    "OperatorIR",
    "OperatorVerificationContext",
    "OptimizationCertificate",
    "OrthogonalRepresentationSpec",
    "ParameterBinding",
    "Positivity",
    "Rank2CartesianSTFOperatorBasis",
    "RepExpr",
    "RepeatedExpr",
    "SymmetricSquareExpr",
    "SymmetricSquareSpec",
    "TrivialScalarsExpr",
    "UnreachableActiveTargetError",
    "UnreachableTargetError",
    "analyze_lifting_graph",
    "build_compilation_report",
    "coverage_deficit",
    "direct_sum_irreps",
    "exterior_square_irreps",
    "irrep_multiplicities",
    "is_rank2_stf_output",
    "plan_lifting_graph",
    "rank2_symmetric_irreps",
    "rank4_elasticity_irreps",
    "required_lifting_depth",
    "supports_cartesian_stf_seed",
    "symmetric_square_irreps",
]
