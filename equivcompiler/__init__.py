"""Public API for representation-compiled probabilistic outputs."""

from equivcompiler.api import (
    compile_predictor,
    compile_readout,
)
from equivcompiler.checkpoint import convert_checkpoint
from equivcompiler.modules import CompiledProbabilisticReadout
from equivcompiler.planning import CompilationPlan, plan_readout
from equivcompiler.policies import (
    AutoBudget,
    ExactOnly,
    FullCovariance,
    GraphPrecision,
    IsotypicBlockCovariance,
    LowRankCovariance,
    TruncatedMultiplicityRank,
)
from equivcompiler.specs import FeatureSpec, OutputSemantics, describe_output
from representations import (
    CompilationCertificate,
    CompilationError,
    CompilationReport,
    UnreachableTargetError,
)

__all__ = [
    "CompiledProbabilisticReadout",
    "CompilationCertificate",
    "CompilationError",
    "CompilationPlan",
    "CompilationReport",
    "UnreachableTargetError",
    "FeatureSpec",
    "OutputSemantics",
    "AutoBudget",
    "ExactOnly",
    "FullCovariance",
    "GraphPrecision",
    "IsotypicBlockCovariance",
    "LowRankCovariance",
    "TruncatedMultiplicityRank",
    "compile_predictor",
    "compile_readout",
    "convert_checkpoint",
    "describe_output",
    "plan_readout",
]
