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
    ExactExecutorCandidates,
    ExactOnly,
    ExecutorMeasurement,
    FirstFeasible,
    FullCovariance,
    GraphPrecision,
    IsotypicBlockCovariance,
    LowRankCovariance,
    MinimizeLatency,
    MinParameterCount,
    OperatorFamilySpec,
    PreferExecutor,
    ShapeSignature,
    SpecificExecutor,
    TruncatedMultiplicityRank,
)
from equivcompiler.distributions import (
    DistributionSpec,
    EllipticalDistribution,
)
from equivcompiler.autotune import BenchmarkTask, DeviceAutotuner
from equivcompiler.executors import (
    CandidateEnumerator,
    ExactLoweringRegistry,
    ExecutionContext,
)
from equivcompiler.specs import FeatureSpec, OutputSemantics, describe_output
from representations import (
    CompilationCertificate,
    CompilationError,
    CompilationReport,
    UnreachableTargetError,
    UnreachableActiveTargetError,
)

__all__ = [
    "CompiledProbabilisticReadout",
    "CompilationCertificate",
    "CompilationError",
    "CompilationPlan",
    "CompilationReport",
    "UnreachableTargetError",
    "UnreachableActiveTargetError",
    "FeatureSpec",
    "OutputSemantics",
    "AutoBudget",
    "FirstFeasible",
    "MinParameterCount",
    "OperatorFamilySpec",
    "ExactOnly",
    "ExactExecutorCandidates",
    "SpecificExecutor",
    "PreferExecutor",
    "MinimizeLatency",
    "ShapeSignature",
    "ExecutorMeasurement",
    "FullCovariance",
    "GraphPrecision",
    "IsotypicBlockCovariance",
    "LowRankCovariance",
    "TruncatedMultiplicityRank",
    "DistributionSpec",
    "EllipticalDistribution",
    "DeviceAutotuner",
    "BenchmarkTask",
    "ExactLoweringRegistry",
    "CandidateEnumerator",
    "ExecutionContext",
    "compile_predictor",
    "compile_readout",
    "convert_checkpoint",
    "describe_output",
    "plan_readout",
]
