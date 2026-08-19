"""Public API for representation-compiled probabilistic outputs."""

from equivcompiler.api import (
    compile_predictor,
    compile_readout,
)
from equivcompiler.autotune import BenchmarkTask, DeviceAutotuner
from equivcompiler.checkpoint import convert_checkpoint
from equivcompiler.distributions import (
    ConditionalScaleStudentTRadial,
    ConditionalStudentTRadial,
    DistributionSpec,
    EllipticalDistribution,
    GaussianRadial,
    RadialLaw,
    StudentTRadial,
)
from equivcompiler.executors import (
    CandidateEnumerator,
    ExactLoweringRegistry,
    ExecutionContext,
)
from equivcompiler.modules import CompiledProbabilisticReadout
from equivcompiler.planning import CompilationPlan, plan_readout
from equivcompiler.policies import (
    AsinhExponentialCovariance,
    AutoBudget,
    CenteredSpectralWindowCovariance,
    ExactExecutorCandidates,
    ExactOnly,
    ExecutionSignature,
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
    SpecificExecutor,
    SpectralWindowCovariance,
    TruncatedMultiplicityRank,
)
from equivcompiler.signatures import execution_signature_for_plan
from equivcompiler.specs import (
    FeatureSpec,
    OutputSemantics,
    TargetTransform,
    describe_output,
)
from representations import (
    CompilationCertificate,
    CompilationError,
    CompilationReport,
    UnreachableActiveTargetError,
    UnreachableTargetError,
)

__all__ = [
    "AsinhExponentialCovariance",
    "AutoBudget",
    "BenchmarkTask",
    "CandidateEnumerator",
    "CenteredSpectralWindowCovariance",
    "CompilationCertificate",
    "CompilationError",
    "CompilationPlan",
    "CompilationReport",
    "CompiledProbabilisticReadout",
    "ConditionalScaleStudentTRadial",
    "ConditionalStudentTRadial",
    "DeviceAutotuner",
    "DistributionSpec",
    "EllipticalDistribution",
    "ExactExecutorCandidates",
    "ExactLoweringRegistry",
    "ExactOnly",
    "ExecutionContext",
    "ExecutionSignature",
    "ExecutorMeasurement",
    "FeatureSpec",
    "FirstFeasible",
    "FullCovariance",
    "GaussianRadial",
    "GraphPrecision",
    "IsotypicBlockCovariance",
    "LowRankCovariance",
    "MinParameterCount",
    "MinimizeLatency",
    "OperatorFamilySpec",
    "OutputSemantics",
    "PreferExecutor",
    "RadialLaw",
    "SpecificExecutor",
    "SpectralWindowCovariance",
    "StudentTRadial",
    "TargetTransform",
    "TruncatedMultiplicityRank",
    "UnreachableActiveTargetError",
    "UnreachableTargetError",
    "compile_predictor",
    "compile_readout",
    "convert_checkpoint",
    "describe_output",
    "execution_signature_for_plan",
    "plan_readout",
]
