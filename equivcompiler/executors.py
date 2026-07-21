"""Executor capability analysis and explicit lowering decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from typing import Literal

from equivcompiler.policies import ExactOnly, FidelityPolicy, TruncatedMultiplicityRank
from equivcompiler.specs import FeatureSpec
from representations import O3IrrepsSpec
from representations.adaptive_lifting import O3LiftingPlan
from representations.cartesian_stf import is_rank2_stf_output
from representations.operator_ir import Equivariance, OperatorIR
from representations.operator_lowering import DEFAULT_PRIMITIVE_LOWERINGS


ExecutorName = Literal["spherical_cg", "cartesian_stf"]


@dataclass(frozen=True)
class ExecutionContext:
    """Complete immutable program presented to an executor analyzer."""

    feature: FeatureSpec
    output: O3IrrepsSpec
    active_plan: O3LiftingPlan
    operator_program: OperatorIR
    operator_domain: Literal["scatter", "precision"]
    fidelity: FidelityPolicy


@dataclass(frozen=True)
class FidelityDecision:
    """Requested and effective contraction fidelity without silent rewriting."""

    requested: str
    effective: Literal["exact", "approximate"]
    requested_contraction_rank: int | None
    effective_contraction_rank: int | None
    normalization_reason: str | None = None

    @property
    def exact(self) -> bool:
        return self.effective == "exact"

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "requested_contraction_rank": self.requested_contraction_rank,
            "effective_contraction_rank": self.effective_contraction_rank,
            "normalization_reason": self.normalization_reason,
        }


@dataclass(frozen=True)
class CapabilityCertificate:
    """Planning-time proof that one executor covers a concrete program."""

    executor: ExecutorName
    supported: bool
    covered_instructions: tuple[str, ...]
    missing_instructions: tuple[str, ...]
    reasons: tuple[str, ...]
    checkpoint_map: Literal["bijective", "not_available"]
    fidelity: FidelityDecision
    feature_fingerprint: str
    active_plan_hash: str
    operator_program_hash: str

    @property
    def exact(self) -> bool:
        return self.supported and self.fidelity.exact

    def as_dict(self) -> dict:
        return {
            "executor": self.executor,
            "supported": self.supported,
            "covered_instructions": list(self.covered_instructions),
            "missing_instructions": list(self.missing_instructions),
            "reasons": list(self.reasons),
            "checkpoint_map": self.checkpoint_map,
            "exact": self.exact,
            "fidelity": self.fidelity.as_dict(),
            "feature_fingerprint": self.feature_fingerprint,
            "active_plan_hash": self.active_plan_hash,
            "operator_program_hash": self.operator_program_hash,
        }


@dataclass(frozen=True)
class ExecutorDecision:
    """Selected executor and its capability proof."""

    name: ExecutorName
    capability: CapabilityCertificate
    selection_basis: dict

    def __post_init__(self) -> None:
        if not self.capability.supported or self.capability.executor != self.name:
            raise ValueError("executor decision requires a matching supported certificate")


def _cg_instruction_multiset(plan: O3LiftingPlan) -> tuple[str, ...]:
    instructions: list[str] = []
    seed_terms = [(mul, irrep) for mul, irrep in plan.seed_irreps]
    for stage in plan.stages:
        for _, left in stage.irreps_in:
            for _, right in seed_terms:
                products = set(left * right)
                for _, output in stage.irreps_out:
                    if output in products:
                        instructions.append(f"cg:{left}x{right}->{output}")
    return tuple(instructions)


def _hash(record: dict) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _context_fingerprints(context: ExecutionContext) -> tuple[str, str, str]:
    return (
        context.feature.fingerprint,
        _hash(context.active_plan.as_dict()),
        context.operator_program.fingerprint,
    )


def _maximum_exact_rank(plan: O3LiftingPlan) -> int | None:
    maximum: int | None = None
    for stage in plan.stages:
        for left_mul, left in stage.irreps_in:
            for right_mul, right in plan.seed_irreps:
                products = set(left * right)
                if any(output in products for _, output in stage.irreps_out):
                    rank = min(left_mul, right_mul)
                    maximum = rank if maximum is None else max(maximum, rank)
    return maximum


def _fidelity_decision(
    fidelity: FidelityPolicy, plan: O3LiftingPlan, *, supports_truncation: bool
) -> FidelityDecision:
    if isinstance(fidelity, ExactOnly):
        return FidelityDecision("exact_only", "exact", None, None)
    requested = fidelity.rank
    if not supports_truncation:
        return FidelityDecision(
            f"truncated_multiplicity_rank({requested})",
            "approximate",
            requested,
            requested,
            "executor_does_not_implement_truncated_contractions",
        )
    maximum = _maximum_exact_rank(plan)
    if maximum is None:
        return FidelityDecision(
            f"truncated_multiplicity_rank({requested})",
            "exact",
            requested,
            None,
            "active_program_has_no_tensor_product_instruction",
        )
    if requested >= maximum:
        return FidelityDecision(
            f"truncated_multiplicity_rank({requested})",
            "exact",
            requested,
            None,
            f"requested_cap_covers_maximum_exact_rank_{maximum}",
        )
    return FidelityDecision(
        f"truncated_multiplicity_rank({requested})",
        "approximate",
        requested,
        requested,
    )


class ExecutorAnalyzer(ABC):
    name: ExecutorName

    @abstractmethod
    def analyze(self, context: ExecutionContext) -> CapabilityCertificate:
        """Return a total planning-time capability certificate."""

    @staticmethod
    def _common_instructions(context: ExecutionContext) -> tuple[str, ...]:
        operator = tuple(
            f"operator:{item}"
            for item in DEFAULT_PRIMITIVE_LOWERINGS.analyze(context.operator_program)
        )
        return _cg_instruction_multiset(context.active_plan) + operator


class SphericalCGAnalyzer(ExecutorAnalyzer):
    name: ExecutorName = "spherical_cg"

    def analyze(self, context: ExecutionContext) -> CapabilityCertificate:
        reasons: list[str] = []
        if context.feature.group != "O3":
            reasons.append("released spherical executor requires O3 features")
        if context.feature.layout != "e3nn" or not context.feature.metric.is_orthonormal:
            reasons.append("spherical executor requires orthonormal e3nn coordinates")
        fidelity = _fidelity_decision(
            context.fidelity, context.active_plan, supports_truncation=False
        )
        if isinstance(context.fidelity, TruncatedMultiplicityRank):
            reasons.append("spherical_cg has no truncated multiplicity lowering")
        instructions = self._common_instructions(context)
        supported = not reasons
        fingerprints = _context_fingerprints(context)
        return CapabilityCertificate(
            self.name,
            supported,
            instructions if supported else (),
            () if supported else instructions,
            tuple(reasons),
            "bijective" if supported else "not_available",
            fidelity,
            *fingerprints,
        )


class CartesianSTFAnalyzer(ExecutorAnalyzer):
    name: ExecutorName = "cartesian_stf"

    def analyze(self, context: ExecutionContext) -> CapabilityCertificate:
        reasons: list[str] = []
        if context.feature.group != "O3":
            reasons.append("dense-projector executor requires O3 features")
        if context.feature.layout != "e3nn" or not context.feature.metric.is_orthonormal:
            reasons.append("dense-projector executor requires orthonormal e3nn coordinates")
        if not is_rank2_stf_output(context.output.irreps):
            reasons.append("cartesian_stf is registered for V=0e+2e")
        verification = context.operator_program.verify()
        if verification.equivariance is not Equivariance.VERIFIED:
            reasons.append("operator program lacks a verified equivariance derivation")
        if not (
            context.operator_program.kind == "spectral_positive"
            and context.operator_program.inputs
            and context.operator_program.inputs[0].kind == "symmetric_operator"
        ):
            reasons.append("cartesian_stf requires the full symmetric-operator program")
        fidelity = _fidelity_decision(
            context.fidelity, context.active_plan, supports_truncation=True
        )
        instructions = self._common_instructions(context)
        supported = not reasons
        fingerprints = _context_fingerprints(context)
        return CapabilityCertificate(
            self.name,
            supported,
            instructions if supported else (),
            () if supported else instructions,
            tuple(reasons),
            "bijective" if supported and fidelity.exact else "not_available",
            fidelity,
            *fingerprints,
        )


class ExactLoweringRegistry:
    """Registry of executor-owned whole-program analyzers."""

    def __init__(self) -> None:
        self._analyzers: dict[str, ExecutorAnalyzer] = {}

    def register(self, analyzer: ExecutorAnalyzer) -> None:
        if analyzer.name in self._analyzers:
            raise ValueError(f"executor {analyzer.name!r} is already registered")
        self._analyzers[analyzer.name] = analyzer

    def analyze(self, name: str, context: ExecutionContext) -> CapabilityCertificate:
        try:
            analyzer = self._analyzers[name]
        except KeyError as error:
            raise ValueError(f"unknown executor {name!r}") from error
        return analyzer.analyze(context)

    def as_dict(self, context: ExecutionContext) -> dict[str, dict]:
        return {
            name: analyzer.analyze(context).as_dict()
            for name, analyzer in self._analyzers.items()
        }


class CandidateEnumerator:
    """Analyze requested executors and retain only supported certificates."""

    def __init__(self, registry: ExactLoweringRegistry) -> None:
        self.registry = registry

    def enumerate(
        self, requested: tuple[str, ...], context: ExecutionContext
    ) -> tuple[CapabilityCertificate, ...]:
        certificates = tuple(
            self.registry.analyze(name, context) for name in requested
        )
        return tuple(item for item in certificates if item.supported)


DEFAULT_EXACT_LOWERINGS = ExactLoweringRegistry()
DEFAULT_EXACT_LOWERINGS.register(SphericalCGAnalyzer())
DEFAULT_EXACT_LOWERINGS.register(CartesianSTFAnalyzer())
