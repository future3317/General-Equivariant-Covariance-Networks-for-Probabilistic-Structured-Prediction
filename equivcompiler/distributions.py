"""Distribution parameter functors for probabilistic representation compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from representations import O3IrrepsSpec
from representations.operator_ir import OperatorFamilyPlan
from representations.representation_ir import DirectSumExpr, IrrepsExpr, RepExpr, SymmetricSquareExpr


class DistributionSpec(ABC):
    """Map an output type and operator family to sufficient-parameter semantics."""

    @abstractmethod
    def canonical_reference(self, output: O3IrrepsSpec) -> RepExpr:
        """Return the full-family reference expression used for diagnostics."""

    @abstractmethod
    def active_parameter_rep(
        self, output: O3IrrepsSpec, operator: OperatorFamilyPlan
    ) -> RepExpr:
        """Return the selected executable parameter expression."""

    @abstractmethod
    def objective_name(self) -> Literal["gaussian", "student_t"]:
        """Return the proper objective implemented by the runtime."""

    @abstractmethod
    def objective(self) -> dict[str, Any]:
        """Return proper-scoring-rule semantics."""

    @abstractmethod
    def calibration_reference(self) -> dict[str, Any]:
        """Return the residual calibration reference law."""

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return stable distribution semantics."""


@dataclass(frozen=True)
class EllipticalDistribution(DistributionSpec):
    """Elliptical family with equivariant location and structured scale/precision."""

    radial: Literal["gaussian", "student_t"] = "gaussian"
    student_t_dof: float = 5.0

    def __post_init__(self) -> None:
        if self.radial not in {"gaussian", "student_t"}:
            raise ValueError(f"unsupported elliptical radial law: {self.radial}")
        if self.student_t_dof <= 0:
            raise ValueError("student_t_dof must be positive")

    @staticmethod
    def _location(output: O3IrrepsSpec) -> IrrepsExpr:
        return IrrepsExpr(output.irreps, "location")

    def canonical_reference(self, output: O3IrrepsSpec) -> RepExpr:
        location = self._location(output)
        return DirectSumExpr((location, SymmetricSquareExpr(location)))

    def active_parameter_rep(
        self, output: O3IrrepsSpec, operator: OperatorFamilyPlan
    ) -> RepExpr:
        return operator.active_expression(self._location(output))

    def objective_name(self) -> Literal["gaussian", "student_t"]:
        return self.radial

    def calibration_reference(self) -> dict[str, Any]:
        return {
            "residual_statistic": "squared_mahalanobis",
            "reference": (
                "chi_square"
                if self.radial == "gaussian"
                else "scaled_f_distribution"
            ),
        }

    def objective(self) -> dict[str, Any]:
        return {
            "name": self.objective_name(),
            "kind": "negative_log_likelihood",
            "proper": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "elliptical_distribution",
            "location": "equivariant_vector_in_output_representation",
            "radial": self.radial,
            "student_t_dof": self.student_t_dof if self.radial == "student_t" else None,
            "objective": self.objective(),
            "proper": True,
            "calibration_reference": self.calibration_reference(),
        }


def normalize_distribution(
    distribution: DistributionSpec | Literal["gaussian", "student_t"],
    *,
    student_t_dof: float,
) -> DistributionSpec:
    if isinstance(distribution, DistributionSpec):
        return distribution
    return EllipticalDistribution(distribution, student_t_dof=student_t_dof)
