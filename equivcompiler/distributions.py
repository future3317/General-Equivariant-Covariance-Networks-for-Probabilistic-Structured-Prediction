"""Distribution and radial-law plugins for probabilistic compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch

from representations import O3IrrepsSpec
from representations.operator_ir import OperatorFamilyPlan
from representations.representation_ir import DirectSumExpr, IrrepsExpr, RepExpr, SymmetricSquareExpr


class RadialLaw(ABC):
    """Executable radial law independent of representation compilation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable law name."""

    @abstractmethod
    def materialize_log_prob(self) -> torch.nn.Module:
        """Build the proper negative log-likelihood module."""

    @abstractmethod
    def calibration_reference(self) -> dict[str, Any]:
        """Return the residual reference law."""

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return serializable statistical semantics."""


@dataclass(frozen=True)
class GaussianRadial(RadialLaw):
    @property
    def name(self) -> str:
        return "gaussian"

    def materialize_log_prob(self) -> torch.nn.Module:
        from distributions import GaussianNLL

        return GaussianNLL()

    def calibration_reference(self) -> dict[str, Any]:
        return {
            "residual_statistic": "squared_mahalanobis",
            "reference": "chi_square",
        }

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "gaussian_radial"}


@dataclass(frozen=True)
class StudentTRadial(RadialLaw):
    degrees_of_freedom: float = 5.0

    def __post_init__(self) -> None:
        if self.degrees_of_freedom <= 0:
            raise ValueError("Student-t degrees of freedom must be positive")

    @property
    def name(self) -> str:
        return "student_t"

    def materialize_log_prob(self) -> torch.nn.Module:
        from distributions import StudentTNLL

        return StudentTNLL(nu=self.degrees_of_freedom)

    def calibration_reference(self) -> dict[str, Any]:
        return {
            "residual_statistic": "squared_mahalanobis",
            "reference": "scaled_f_distribution",
            "degrees_of_freedom": self.degrees_of_freedom,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "student_t_radial",
            "degrees_of_freedom": self.degrees_of_freedom,
        }


class DistributionSpec(ABC):
    """Map output and operator semantics to an executable probability law."""

    @abstractmethod
    def canonical_reference(self, output: O3IrrepsSpec) -> RepExpr:
        """Return the unrestricted reference expression used for diagnostics."""

    @abstractmethod
    def active_parameter_rep(
        self, output: O3IrrepsSpec, operator: OperatorFamilyPlan
    ) -> RepExpr:
        """Return the selected executable parameter expression."""

    @abstractmethod
    def materialize_log_prob(self) -> torch.nn.Module:
        """Build the proper executable objective without compiler branching."""

    @abstractmethod
    def objective_name(self) -> str:
        """Return a stable objective name for reporting."""

    @abstractmethod
    def objective(self) -> dict[str, Any]:
        """Return proper-scoring-rule semantics."""

    @abstractmethod
    def calibration_reference(self) -> dict[str, Any]:
        """Return the residual calibration reference law."""

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return stable distribution semantics."""


@dataclass(frozen=True, init=False)
class EllipticalDistribution(DistributionSpec):
    """Elliptical family with a pluggable executable radial law."""

    radial: RadialLaw

    def __init__(
        self,
        radial: RadialLaw | Literal["gaussian", "student_t"] = "gaussian",
        student_t_dof: float = 5.0,
    ) -> None:
        if isinstance(radial, str):
            law: RadialLaw = (
                GaussianRadial()
                if radial == "gaussian"
                else StudentTRadial(student_t_dof)
                if radial == "student_t"
                else _unsupported_radial(radial)
            )
        elif isinstance(radial, RadialLaw):
            law = radial
        else:
            raise TypeError("radial must be a RadialLaw or a supported compatibility name")
        object.__setattr__(self, "radial", law)

    @property
    def student_t_dof(self) -> float | None:
        return (
            self.radial.degrees_of_freedom
            if isinstance(self.radial, StudentTRadial)
            else None
        )

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

    def materialize_log_prob(self) -> torch.nn.Module:
        return self.radial.materialize_log_prob()

    def objective_name(self) -> str:
        return self.radial.name

    def calibration_reference(self) -> dict[str, Any]:
        return self.radial.calibration_reference()

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
            "radial": self.radial.as_dict(),
            "objective": self.objective(),
            "proper": True,
            "calibration_reference": self.calibration_reference(),
        }


def _unsupported_radial(name: str) -> RadialLaw:
    raise ValueError(f"unsupported elliptical radial law: {name}")


def normalize_distribution(
    distribution: DistributionSpec | Literal["gaussian", "student_t"],
    *,
    student_t_dof: float,
) -> DistributionSpec:
    if isinstance(distribution, DistributionSpec):
        return distribution
    return EllipticalDistribution(distribution, student_t_dof=student_t_dof)
