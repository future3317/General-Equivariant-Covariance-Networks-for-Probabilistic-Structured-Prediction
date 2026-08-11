"""Task-independent frozen-H,mu distribution-family compositions for E1."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from compatibility.e3nn import o3
from distributions import GaussianNLL, StudentTNLL
from evaluation.ensemble import finite_mixture_nll
from spd_maps.base import SPDMap


class InvariantDegreesOfFreedomReadout(torch.nn.Module):
    """Predict finite-covariance Student-t degrees of freedom from 0e channels."""

    def __init__(
        self,
        feature_irreps,
        *,
        minimum: float = 2.05,
        initial: float = 5.0,
    ) -> None:
        super().__init__()
        if not 2.0 < minimum < initial:
            raise ValueError("require 2 < minimum < initial degrees of freedom")
        self.feature_irreps = o3.Irreps(feature_irreps)
        self.minimum = float(minimum)
        self.projection = o3.Linear(self.feature_irreps, o3.Irreps("0e"))
        for parameter in self.projection.parameters():
            torch.nn.init.zeros_(parameter)
        raw_initial = math.log(math.expm1(initial - minimum))
        self.raw_intercept = torch.nn.Parameter(torch.tensor(raw_initial))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.projection(features).squeeze(-1) + self.raw_intercept
        return self.minimum + F.softplus(raw)

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "conditional_student_t_degrees_of_freedom",
            "input": str(self.feature_irreps),
            "output": "0e",
            "parameterization": "minimum_plus_softplus",
            "minimum": self.minimum,
        }


class EquivariantOffsetReadout(torch.nn.Module):
    """Project legal typed features to an offset in the output representation."""

    def __init__(self, feature_irreps, output_irreps, *, initialization: float = 1e-3):
        super().__init__()
        if initialization <= 0:
            raise ValueError("offset initialization scale must be positive")
        self.feature_irreps = o3.Irreps(feature_irreps)
        self.output_irreps = o3.Irreps(output_irreps)
        self.projection = o3.Linear(self.feature_irreps, self.output_irreps)
        with torch.no_grad():
            for parameter in self.projection.parameters():
                parameter.normal_(mean=0.0, std=initialization)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "equivariant_symmetric_component_offset",
            "input": str(self.feature_irreps),
            "output": str(self.output_irreps),
            "component_means": "mu_plus_minus_delta",
        }


class FrozenConditionalStudentT(torch.nn.Module):
    """Keep mean and scatter fixed while fitting only invariant conditional nu."""

    def __init__(
        self,
        feature_irreps,
        spd_map: SPDMap,
        *,
        minimum_nu: float = 2.05,
        initial_nu: float = 5.0,
    ) -> None:
        super().__init__()
        self.spd_map = spd_map
        self.objective = StudentTNLL(nu=initial_nu)
        self.nu_readout = InvariantDegreesOfFreedomReadout(
            feature_irreps, minimum=minimum_nu, initial=initial_nu
        )

    def forward(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        nu = self.nu_readout(features)
        log_prob, statistics = self.objective.log_prob(
            mean, params, target, self.spd_map, nu=nu
        )
        return {
            "loss": -log_prob.mean(),
            "log_prob": log_prob,
            "params": params,
            "nu": nu,
            "mahalanobis2": statistics["mahalanobis2"],
            "logdet": statistics["logdet"],
        }

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "single_elliptical_conditional_nu_student_t",
            "mean": "frozen_artifact",
            "scatter": "frozen_compiled_operator_artifact",
            "degrees_of_freedom": self.nu_readout.schema(),
            "objective": "exact_student_t_log_likelihood",
        }


class FrozenSymmetricStudentTMixture(torch.nn.Module):
    """K=2 fixed-weight Student-t mixture with frozen shared scatter."""

    def __init__(
        self,
        feature_irreps,
        output_irreps,
        spd_map: SPDMap,
        *,
        student_t_dof: float = 5.0,
        offset_initialization: float = 1e-3,
    ) -> None:
        super().__init__()
        if student_t_dof <= 2.0:
            raise ValueError(
                "E1 mixture requires finite-covariance Student-t components"
            )
        self.spd_map = spd_map
        self.student_t_dof = float(student_t_dof)
        self.offset_readout = EquivariantOffsetReadout(
            feature_irreps,
            output_irreps,
            initialization=offset_initialization,
        )

    def forward(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta = self.offset_readout(features)
        component_means = torch.stack((mean + delta, mean - delta))
        shared_scale = self.spd_map(params)
        component_scales = shared_scale.unsqueeze(0).expand(2, *shared_scale.shape)
        loss = finite_mixture_nll(
            component_means,
            component_scales,
            target,
            distribution="student_t",
            student_t_dof=self.student_t_dof,
        )
        return {
            "loss": loss,
            "delta": delta,
            "component_means": component_means,
            "component_scales": component_scales,
            "weights": mean.new_full((2, mean.shape[0]), 0.5),
        }

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "finite_student_t_mixture",
            "components": 2,
            "base_mean": "frozen_artifact",
            "component_location": self.offset_readout.schema(),
            "scatter": "shared_frozen_compiled_operator_artifact",
            "weights": {"kind": "fixed_invariant", "values": [0.5, 0.5]},
            "degrees_of_freedom": {
                "kind": "fixed",
                "value": self.student_t_dof,
            },
            "objective": "exact_finite_mixture_logsumexp",
            "moment_matching": False,
        }


class FrozenMeanScatterElliptical(torch.nn.Module):
    """Retrain one typed operator projection while ``H`` and ``mu`` stay frozen."""

    def __init__(
        self,
        feature_irreps,
        parameter_irreps,
        spd_map: SPDMap,
        *,
        distribution: str,
        student_t_dof: float = 5.0,
    ) -> None:
        super().__init__()
        self.feature_irreps = o3.Irreps(feature_irreps)
        self.parameter_irreps = o3.Irreps(parameter_irreps)
        self.parameter_projection = o3.Linear(
            self.feature_irreps, self.parameter_irreps
        )
        self.spd_map = spd_map
        self.distribution = distribution
        if distribution == "gaussian":
            self.objective = GaussianNLL()
        elif distribution == "student_t":
            if student_t_dof <= 2.0:
                raise ValueError("factorial Student-t requires nu > 2")
            self.objective = StudentTNLL(nu=student_t_dof)
        else:
            raise ValueError(
                f"unsupported elliptical distribution: {distribution}"
            )

    def forward(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        params = self.parameter_projection(features)
        loss, components = self.objective(mean, params, target, self.spd_map)
        return {"loss": loss, "params": params, **components}

    def schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "kind": (
                "single_elliptical_fixed_nu_student_t"
                if self.distribution == "student_t"
                else "single_elliptical_gaussian"
            ),
            "mean": "frozen_artifact",
            "scatter": {
                "kind": "trainable_typed_operator_projection",
                "input": str(self.feature_irreps),
                "output": str(self.parameter_irreps),
            },
            "objective": (
                "exact_student_t_log_likelihood"
                if self.distribution == "student_t"
                else "exact_gaussian_log_likelihood"
            ),
        }
        if self.distribution == "student_t":
            schema["degrees_of_freedom"] = {
                "kind": "fixed",
                "value": float(self.objective.nu),
            }
        return schema


class FrozenMeanScatterStudentT(FrozenMeanScatterElliptical):
    """Compatibility wrapper for the E1 fixed-``nu`` Student-t readout."""

    def __init__(
        self,
        feature_irreps,
        parameter_irreps,
        spd_map: SPDMap,
        *,
        student_t_dof: float = 5.0,
    ) -> None:
        super().__init__(
            feature_irreps,
            parameter_irreps,
            spd_map,
            distribution="student_t",
            student_t_dof=student_t_dof,
        )
