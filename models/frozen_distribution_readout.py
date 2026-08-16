"""Task-independent frozen-H,mu distribution-family compositions for E1."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from compatibility.e3nn import o3
from distributions import FiniteMixtureStudentTNLL, GaussianNLL, StudentTNLL
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


class GlobalDegreesOfFreedomReadout(torch.nn.Module):
    """Train one invariant Student-t degrees-of-freedom scalar."""

    def __init__(
        self,
        *,
        minimum: float = 2.05,
        initial: float = 5.0,
    ) -> None:
        super().__init__()
        if not 2.0 < minimum < initial:
            raise ValueError("require 2 < minimum < initial degrees of freedom")
        self.minimum = float(minimum)
        raw_initial = math.log(math.expm1(initial - minimum))
        self.raw_intercept = torch.nn.Parameter(torch.tensor(raw_initial))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.minimum + F.softplus(self.raw_intercept).expand(features.shape[0])

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "global_student_t_degrees_of_freedom",
            "parameterization": "minimum_plus_softplus",
            "minimum": self.minimum,
        }


class InvariantMixtureLogitsReadout(torch.nn.Module):
    """Map typed features to sample-conditional invariant mixture weights."""

    def __init__(self, feature_irreps, *, components: int = 2) -> None:
        super().__init__()
        if components < 2:
            raise ValueError("a finite mixture requires at least two components")
        self.feature_irreps = o3.Irreps(feature_irreps)
        self.components = int(components)
        self.projection = o3.Linear(
            self.feature_irreps, o3.Irreps(f"{components}x0e")
        )
        for parameter in self.projection.parameters():
            torch.nn.init.zeros_(parameter)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.projection(features).reshape(features.shape[0], self.components)
        return torch.softmax(logits, dim=-1).transpose(0, 1)

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "invariant_softmax_mixture_weights",
            "input": str(self.feature_irreps),
            "components": self.components,
            "output": f"{self.components}x0e_logits",
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

class FrozenUncertaintyBranchConditionalStudentT(torch.nn.Module):
    """Train only an equivariant scatter residual and conditional radial law."""

    def __init__(
        self,
        feature_irreps,
        parameter_irreps,
        spd_map: SPDMap,
        *,
        minimum_nu: float = 2.05,
        initial_nu: float = 5.0,
    ) -> None:
        super().__init__()
        self.feature_irreps = o3.Irreps(feature_irreps)
        self.parameter_irreps = o3.Irreps(parameter_irreps)
        self.spd_map = spd_map
        self.objective = StudentTNLL(nu=initial_nu)
        self.residual_projection = o3.Linear(
            self.feature_irreps, self.parameter_irreps
        )
        for parameter in self.residual_projection.parameters():
            torch.nn.init.zeros_(parameter)
        self.nu_readout = InvariantDegreesOfFreedomReadout(
            self.feature_irreps, minimum=minimum_nu, initial=initial_nu
        )

    def forward(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        residual = self.residual_projection(features)
        updated_params = params.detach() + residual
        frozen_mean = mean.detach()
        nu = self.nu_readout(features)
        log_prob, statistics = self.objective.log_prob(
            frozen_mean, updated_params, target, self.spd_map, nu=nu
        )
        return {
            "loss": -log_prob.mean(),
            "log_prob": log_prob,
            "mean": frozen_mean,
            "params": updated_params,
            "nu": nu,
            "residual_params": residual,
            "mahalanobis2": statistics["mahalanobis2"],
            "logdet": statistics["logdet"],
        }

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "uncertainty_only_equivariant_branch_conditional_nu_student_t",
            "mean": "frozen_artifact_detached",
            "base_scatter": "frozen_compiled_operator_artifact",
            "residual": {
                "input": str(self.feature_irreps),
                "output": str(self.parameter_irreps),
                "initialization": "zero",
            },
            "degrees_of_freedom": self.nu_readout.schema(),
            "objective": "exact_student_t_log_likelihood",
        }


class FrozenOperatorProjection(torch.nn.Module):
    """Materialize one frozen typed scatter projection from a checkpoint."""

    def __init__(self, feature_irreps, parameter_irreps) -> None:
        super().__init__()
        self.feature_irreps = o3.Irreps(feature_irreps)
        self.parameter_irreps = o3.Irreps(parameter_irreps)
        self.projection = o3.Linear(self.feature_irreps, self.parameter_irreps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class FrozenGlobalStudentT(torch.nn.Module):
    """Keep mean and scatter fixed while fitting one global finite-covariance nu."""

    def __init__(
        self,
        spd_map: SPDMap,
        *,
        minimum_nu: float = 2.05,
        initial_nu: float = 5.0,
    ) -> None:
        super().__init__()
        self.spd_map = spd_map
        self.objective = StudentTNLL(nu=initial_nu)
        self.nu_readout = GlobalDegreesOfFreedomReadout(
            minimum=minimum_nu, initial=initial_nu
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
            "kind": "single_elliptical_global_nu_student_t",
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


class _FrozenStudentTMixtureBase(torch.nn.Module):
    """Shared K-component frozen-feature Student-t head implementation."""

    def __init__(
        self,
        feature_irreps,
        parameter_irreps,
        output_irreps,
        spd_map: SPDMap,
        *,
        components: int = 2,
        minimum_nu: float = 2.05,
        initial_nu: float = 5.0,
    ) -> None:
        super().__init__()
        if components < 2:
            raise ValueError("a finite mixture requires at least two components")
        self.feature_irreps = o3.Irreps(feature_irreps)
        self.parameter_irreps = o3.Irreps(parameter_irreps)
        self.output_irreps = o3.Irreps(output_irreps)
        self.components = int(components)
        self.spd_map = spd_map
        self.component_projections = torch.nn.ModuleList(
            [
                o3.Linear(self.feature_irreps, self.parameter_irreps)
                for _ in range(components)
            ]
        )
        self.weight_readout = InvariantMixtureLogitsReadout(
            self.feature_irreps, components=components
        )
        self.nu_readouts = torch.nn.ModuleList(
            [
                InvariantDegreesOfFreedomReadout(
                    self.feature_irreps,
                    minimum=minimum_nu,
                    initial=initial_nu,
                )
                for _ in range(components)
            ]
        )
        self.objective = FiniteMixtureStudentTNLL(
            require_finite_covariance=True
        )

    def _component_params(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [projection(features) for projection in self.component_projections]
        )

    def _component_nu(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([readout(features) for readout in self.nu_readouts])

    def _forward_components(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        component_params = self._component_params(features)
        weights = self.weight_readout(features)
        component_nu = self._component_nu(features)
        component_means = self._component_means(features, mean, weights)
        loss, diagnostics = self.objective(
            component_means,
            component_params,
            target,
            self.spd_map,
            weights=weights,
            nu=component_nu,
        )
        result = {
            "loss": loss,
            "component_params": component_params,
            "component_scales": self.spd_map(component_params),
            "component_means": component_means,
            "weights": weights,
            "nu": component_nu,
            **diagnostics,
        }
        return result

    def _component_means(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class FrozenSharedMeanStudentTMixture(_FrozenStudentTMixtureBase):
    """K-component mixture with the current frozen mean shared by all components."""

    def _component_means(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        del features, weights
        return mean.unsqueeze(0).expand(self.components, -1, -1)

    def forward(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del params
        return self._forward_components(features, mean, target)

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "finite_student_t_mixture",
            "components": self.components,
            "component_location": "shared_frozen_mean",
            "component_scatter": "independent_existing_typed_spd_map",
            "weights": self.weight_readout.schema(),
            "degrees_of_freedom": "sample_conditional_invariant",
            "objective": "exact_finite_mixture_logsumexp",
            "moment_matching": False,
        }


class FrozenMultimodalStudentTMixture(_FrozenStudentTMixtureBase):
    """K-component mixture with equivariant, weight-centered mean residuals."""

    def __init__(self, *args, offset_initialization: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset_projections = torch.nn.ModuleList(
            [
                o3.Linear(self.feature_irreps, self.output_irreps)
                for _ in range(self.components)
            ]
        )
        for projection in self.offset_projections:
            for parameter in projection.parameters():
                torch.nn.init.normal_(parameter, mean=0.0, std=offset_initialization)

    def _component_means(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        deltas = torch.stack([projection(features) for projection in self.offset_projections])
        centered = deltas - (
            weights.unsqueeze(-1) * deltas
        ).sum(0, keepdim=True)
        return mean.unsqueeze(0) + centered

    def forward(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del params
        return self._forward_components(features, mean, target)

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "finite_student_t_mixture",
            "components": self.components,
            "component_location": "equivariant_weight_centered_residual",
            "component_scatter": "independent_existing_typed_spd_map",
            "weights": self.weight_readout.schema(),
            "degrees_of_freedom": "sample_conditional_invariant",
            "objective": "exact_finite_mixture_logsumexp",
            "moment_matching": False,
        }

class FrozenMeanScatterElliptical(torch.nn.Module):
    """Fit a typed scatter projection with a selected proper elliptical law."""

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
            raise ValueError(f"unsupported elliptical distribution: {distribution}")

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
    """Compatibility wrapper for the fixed-``nu`` Student-t readout."""

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
