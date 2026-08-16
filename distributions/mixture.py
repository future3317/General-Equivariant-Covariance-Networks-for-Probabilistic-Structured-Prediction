"""Exact finite mixtures of normalized multivariate Student-t laws."""

from __future__ import annotations

import torch

from distributions.student_t import (
    _quadratic_from_log_for_diagnostics,
    student_t_log_prob_from_log_statistics,
    student_t_log_prob_from_statistics,
)
from spd_maps.base import SPDMap


def _component_parameter(
    value: float | torch.Tensor,
    *,
    components: int,
    observations: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Broadcast a scalar, component, observation, or component-observation value."""
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        return tensor.expand(components, observations)
    if tensor.shape == (components,):
        return tensor[:, None].expand(components, observations)
    if tensor.shape == (observations,):
        return tensor[None, :].expand(components, observations)
    if tensor.shape == (components, observations):
        return tensor
    raise ValueError(
        f"{name} must be scalar, ({components},), ({observations},), or "
        f"({components}, {observations}); got {tuple(tensor.shape)}"
    )


class FiniteMixtureStudentTNLL(torch.nn.Module):
    """Normalized K-component Student-t NLL with exact logsumexp semantics.

    Each component receives its own equivariant mean and raw SPD parameters,
    but all components use the supplied, already-compiled ``SPDMap``.  The
    class is deliberately outside the compiler planner: it composes existing
    certified component lowerings without changing their typed semantics.
    """

    def __init__(
        self,
        *,
        require_finite_covariance: bool = False,
        quadratic_oracle: str = "direct",
    ):
        super().__init__()
        self.require_finite_covariance = bool(require_finite_covariance)
        if quadratic_oracle not in {"direct", "shifted_log"}:
            raise ValueError(
                "quadratic_oracle must be 'direct' or 'shifted_log'"
            )
        self.quadratic_oracle = quadratic_oracle

    def predictive_law_contract(self, *, components: int = 2) -> dict[str, object]:
        """Declare exact mixture scoring and law-correct diagnostic semantics."""
        if components < 2:
            raise ValueError("a finite mixture contract requires at least two components")
        return {
            "kind": "finite_mixture_student_t",
            "components": int(components),
            "log_prob": "exact_component_logsumexp",
            "sample": "categorical_component_student_t",
            "moment_existence": {
                "mean": "finite_for_component_nu>1",
                "covariance": "finite_for_component_nu>2",
            },
            "scatter_to_covariance": "componentwise_nu/(nu-2) * S",
            "marginal_quantile": "simulation_based_mixture_quantile",
            "radial_reference": "simulation_based_mixture_reference",
            "diagnostic_oracle": {
                "kind": "simulation_based_mixture_oracle",
                "moment_matched": False,
                "radius_direction_null": "not_single_ellipse_null",
            },
        }

    def forward(
        self,
        component_means: torch.Tensor,
        component_params: torch.Tensor,
        target: torch.Tensor,
        spd_map: SPDMap,
        *,
        weights: torch.Tensor | None = None,
        nu: float | torch.Tensor = 5.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if component_means.ndim != 3:
            raise ValueError("component_means must have shape (K,N,d)")
        if target.ndim != 2:
            raise ValueError("target must have shape (N,d)")
        components, observations, dimension = component_means.shape
        if components < 2:
            raise ValueError("a finite mixture requires at least two components")
        if target.shape != (observations, dimension):
            raise ValueError("component means and target shapes are incompatible")
        if component_params.ndim < 3 or component_params.shape[:2] != (
            components,
            observations,
        ):
            raise ValueError("component_params must have leading shape (K,N)")

        residual = target.unsqueeze(0) - component_means
        flat_params = component_params.reshape(components * observations, *component_params.shape[2:])
        flat_residual = residual.reshape(components * observations, dimension)
        logdet = spd_map.logdet(flat_params)
        if self.quadratic_oracle == "shifted_log":
            log_mahalanobis2 = spd_map.log_precision_action(
                flat_params, flat_residual
            )
            mahalanobis2 = _quadratic_from_log_for_diagnostics(log_mahalanobis2)
        else:
            mahalanobis2 = spd_map.precision_action(flat_params, flat_residual)
            log_mahalanobis2 = None
        logdet = logdet.reshape(components, observations)
        mahalanobis2 = mahalanobis2.reshape(components, observations)
        if log_mahalanobis2 is not None:
            log_mahalanobis2 = log_mahalanobis2.reshape(components, observations)

        nu_tensor = _component_parameter(
            nu,
            components=components,
            observations=observations,
            dtype=mahalanobis2.dtype,
            device=mahalanobis2.device,
            name="nu",
        )
        if bool((nu_tensor <= 0).any()):
            raise ValueError("Student-t degrees of freedom must be positive")
        if self.require_finite_covariance and bool((nu_tensor <= 2).any()):
            raise ValueError("finite-covariance mixture requires nu > 2")

        if weights is None:
            normalized_weights = mahalanobis2.new_full(
                (components, observations), 1.0 / components
            )
        else:
            normalized_weights = _component_parameter(
                weights,
                components=components,
                observations=observations,
                dtype=mahalanobis2.dtype,
                device=mahalanobis2.device,
                name="weights",
            )
            if not bool(torch.isfinite(normalized_weights).all()) or bool(
                (normalized_weights <= 0).any()
            ):
                raise ValueError("mixture weights must be finite and positive")
            normalized_weights = normalized_weights / normalized_weights.sum(
                dim=0, keepdim=True
            )

        if log_mahalanobis2 is None:
            component_log_prob = student_t_log_prob_from_statistics(
                logdet, mahalanobis2, dimension, nu_tensor
            )
        else:
            component_log_prob = student_t_log_prob_from_log_statistics(
                logdet, log_mahalanobis2, dimension, nu_tensor
            )
        log_weights = normalized_weights.log()
        log_prob = torch.logsumexp(log_weights + component_log_prob, dim=0)
        loss = -log_prob.mean()
        return loss, {
            "loss": loss,
            "log_prob": log_prob,
            "log_weights": log_weights,
            "weights": normalized_weights,
            "component_log_prob": component_log_prob,
            "responsibilities": torch.softmax(
                log_weights + component_log_prob, dim=0
            ),
            "logdet": logdet,
            "mahalanobis2": mahalanobis2,
            "nu": nu_tensor,
        }
