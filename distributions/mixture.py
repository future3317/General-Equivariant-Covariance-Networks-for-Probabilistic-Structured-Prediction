"""Exact finite mixtures of normalized multivariate Student-t laws."""

from __future__ import annotations

import torch

from distributions.student_t import student_t_log_prob_from_statistics
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

    def __init__(self, *, require_finite_covariance: bool = False):
        super().__init__()
        self.require_finite_covariance = bool(require_finite_covariance)

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
        logdet, mahalanobis2 = spd_map.statistics(flat_params, flat_residual)
        logdet = logdet.reshape(components, observations)
        mahalanobis2 = mahalanobis2.reshape(components, observations)

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

        component_log_prob = student_t_log_prob_from_statistics(
            logdet,
            mahalanobis2,
            dimension,
            nu_tensor,
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
