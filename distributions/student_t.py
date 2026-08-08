"""Multivariate Student-t negative log-likelihood."""

from __future__ import annotations

import math
from functools import cache

import torch

from distributions.base import StructuredDistributionLoss, diagnostic_components
from spd_maps.base import SPDMap


def student_t_log_prob_from_statistics(
    logdet: torch.Tensor,
    mahalanobis2: torch.Tensor,
    dimension: int,
    nu: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate Student-t log density from existing SPD sufficient statistics.

    ``nu`` may be a scalar or a tensor broadcastable to the sample shape.  This
    is the single tensor-valued normalization path used by conditional-nu and
    finite-mixture compositions; fixed-nu :class:`StudentTNLL` retains its
    cached scalar normalization in ``forward``.
    """
    if dimension < 1:
        raise ValueError("dimension must be positive")
    nu_tensor = torch.as_tensor(
        nu, dtype=mahalanobis2.dtype, device=mahalanobis2.device
    )
    if bool((nu_tensor <= 0).any()):
        raise ValueError("Student-t degrees of freedom nu must be positive")
    try:
        _, nu_tensor = torch.broadcast_tensors(mahalanobis2, nu_tensor)
    except RuntimeError as error:
        raise ValueError("nu is not broadcastable to the sample shape") from error
    normalization = (
        torch.lgamma((nu_tensor + dimension) / 2.0)
        - torch.lgamma(nu_tensor / 2.0)
        - 0.5 * dimension * torch.log(nu_tensor * math.pi)
    )
    return (
        normalization
        - 0.5 * logdet
        - 0.5 * (nu_tensor + dimension) * torch.log1p(mahalanobis2 / nu_tensor)
    )


class StudentTNLL(StructuredDistributionLoss):
    """Multivariate Student-t NLL with scale-matrix parameterization.

    The model outputs a scale (scatter) matrix :math:`S` through ``spd_map``.
    For a Student-t with :math:`\\nu` degrees of freedom,

    .. math::

        p(y \\mid x) \\propto |S|^{-1/2}
        \\left(1 + \\frac1\\nu (y-\\mu)^\\top S^{-1} (y-\\mu)\\right)^{-(\\nu+d)/2}.

    The negative log-likelihood is

    .. math::

        \\mathcal L_t = -\\log\\Gamma\\!\\left(\\frac{\\nu+d}{2}\\right)
        + \\log\\Gamma\\!\\left(\\frac{\\nu}{2}\\right)
        + \\frac d2 \\log(\\nu\\pi)
        + \\frac12 \\log\\det S
        + \\frac{\\nu+d}{2} \\log\\!\\left(1 + \\frac{q}{\\nu}\\right),

    where :math:`q = (y-\\mu)^\\top S^{-1}(y-\\mu)`. When :math:`\\nu > 2`, the
    statistical covariance is :math:`\\frac{\\nu}{\\nu-2} S`.
    """

    def __init__(self, nu: float = 5.0):
        super().__init__()
        if nu <= 0:
            raise ValueError("Student-t degrees of freedom nu must be positive.")
        self.nu = nu

    @staticmethod
    @cache
    def _normalization_constant(nu: float, dimension: int) -> float:
        """Cache the fixed scalar term without launching device kernels."""
        return (
            -math.lgamma((nu + dimension) / 2.0)
            + math.lgamma(nu / 2.0)
            + 0.5 * dimension * math.log(nu * math.pi)
        )

    def forward(
        self,
        mu: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
        spd_map: SPDMap,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual = target - mu
        d = residual.shape[-1]
        nu = self.nu

        logdet, quad = spd_map.statistics(params, residual)
        const = self._normalization_constant(nu, d)

        fit = 0.5 * (nu + d) * torch.log1p(quad / nu)
        uncertainty = 0.5 * logdet
        loss = const + uncertainty + fit
        loss = loss.mean()
        components = diagnostic_components(fit, uncertainty, quad, logdet)
        components["nu"] = mu.new_tensor(nu)
        return loss, components

    def log_prob(
        self,
        mu: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
        spd_map: SPDMap,
        *,
        nu: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return unreduced log probabilities, optionally with conditional nu."""
        residual = target - mu
        logdet, quad = spd_map.statistics(params, residual)
        effective_nu: float | torch.Tensor = self.nu if nu is None else nu
        log_prob = student_t_log_prob_from_statistics(
            logdet, quad, residual.shape[-1], effective_nu
        )
        return log_prob, {
            "mahalanobis2": quad,
            "logdet": logdet,
            "nu": torch.as_tensor(effective_nu, dtype=mu.dtype, device=mu.device),
        }

    @staticmethod
    def scale_to_covariance(scale: torch.Tensor, nu: float) -> torch.Tensor:
        """Convert scale matrix to covariance when :math:`\\nu > 2`."""
        if nu <= 2:
            raise ValueError("Covariance is only finite for nu > 2.")
        return (nu / (nu - 2.0)) * scale
