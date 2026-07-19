"""Multivariate Student-t negative log-likelihood."""

from __future__ import annotations

import math
from typing import Dict, Tuple
import torch

from gecn.distributions.base import StructuredDistributionLoss
from gecn.spd_maps.base import SPDMap


class StudentTNLL(StructuredDistributionLoss):
    """Multivariate Student-t NLL with scale-matrix parameterization.

    The model outputs a scale (scatter) matrix :math:`S` through ``spd_map``.
    For a Student-t with :math:`\\nu` degrees of freedom,

    .. math::

        p(y \\mid x) \\propto |S|^{-1/2}
        \\left(1 + \\frac1\\nu (y-\\mu)^\\top S^{-1} (y-\\mu)\\right)^{-(\\nu+d)/2}.

    The negative log-likelihood is

    .. math::

        \\mathcal L_t = \\log\\Gamma\\!\\left(\\frac{\\nu+d}{2}\\right)
        - \\log\\Gamma\\!\\left(\\frac{\\nu}{2}\\right)
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

    def forward(
        self,
        mu: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
        spd_map: SPDMap,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = target - mu
        d = residual.shape[-1]
        nu = self.nu

        logdet = spd_map.logdet(params)
        quad = spd_map.precision_action(params, residual)

        const = (
            -torch.lgamma(torch.tensor((nu + d) / 2.0, device=mu.device, dtype=mu.dtype))
            + torch.lgamma(torch.tensor(nu / 2.0, device=mu.device, dtype=mu.dtype))
            + 0.5 * d * math.log(nu * math.pi)
        )

        loss = (
            const
            + 0.5 * logdet
            + 0.5 * (nu + d) * torch.log1p(quad / nu)
        )
        loss = loss.mean()

        components = {
            "loss_fit": (0.5 * (nu + d) * torch.log1p(quad / nu)).mean().detach(),
            "loss_uncertainty": (0.5 * logdet).mean().detach(),
            "mahalanobis2_mean": quad.mean().detach(),
            "logdet_mean": logdet.mean().detach(),
            "nu": torch.tensor(nu, device=mu.device, dtype=mu.dtype),
        }
        return loss, components

    @staticmethod
    def scale_to_covariance(scale: torch.Tensor, nu: float) -> torch.Tensor:
        """Convert scale matrix to covariance when :math:`\\nu > 2`."""
        if nu <= 2:
            raise ValueError("Covariance is only finite for nu > 2.")
        return (nu / (nu - 2.0)) * scale
