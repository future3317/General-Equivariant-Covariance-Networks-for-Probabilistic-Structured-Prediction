"""Gaussian negative log-likelihood in log-scale coordinates."""

from __future__ import annotations

import math
import torch

from distributions.base import StructuredDistributionLoss, diagnostic_components
from spd_maps.base import SPDMap


class GaussianNLL(StructuredDistributionLoss):
    """Multivariate Gaussian NLL with log-covariance parameterization.

    If the SPD map is ``S = exp(A)``, the negative log-likelihood is

    .. math::

        \\mathcal L_G = \\frac d2 \\log(2\\pi)
        + \\frac12 \\operatorname{tr}(A)
        + \\frac12 (y-\\mu)^\\top \\exp(-A) (y-\\mu).

    The loss uses ``spd_map.logdet`` and ``spd_map.precision_action`` so that
    it is numerically stable and does not require an explicit eigendecomposition
    in the training loop.
    """

    def forward(
        self,
        mu: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
        spd_map: SPDMap,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual = target - mu
        d = residual.shape[-1]

        logdet, quad = spd_map.statistics(params, residual)

        fit = 0.5 * quad
        uncertainty = 0.5 * logdet
        loss = 0.5 * d * math.log(2.0 * math.pi) + uncertainty + fit
        loss = loss.mean()
        components = diagnostic_components(fit, uncertainty, quad, logdet)
        return loss, components
