"""Gaussian negative log-likelihood in log-scale coordinates."""

from __future__ import annotations

import math
from typing import Dict, Tuple
import torch

from distributions.base import StructuredDistributionLoss
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = target - mu
        d = residual.shape[-1]

        logdet = spd_map.logdet(params)
        quad = spd_map.precision_action(params, residual)

        loss = 0.5 * d * math.log(2.0 * math.pi) + 0.5 * logdet + 0.5 * quad
        loss = loss.mean()

        components = {
            "loss_fit": (0.5 * quad).mean().detach(),
            "loss_uncertainty": (0.5 * logdet).mean().detach(),
            "mahalanobis2_mean": quad.mean().detach(),
            "logdet_mean": logdet.mean().detach(),
        }
        return loss, components
