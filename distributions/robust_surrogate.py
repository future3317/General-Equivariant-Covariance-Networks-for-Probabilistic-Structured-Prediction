"""Robust non-probabilistic surrogate loss."""

from __future__ import annotations

import torch

from distributions.base import StructuredDistributionLoss, diagnostic_components
from spd_maps.base import SPDMap


class RobustSurrogateLoss(StructuredDistributionLoss):
    """A robust surrogate that mimics LE-ESO without claiming to be a likelihood.

    This loss is intentionally **not** a proper negative log-likelihood. It
    combines a log-determinant uncertainty penalty with a Huber-smoothed
    Mahalanobis distance and is useful as a baseline or when the data have
    heavy tails that violate Gaussian/Student-t assumptions.

    Use this only when the goal is robust point-prediction-quality with
    reasonable uncertainty, not probabilistic calibration.
    """

    def __init__(
        self,
        huber_threshold: float = 20.0,
        log_det_weight: float = 1.0,
    ):
        super().__init__()
        self.huber_threshold = huber_threshold
        self.log_det_weight = log_det_weight

    def forward(
        self,
        mu: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
        spd_map: SPDMap,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual = target - mu

        logdet, quad = spd_map.statistics(params, residual)

        # Huber loss on the squared Mahalanobis distance.
        threshold_tensor = quad.new_tensor(self.huber_threshold)
        quadratic_part = torch.minimum(quad, threshold_tensor)
        linear_part = torch.clamp(quad - threshold_tensor, min=0.0)
        huber_quad = quadratic_part + threshold_tensor * linear_part.sqrt()

        fit = 0.5 * huber_quad
        uncertainty = self.log_det_weight * logdet
        loss = uncertainty + fit
        loss = loss.mean()
        components = diagnostic_components(fit, uncertainty, quad, logdet)
        return loss, components
