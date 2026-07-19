"""Robust non-probabilistic surrogate loss."""

from __future__ import annotations

from typing import Dict, Tuple
import torch

from distributions.base import StructuredDistributionLoss
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = target - mu

        logdet = spd_map.logdet(params)
        quad = spd_map.precision_action(params, residual)

        # Huber loss on the squared Mahalanobis distance.
        threshold_tensor = torch.tensor(
            self.huber_threshold, device=quad.device, dtype=quad.dtype
        )
        quadratic_part = torch.minimum(quad, threshold_tensor)
        linear_part = torch.clamp(quad - threshold_tensor, min=0.0)
        huber_quad = quadratic_part + threshold_tensor * linear_part.sqrt()

        loss = self.log_det_weight * logdet + 0.5 * huber_quad
        loss = loss.mean()

        components = {
            "loss_fit": (0.5 * huber_quad).mean().detach(),
            "loss_uncertainty": (self.log_det_weight * logdet).mean().detach(),
            "mahalanobis2_mean": quad.mean().detach(),
            "logdet_mean": logdet.mean().detach(),
        }
        return loss, components
