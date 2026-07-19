"""Abstract base class for distribution losses."""

from __future__ import annotations

import abc
from typing import Dict, Tuple
import torch

from spd_maps.base import SPDMap


class StructuredDistributionLoss(abc.ABC):
    """Negative log-likelihood (or surrogate) for a probabilistic predictor.

    Each concrete loss takes the predicted mean ``mu``, the raw parameters
    ``params`` of the SPD map, the target ``y``, and the ``SPDMap`` instance.
    Working directly with the SPD map lets losses use stable custom methods
    such as ``logdet`` and ``precision_action`` without materializing full
    covariance matrices when unnecessary.
    """

    def __call__(
        self,
        mu: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
        spd_map: SPDMap,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.forward(mu, params, target, spd_map)

    @abc.abstractmethod
    def forward(
        self,
        mu: torch.Tensor,
        params: torch.Tensor,
        target: torch.Tensor,
        spd_map: SPDMap,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss and diagnostic components.

        Args:
            mu: Predicted mean of shape ``(..., d)``.
            params: Raw parameters accepted by ``spd_map``.
            target: Target values of shape ``(..., d)``.
            spd_map: The SPD map used to obtain the scale matrix.

        Returns:
            ``(loss, components)`` where ``components`` is a dict of detached
            diagnostic tensors.
        """
        ...
