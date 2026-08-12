"""Conventional fixed-coordinate heteroscedastic uncertainty baseline.

This module intentionally does not use the equivariant compiler.  It supplies
the ordinary diagonal readout requested as an external control while reusing
the production Student-t objective and evaluation path.
"""

from __future__ import annotations

import torch

from spd_maps.base import SPDMap


class FixedCoordinateDiagonalReadout(torch.nn.Module):
    """Predict independent log scatter in the dataset coordinate frame."""

    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        self.projection = torch.nn.Linear(feature_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class FixedCoordinateDiagonalMap(SPDMap):
    """Map coordinate-wise log scatter to a positive diagonal matrix."""

    @staticmethod
    def _variance(params: torch.Tensor) -> torch.Tensor:
        return torch.exp(params)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(self._variance(params))

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        return params.sum(-1)

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return (residual.square() / self._variance(params)).sum(-1)
