"""Isotropic SPD map: :math:`S = \\sigma^2 I`."""

from __future__ import annotations

import torch

from spd_maps.base import SPDMap


class IsotropicMap(SPDMap):
    """Predict a single scalar variance shared across all output dimensions.

    Parameters have shape ``(..., 1)`` where the last entry is ``\\log(\\sigma^2)``.
    This map is equivariant for any orthogonal representation because the
    identity commutes with every :math:`\\rho(g)`.
    """

    def __init__(self, dim: int, min_sigma2: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.min_sigma2 = min_sigma2

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        *batch, _ = params.shape
        sigma2 = torch.nn.functional.softplus(params[..., -1]) + self.min_sigma2
        eye = torch.eye(self.dim, device=params.device, dtype=params.dtype)
        return sigma2[..., None, None] * eye

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        S = self.forward(params)
        return torch.logdet(S)

    def precision_action(self, params: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        S = self.forward(params)
        x = torch.linalg.solve(S, residual.unsqueeze(-1))
        return torch.sum(residual * x.squeeze(-1), dim=-1)
