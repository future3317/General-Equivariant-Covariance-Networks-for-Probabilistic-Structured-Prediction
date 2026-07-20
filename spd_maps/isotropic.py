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
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.min_sigma2 = min_sigma2

    def _variance(self, params: torch.Tensor) -> torch.Tensor:
        if params.shape[-1] != 1:
            raise ValueError(f"params last dim {params.shape[-1]} != 1")
        return torch.nn.functional.softplus(params[..., 0]) + self.min_sigma2

    def _squared_norm(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[-1] != self.dim:
            raise ValueError(f"residual last dim {residual.shape[-1]} != {self.dim}")
        return residual.square().sum(-1)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        sigma2 = self._variance(params)
        eye = torch.eye(self.dim, device=params.device, dtype=params.dtype)
        return sigma2[..., None, None] * eye

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        sigma2 = self._variance(params)
        return self.dim * torch.log(sigma2)

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return self._squared_norm(residual) / self._variance(params)

    def statistics(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma2 = self._variance(params)
        return self.dim * torch.log(sigma2), self._squared_norm(residual) / sigma2
