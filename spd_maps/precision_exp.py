"""Precision exponential SPD map."""

import torch

from spd_maps.base import SPDMap, symmetrize


class PrecisionExponentialMap(SPDMap):
    """SPD map via log-precision: :math:`S = \\exp(-B)`.

    This is equivalent in expressive power to the covariance exponential map.
    The only difference is the optimization coordinate: ``B`` is the
    log-precision rather than the log-covariance.
    """

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        B = symmetrize(B)
        return torch.linalg.matrix_exp(-B)

    def logdet(self, B: torch.Tensor) -> torch.Tensor:
        B = symmetrize(B)
        # det(S) = det(exp(-B)) = exp(-tr(B))
        return -torch.diagonal(B, dim1=-2, dim2=-1).sum(dim=-1)

    def precision_action(self, B: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        B = symmetrize(B)
        # S^{-1} = exp(B)
        P = torch.linalg.matrix_exp(B)
        return torch.einsum("...i,...ij,...j->...", residual, P, residual)
