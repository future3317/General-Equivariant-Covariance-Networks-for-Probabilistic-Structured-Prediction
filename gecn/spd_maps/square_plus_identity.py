"""Square-plus-identity SPD map."""

import torch

from gecn.spd_maps.base import SPDMap, symmetrize


class SquarePlusIdentityMap(SPDMap):
    """SPD map via :math:`S = A^2 + \\varepsilon I`.

    This map is fast (no eigendecomposition), strictly SPD, and conjugation
    equivariant. It is not injective and its range is :math:`S \\succeq \\varepsilon I`.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        d = A.shape[-1]
        eye = torch.eye(d, device=A.device, dtype=A.dtype)
        return A @ A + self.eps * eye

    def logdet(self, A: torch.Tensor) -> torch.Tensor:
        S = self.forward(A)
        return torch.logdet(S)

    def precision_action(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        S = self.forward(A)
        # Solve S x = r for each sample.
        x = torch.linalg.solve(S, residual.unsqueeze(-1))
        return torch.sum(residual * x.squeeze(-1), dim=-1)
