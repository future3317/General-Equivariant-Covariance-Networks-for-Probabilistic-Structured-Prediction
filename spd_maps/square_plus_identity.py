"""Square-plus-identity SPD map."""

import torch

from spd_maps.base import SPDMap, symmetrize


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
        cholesky = torch.linalg.cholesky(self.forward(A))
        return 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)

    def precision_action(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        cholesky = torch.linalg.cholesky(self.forward(A))
        x = torch.cholesky_solve(residual.unsqueeze(-1), cholesky)
        return torch.sum(residual * x.squeeze(-1), dim=-1)

    def statistics(
        self, A: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse one scale construction and Cholesky factorization."""
        cholesky = torch.linalg.cholesky(self.forward(A))
        logdet = 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)
        solved = torch.cholesky_solve(residual.unsqueeze(-1), cholesky)
        quadratic = torch.sum(residual * solved.squeeze(-1), dim=-1)
        return logdet, quadratic
