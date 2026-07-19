"""Matrix exponential SPD map."""

import torch

from spd_maps.base import SPDMap, symmetrize


class MatrixExponentialMap(SPDMap):
    """SPD map via matrix exponential: :math:`S = \\exp(A)`.

    This is the default full-rank parameterization. It is a bijection between
    symmetric matrices and SPD matrices, with inverse :math:`A = \\log(S)`.
    """

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        # torch.linalg.matrix_exp is batch-friendly and uses stable
        # scaling-and-squaring; it handles degenerate eigenvalues correctly.
        return torch.linalg.matrix_exp(A)

    def logdet(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        # det(exp(A)) = exp(tr(A)), so logdet = tr(A).
        return torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)

    def precision_action(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        # S^{-1} = exp(-A)
        P = torch.linalg.matrix_exp(-A)
        return torch.einsum("...i,...ij,...j->...", residual, P, residual)
