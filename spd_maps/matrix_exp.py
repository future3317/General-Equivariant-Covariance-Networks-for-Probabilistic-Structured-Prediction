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

    def log_precision_action(
        self, A: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate ``log(r^T exp(-A) r)`` with an exact shifted oracle.

        For any scalar ``c``, ``exp(-A) = exp(c) exp(-A-cI)``.  Choosing
        ``c=-lambda_min(A)`` makes the largest eigenvalue of the matrix
        exponential argument zero, avoiding the overflow-prone direct
        ``exp(-A)`` while preserving the same real-arithmetic quadratic form.
        The shift is detached because it is only a numerical rescaling
        parameter; the value identity cancels its derivative analytically.
        """
        A = symmetrize(A)
        eigenvalues = torch.linalg.eigvalsh(A.detach())
        shift = -eigenvalues[..., 0]
        dimension = A.shape[-1]
        identity = torch.eye(dimension, dtype=A.dtype, device=A.device)
        scaled_precision = torch.linalg.matrix_exp(
            -A - shift[..., None, None] * identity
        )
        scaled_quadratic = torch.einsum(
            "...i,...ij,...j->...", residual, scaled_precision, residual
        )
        return shift + torch.log(scaled_quadratic)
