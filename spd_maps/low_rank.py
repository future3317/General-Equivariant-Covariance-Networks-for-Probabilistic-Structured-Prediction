"""Low-rank-plus-isotropic SPD map."""

import torch

from spd_maps.base import SPDMap


class LowRankPlusIsotropicMap(SPDMap):
    """Structured low-rank SPD map: :math:`S = \\sigma^2 I + L L^T`.

    The factor matrix ``L`` is a ``(d, rank)`` matrix whose columns transform
    as vectors in the output representation ``V``. This makes the model
    equivariant and parameter-efficient for high-dimensional outputs such as
    the 21-dimensional elasticity tensor.
    """

    def __init__(self, dim: int, rank: int, min_sigma2: float = 1e-4):
        super().__init__()
        if dim < 1 or rank < 1:
            raise ValueError("dim and rank must be positive")
        if min_sigma2 < 0:
            raise ValueError("min_sigma2 must be nonnegative")
        self.dim = dim
        self.rank = rank
        self.min_sigma2 = min_sigma2

    def _unpack(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.dim * self.rank + 1
        if params.shape[-1] != expected:
            raise ValueError(f"params last dim {params.shape[-1]} != {expected}")
        L = params[..., :-1].reshape(*params.shape[:-1], self.dim, self.rank)
        sigma2 = torch.nn.functional.softplus(params[..., -1]) + self.min_sigma2
        return L, sigma2

    def _factor_system(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``L``, ``sigma2`` and the Cholesky factor of Woodbury ``M``."""
        L, sigma2 = self._unpack(params)
        gram = torch.matmul(L.transpose(-1, -2), L)
        identity = torch.eye(self.rank, device=params.device, dtype=params.dtype)
        system = identity + gram / sigma2[..., None, None]
        return L, sigma2, torch.linalg.cholesky(system)

    def _statistics_from_factor(
        self,
        L: torch.Tensor,
        sigma2: torch.Tensor,
        cholesky: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual.shape[-1] != self.dim:
            raise ValueError(f"residual last dim {residual.shape[-1]} != {self.dim}")
        logdet = self.dim * torch.log(sigma2) + 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(-1)
        projected = torch.matmul(L.transpose(-1, -2), residual.unsqueeze(-1))
        solved = torch.cholesky_solve(projected, cholesky)
        precision_residual = (
            residual / sigma2[..., None]
            - torch.matmul(L, solved).squeeze(-1) / sigma2[..., None].square()
        )
        quadratic = torch.sum(residual * precision_residual, dim=-1)
        return logdet, quadratic

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Assemble SPD matrix from unconstrained parameters.

        Args:
            params: Tensor of shape ``(..., dim * rank + 1)`` where the last
                entry is the log-variance parameter and the leading entries
                are flattened ``L``.

        Returns:
            SPD matrices of shape ``(..., dim, dim)``.
        """
        L, sigma2 = self._unpack(params)
        eye = torch.eye(self.dim, device=params.device, dtype=params.dtype)
        return sigma2[..., None, None] * eye + torch.matmul(L, L.transpose(-1, -2))

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        """Compute logdet(S) using the matrix determinant lemma.

        For :math:`S = \\sigma^2 I + L L^\\top`, let
        :math:`M = I_r + L^\\top L / \\sigma^2`. Then
        :math:`\\log\\det S = d\\log\\sigma^2 + \\log\\det M`.
        """
        _, sigma2, cholesky = self._factor_system(params)
        return self.dim * torch.log(sigma2) + 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(-1)

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Compute r^T S^{-1} r using the Woodbury identity.

        For :math:`S = \\sigma^2 I + L L^\\top`,

        .. math::

            S^{-1} r = \\frac{r}{\\sigma^2}
                - \\frac{L M^{-1} (L^\\top r)}{\\sigma^4},

        where :math:`M = I_r + L^\\top L / \\sigma^2`.
        """
        L, sigma2, cholesky = self._factor_system(params)
        return self._statistics_from_factor(L, sigma2, cholesky, residual)[1]

    def statistics(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse one Woodbury Gram matrix and Cholesky factor for both terms."""
        L, sigma2, cholesky = self._factor_system(params)
        return self._statistics_from_factor(L, sigma2, cholesky, residual)
