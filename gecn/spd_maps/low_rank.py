"""Low-rank-plus-isotropic SPD map."""

import torch

from gecn.spd_maps.base import SPDMap


class LowRankPlusIsotropicMap(SPDMap):
    """Structured low-rank SPD map: :math:`S = \\sigma^2 I + L L^T`.

    The factor matrix ``L`` is a ``(d, rank)`` matrix whose columns transform
    as vectors in the output representation ``V``. This makes the model
    equivariant and parameter-efficient for high-dimensional outputs such as
    the 21-dimensional elasticity tensor.
    """

    def __init__(self, dim: int, rank: int, min_sigma2: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.min_sigma2 = min_sigma2

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Assemble SPD matrix from unconstrained parameters.

        Args:
            params: Tensor of shape ``(..., dim * rank + 1)`` where the last
                entry is the log-variance parameter and the leading entries
                are flattened ``L``.

        Returns:
            SPD matrices of shape ``(..., dim, dim)``.
        """
        *batch, _ = params.shape
        L_flat = params[..., :-1]
        log_sigma2 = params[..., -1]
        L = L_flat.reshape(*batch, self.dim, self.rank)
        sigma2 = torch.nn.functional.softplus(log_sigma2) + self.min_sigma2
        eye = torch.eye(self.dim, device=params.device, dtype=params.dtype)
        return sigma2[..., None, None] * eye + torch.matmul(L, L.transpose(-1, -2))

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        S = self.forward(params)
        return torch.logdet(S)

    def precision_action(self, params: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        S = self.forward(params)
        x = torch.linalg.solve(S, residual.unsqueeze(-1))
        return torch.sum(residual * x.squeeze(-1), dim=-1)
