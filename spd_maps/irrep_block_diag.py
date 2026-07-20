"""Block-diagonal covariance with one scalar variance per irrep block.

This preserves O(3) equivariance because each block is proportional to the
identity on its irrep subspace, and the identity commutes with the
representation matrices.
"""

from __future__ import annotations

import torch
from compatibility.e3nn import o3

from spd_maps.base import SPDMap


class IrrepBlockDiagonalMap(SPDMap):
    """Block-diagonal covariance: one :math:`\\sigma_\\lambda^2 I` per irrep.

    Parameters have shape ``(..., num_blocks)`` where each entry is
    :math:`\\log(\\sigma_\\lambda^2)` for the corresponding irrep block.
    """

    def __init__(self, irreps: o3.Irreps, min_sigma2: float = 1e-4):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.min_sigma2 = min_sigma2

        # Build block sizes and a mapping from parameter index to output slices.
        self._block_sizes = []
        self._block_slices = []
        start = 0
        for mul, ir in self.irreps:
            for _ in range(mul):
                dim = ir.dim
                self._block_sizes.append(dim)
                self._block_slices.append((start, start + dim))
                start += dim

    @property
    def num_blocks(self) -> int:
        return len(self._block_sizes)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if params.shape[-1] != self.num_blocks:
            raise ValueError(
                f"params last dim {params.shape[-1]} != num_blocks {self.num_blocks}"
            )
        *batch, _ = params.shape
        dim = self.irreps.dim
        S = torch.zeros(*batch, dim, dim, device=params.device, dtype=params.dtype)
        sigma2 = torch.nn.functional.softplus(params) + self.min_sigma2
        for block_idx, ((i, j), size) in enumerate(zip(self._block_slices, self._block_sizes)):
            S[..., i:j, i:j] = sigma2[..., block_idx][..., None, None] * torch.eye(
                size, device=params.device, dtype=params.dtype
            )
        return S

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        S = self.forward(params)
        return torch.logdet(S)

    def precision_action(self, params: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        S = self.forward(params)
        x = torch.linalg.solve(S, residual.unsqueeze(-1))
        return torch.sum(residual * x.squeeze(-1), dim=-1)
