"""SPD covariance on O(3) isotypic multiplicity spaces."""

from __future__ import annotations

import torch
from compatibility.e3nn import o3

from spd_maps.base import SPDMap


class IsotypicBlockMap(SPDMap):
    r"""Assemble ``K_l,p (x) I_(2l+1)`` for each isotypic component.

    Unlike a scalar-per-copy diagonal baseline, this map learns a full SPD
    matrix between repeated copies of the same irrep.  It is the complete
    input-dependent covariance that commutes with the O(3) action while using
    invariant parameters only.
    """

    def __init__(self, irreps: o3.Irreps, min_diagonal: float = 1e-4):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.min_diagonal = min_diagonal
        groups: dict[o3.Irrep, list[list[int]]] = {}
        cursor = 0
        for multiplicity, irrep in self.irreps:
            copies = groups.setdefault(irrep, [])
            for _ in range(multiplicity):
                copies.append(list(range(cursor, cursor + irrep.dim)))
                cursor += irrep.dim
        self._groups = tuple(
            (irrep, tuple(tuple(copy) for copy in copies))
            for irrep, copies in sorted(
                groups.items(), key=lambda item: (item[0].l, -item[0].p)
            )
        )
        self._parameter_slices: list[tuple[int, int]] = []
        parameter_cursor = 0
        for _, copies in self._groups:
            multiplicity = len(copies)
            count = multiplicity * (multiplicity + 1) // 2
            self._parameter_slices.append(
                (parameter_cursor, parameter_cursor + count)
            )
            parameter_cursor += count
        self._num_parameters = parameter_cursor

    @property
    def num_parameters(self) -> int:
        return self._num_parameters

    def _blocks(self, params: torch.Tensor) -> list[torch.Tensor]:
        if params.shape[-1] != self.num_parameters:
            raise ValueError(
                f"params last dim {params.shape[-1]} != {self.num_parameters}"
            )
        blocks = []
        for (_, copies), (start, end) in zip(self._groups, self._parameter_slices):
            multiplicity = len(copies)
            lower = params.new_zeros((*params.shape[:-1], multiplicity, multiplicity))
            rows, cols = torch.tril_indices(
                multiplicity, multiplicity, device=params.device
            )
            lower[..., rows, cols] = params[..., start:end]
            diagonal = torch.arange(multiplicity, device=params.device)
            lower[..., diagonal, diagonal] = (
                torch.nn.functional.softplus(lower[..., diagonal, diagonal])
                + self.min_diagonal
            )
            blocks.append(lower @ lower.transpose(-1, -2))
        return blocks

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        covariance = params.new_zeros(
            (*params.shape[:-1], self.irreps.dim, self.irreps.dim)
        )
        for (irrep, copies), block in zip(self._groups, self._blocks(params)):
            for row_copy, row_indices in enumerate(copies):
                for col_copy, col_indices in enumerate(copies):
                    covariance[..., row_indices, col_indices] = (
                        block[..., row_copy, col_copy, None]
                        * torch.ones(irrep.dim, device=params.device, dtype=params.dtype)
                    )
        return covariance

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        result = params.new_zeros(params.shape[:-1])
        for (irrep, _), block in zip(self._groups, self._blocks(params)):
            result = result + irrep.dim * torch.linalg.slogdet(block).logabsdet
        return result

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        if residual.shape[-1] != self.irreps.dim:
            raise ValueError(
                f"residual last dim {residual.shape[-1]} != {self.irreps.dim}"
            )
        result = params.new_zeros(params.shape[:-1])
        for (irrep, copies), block in zip(self._groups, self._blocks(params)):
            indices = [index for copy in copies for index in copy]
            values = residual[..., indices].reshape(
                *residual.shape[:-1], len(copies), irrep.dim
            )
            solved = torch.linalg.solve(block, values)
            result = result + torch.sum(values * solved, dim=(-2, -1))
        return result
