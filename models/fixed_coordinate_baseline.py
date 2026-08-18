"""Conventional fixed-coordinate heteroscedastic uncertainty baseline.

This module intentionally does not use the equivariant compiler.  It supplies
the ordinary diagonal readout requested as an external control while reusing
the production Student-t objective and evaluation path.
"""

from __future__ import annotations

import torch

from spd_maps.base import SPDMap


class FixedCoordinateDiagonalReadout(torch.nn.Module):
    """Predict independent log scatter in the dataset coordinate frame."""

    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        self.projection = torch.nn.Linear(feature_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class FixedCoordinateDiagonalMap(SPDMap):
    """Map coordinate-wise log scatter to a positive diagonal matrix."""

    @staticmethod
    def _variance(params: torch.Tensor) -> torch.Tensor:
        return torch.exp(params)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(self._variance(params))

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        return params.sum(-1)

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return (residual.square() / self._variance(params)).sum(-1)


class FixedCoordinateCholeskyReadout(torch.nn.Module):
    """Predict a lower-triangular factor in the declared coordinate frame.

    The module is deliberately outside the equivariant compiler.  It is an
    external coordinate-dependent SPD control, not an equivariant family.
    Diagonal entries are represented as unconstrained values and transformed
    by :class:`FixedCoordinateCholeskyMap`.
    """

    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        if feature_dim < 1 or output_dim < 1:
            raise ValueError("feature_dim and output_dim must be positive")
        self.output_dim = int(output_dim)
        self.parameter_count = self.output_dim * (self.output_dim + 1) // 2
        self.projection = torch.nn.Linear(feature_dim, self.parameter_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class FixedCoordinateCholeskyMap(SPDMap):
    """Map unconstrained triangular coordinates to an SPD matrix."""

    def __init__(self, output_dim: int, *, diagonal_floor: float = 1e-6):
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        if diagonal_floor <= 0:
            raise ValueError("diagonal_floor must be positive")
        self.output_dim = int(output_dim)
        self.diagonal_floor = float(diagonal_floor)
        rows, columns = torch.tril_indices(self.output_dim, self.output_dim)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("columns", columns, persistent=False)
        diagonal = torch.arange(self.output_dim)
        diagonal_index = []
        for row in diagonal.tolist():
            diagonal_index.append(
                int(((self.rows == row) & (self.columns == row)).nonzero()[0])
            )
        self.register_buffer(
            "diagonal_index",
            torch.tensor(diagonal_index, dtype=torch.long),
            persistent=False,
        )

    @property
    def parameter_count(self) -> int:
        return self.output_dim * (self.output_dim + 1) // 2

    def _factor(self, params: torch.Tensor) -> torch.Tensor:
        if params.shape[-1] != self.parameter_count:
            raise ValueError(
                f"expected {self.parameter_count} Cholesky parameters, "
                f"got {params.shape[-1]}"
            )
        factor = params.new_zeros(*params.shape[:-1], self.output_dim, self.output_dim)
        factor[..., self.rows, self.columns] = params
        diagonal = params.index_select(-1, self.diagonal_index)
        diagonal_rows = torch.arange(self.output_dim, device=self.rows.device)
        factor[..., diagonal_rows, diagonal_rows] = (
            torch.nn.functional.softplus(diagonal) + self.diagonal_floor
        )
        return factor

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        factor = self._factor(params)
        return factor @ factor.transpose(-1, -2)

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        factor = self._factor(params)
        return 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        factor = self._factor(params)
        solved = torch.linalg.solve_triangular(
            factor, residual.unsqueeze(-1), upper=False
        ).squeeze(-1)
        return solved.square().sum(-1)
