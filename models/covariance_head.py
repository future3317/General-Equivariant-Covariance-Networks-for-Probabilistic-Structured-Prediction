"""Equivariant covariance heads."""

from __future__ import annotations

import torch
from e3nn import o3
from torch_scatter import scatter

from representations import O3IrrepsSpec, O3SymmetricOperatorBasis


class O3EquivariantSymmetricOperatorHead(torch.nn.Module):
    """Predict an equivariant symmetric operator ``A(x)`` from node features.

    The output coefficients live in :math:`\\operatorname{Sym}^2(V)` and are
    assembled into a symmetric matrix by ``O3SymmetricOperatorBasis``.
    """

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        pool: bool = True,
    ):
        super().__init__()
        self.output_spec = output_spec
        self.pool = pool
        self.operator_basis = output_spec.symmetric_square()
        self.coefficient_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            self.operator_basis.operator_irreps,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pool:
            if batch is None:
                raise ValueError("batch is required when pool=True")
            pooled = scatter(node_features, batch, dim=0, reduce="mean")
            coeffs = self.coefficient_head(pooled)
        else:
            coeffs = self.coefficient_head(node_features)
        return self.operator_basis.assemble(coeffs)


class O3EquivariantLowRankCovarianceHead(torch.nn.Module):
    """Predict a low-rank-plus-isotropic covariance from node features.

    The head outputs ``rank`` copies of the output representation ``V`` for the
    factor matrix ``L``, plus one invariant scalar for ``log(\\sigma^2)``. The
    parameters are concatenated into a single vector accepted by
    ``LowRankPlusIsotropicMap``.
    """

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        rank: int,
        pool: bool = True,
    ):
        super().__init__()
        self.output_spec = output_spec
        self.rank = rank
        self.pool = pool
        self.factor_irreps = output_spec.irreps * rank
        self.factor_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            self.factor_irreps,
        )
        self.log_sigma2_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            o3.Irreps("1x0e"),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pool:
            if batch is None:
                raise ValueError("batch is required when pool=True")
            pooled = scatter(node_features, batch, dim=0, reduce="mean")
            factors = self.factor_head(pooled)
            log_sigma2 = self.log_sigma2_head(pooled)
        else:
            factors = self.factor_head(node_features)
            log_sigma2 = self.log_sigma2_head(node_features)

        batch_size = factors.shape[0]
        dim = self.output_spec.dim
        L = factors.reshape(batch_size, dim, self.rank)
        return torch.cat([L.reshape(batch_size, dim * self.rank), log_sigma2], dim=-1)
