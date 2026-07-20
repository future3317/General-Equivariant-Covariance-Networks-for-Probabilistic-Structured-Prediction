"""Equivariant covariance heads."""

from __future__ import annotations

import torch
from compatibility.e3nn import o3
from torch_scatter import scatter

from representations import O3IrrepsSpec, O3SymmetricOperatorBasis


class O3QuadraticSymmetricOperatorHead(torch.nn.Module):
    """Graph-level quadratic equivariant head for symmetric operators.

    After graph pooling, the head applies a small equivariant bottleneck and
    then produces :math:`\\operatorname{Sym}^2(V)` coefficients through two
    branches:

    * an equivariant linear branch, and
    * an equivariant quadratic branch via ``e3nn.o3.TensorSquare``.

    The quadratic branch is essential when the edge-level backbone is capped at
    ``lmax=2`` but the output representation requires :math:`\\ell=4` covariance
    coefficients (e.g. ``V = 0e + 2e``).
    """

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        bottleneck_irreps: o3.Irreps = "16x0e + 8x1o + 8x2e",
        pool: bool = True,
    ):
        super().__init__()
        self.output_spec = output_spec
        self.pool = pool
        self.bottleneck_irreps = o3.Irreps(bottleneck_irreps)
        self.operator_basis = output_spec.symmetric_square()

        self.pre = o3.Linear(
            o3.Irreps(hidden_irreps),
            self.bottleneck_irreps,
        )
        self.linear = o3.Linear(
            self.bottleneck_irreps,
            self.operator_basis.operator_irreps,
        )
        self.square = o3.TensorSquare(
            self.bottleneck_irreps,
            irreps_out=self.operator_basis.operator_irreps,
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
        else:
            pooled = node_features

        z = self.pre(pooled)
        coeffs = self.linear(z) + self.square(z)
        return self.operator_basis.assemble(coeffs)


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
        # Build factor_irreps with multiplicity rank for each output irrep.
        # The e3nn layout concatenates all rank copies of each irrep type.
        self.factor_irreps = o3.Irreps(
            [(mul * rank, ir) for mul, ir in output_spec.irreps]
        )
        self.factor_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            self.factor_irreps,
        )
        self.log_sigma2_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            o3.Irreps("1x0e"),
        )
        # Precompute slices so we can pack each rank slot into a full V vector.
        self._factor_slices = self._build_factor_slices(output_spec.irreps, rank)

    @staticmethod
    def _build_factor_slices(output_irreps: o3.Irreps, rank: int) -> list[list[tuple[int, int]]]:
        """Return, for each rank slot, the list of (start, end) slices in factor_irreps.

        Layout of factor_irreps: for each irrep type (mul, ir), all rank*mul copies
        are stored consecutively. Within that block, copies are ordered by rank slot.
        """
        slices_per_rank = [[] for _ in range(rank)]
        cursor = 0
        for mul, ir in output_irreps:
            dim = ir.dim
            total_copies = mul * rank
            block_size = total_copies * dim
            for copy_idx in range(total_copies):
                rank_slot = copy_idx % rank
                start = cursor + copy_idx * dim
                end = start + dim
                slices_per_rank[rank_slot].append((start, end))
            cursor += block_size
        return slices_per_rank

    def _pack_factors(self, factors: torch.Tensor) -> torch.Tensor:
        """Pack factor_irreps output into L of shape (..., dim, rank)."""
        *batch_shape, _ = factors.shape
        dim = self.output_spec.dim
        L = factors.new_empty((*batch_shape, dim, self.rank))
        for rank_slot, slices in enumerate(self._factor_slices):
            row_cursor = 0
            for start, end in slices:
                width = end - start
                L[..., row_cursor : row_cursor + width, rank_slot] = factors[..., start:end]
                row_cursor += width
        return L

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

        L = self._pack_factors(factors)
        batch_size = L.shape[0]
        return torch.cat([L.reshape(batch_size, -1), log_sigma2], dim=-1)
