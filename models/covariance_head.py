"""Equivariant covariance heads."""

from __future__ import annotations

import torch
from compatibility.e3nn import o3

from models.pooling import GraphOutputHead
from representations import O3IrrepsSpec
from representations.cartesian_stf import (
    MultiplicityFirstCartesianTensorSquare,
    Rank2CartesianSTFOperatorBasis,
    is_rank2_stf_output,
    supports_cartesian_stf_seed,
)
from representations.irrep_layout import RepeatedIrrepLayout


QUADRATIC_HEAD_BACKENDS = ("auto", "spherical_cg", "cartesian_stf")


class O3QuadraticSymmetricOperatorHead(GraphOutputHead):
    """Graph-level quadratic equivariant head for symmetric operators.

    After graph pooling, the head applies a small equivariant bottleneck and
    then produces :math:`\\operatorname{Sym}^2(V)` coefficients through two
    branches:

    * an equivariant linear branch, and
    * an equivariant quadratic branch via either spherical CG or exact
      multiplicity-first STF-coordinate/dense-projector lowering.

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
        backend: str = "auto",
        contraction_rank: int | None = None,
    ):
        super().__init__(pool=pool)
        self.output_spec = output_spec
        self.bottleneck_irreps = o3.Irreps(bottleneck_irreps)
        if backend not in QUADRATIC_HEAD_BACKENDS:
            raise ValueError(
                f"unknown quadratic head backend {backend!r}; "
                f"expected one of {QUADRATIC_HEAD_BACKENDS}"
            )
        stf_supported = is_rank2_stf_output(
            output_spec.irreps
        ) and supports_cartesian_stf_seed(self.bottleneck_irreps)
        if backend == "auto":
            backend = "cartesian_stf" if stf_supported else "spherical_cg"
        if backend == "cartesian_stf" and not stf_supported:
            raise ValueError(
                "cartesian_stf is exact only for V=0e+2e with bottleneck "
                "types drawn from 0e, 1o and 2e"
            )
        if contraction_rank is not None and backend != "cartesian_stf":
            raise ValueError("contraction_rank is only valid for cartesian_stf")
        self.backend = backend
        self.operator_basis = (
            Rank2CartesianSTFOperatorBasis()
            if backend == "cartesian_stf"
            else output_spec.symmetric_square()
        )

        self.pre = o3.Linear(
            o3.Irreps(hidden_irreps),
            self.bottleneck_irreps,
        )
        self.linear = o3.Linear(
            self.bottleneck_irreps,
            self.operator_basis.operator_irreps,
        )
        self.square = (
            MultiplicityFirstCartesianTensorSquare(
                self.bottleneck_irreps,
                self.operator_basis.operator_irreps,
                contraction_rank=contraction_rank,
            )
            if backend == "cartesian_stf"
            else o3.TensorSquare(
                self.bottleneck_irreps,
                irreps_out=self.operator_basis.operator_irreps,
            )
        )

    @property
    def is_exact_backend(self) -> bool:
        """Whether the selected execution backend preserves the full CG map."""
        return self.backend == "spherical_cg" or bool(self.square.is_exact)

    def load_spherical_head(self, spherical_head: "O3QuadraticSymmetricOperatorHead") -> None:
        """Map a spherical-CG head to the exact dense-projector backend."""
        if self.backend != "cartesian_stf" or not self.square.is_exact:
            raise RuntimeError("weight mapping requires an exact cartesian_stf head")
        if spherical_head.backend != "spherical_cg":
            raise ValueError("source head must use spherical_cg")
        self.pre.load_state_dict(spherical_head.pre.state_dict())
        self.linear.load_state_dict(spherical_head.linear.state_dict())
        self.square.load_e3nn_weights(spherical_head.square)

    def forward_pooled(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Project features that have already been graph pooled."""
        z = self.pre(pooled_features)
        coefficients = self.linear(z) + self.square(z)
        return self.operator_basis.assemble(coefficients)


class O3EquivariantSymmetricOperatorHead(GraphOutputHead):
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
        super().__init__(pool=pool)
        self.output_spec = output_spec
        self.operator_basis = output_spec.symmetric_square()
        self.coefficient_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            self.operator_basis.operator_irreps,
        )

    def forward_pooled(self, pooled_features: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficient_head(pooled_features)
        return self.operator_basis.assemble(coefficients)


class O3EquivariantLowRankCovarianceHead(GraphOutputHead):
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
        super().__init__(pool=pool)
        self.output_spec = output_spec
        self.factor_layout = RepeatedIrrepLayout(output_spec.irreps, rank)
        self.rank = self.factor_layout.copies
        self.factor_irreps = self.factor_layout.expanded_irreps
        self.factor_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            self.factor_irreps,
        )
        self.log_sigma2_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            o3.Irreps("1x0e"),
        )

    def forward_pooled(self, pooled_features: torch.Tensor) -> torch.Tensor:
        factors = self.factor_head(pooled_features)
        log_sigma2 = self.log_sigma2_head(pooled_features)
        factor_matrix = self._pack_factors(factors)
        return torch.cat([factor_matrix.flatten(start_dim=-2), log_sigma2], dim=-1)

    def _pack_factors(self, factors: torch.Tensor) -> torch.Tensor:
        """Pack factor_irreps output into L of shape (..., dim, rank)."""
        return self.factor_layout.pack(factors).transpose(-1, -2)
