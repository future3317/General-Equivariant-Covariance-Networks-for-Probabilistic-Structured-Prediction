"""Baseline prediction heads for controlled comparisons.

These heads implement simpler uncertainty parameterizations that can be
plugged into ``StructuredProbabilisticPredictor`` together with a compatible
SPD map. They are useful as fair baselines in the dielectric and elasticity
experiments.
"""

from __future__ import annotations

import torch
from compatibility.e3nn import o3

from models.pooling import GraphOutputHead
from representations import O3IrrepsSpec


class DeterministicHead(GraphOutputHead):
    """Predict only the mean :math:`\\mu(x)`. No covariance output."""

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        pool: bool = True,
    ):
        super().__init__(pool=pool)
        self.output_spec = output_spec
        self.head = o3.Linear(o3.Irreps(hidden_irreps), output_spec.irreps)

    def forward_pooled(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.head(pooled_features)


class IsotropicCovarianceHead(GraphOutputHead):
    """Predict mean and a single isotropic variance :math:`\\sigma^2 I`."""

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        pool: bool = True,
    ):
        super().__init__(pool=pool)
        self.output_spec = output_spec
        self.mean_head = o3.Linear(o3.Irreps(hidden_irreps), output_spec.irreps)
        # One invariant scalar for log(sigma^2).
        self.log_sigma2_head = o3.Linear(o3.Irreps(hidden_irreps), o3.Irreps("1x0e"))

    def forward_pooled(
        self, pooled_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mean_head(pooled_features)
        log_sigma2 = self.log_sigma2_head(pooled_features)
        return mu, log_sigma2


class IrrepBlockDiagonalCovarianceHead(GraphOutputHead):
    """Predict mean and one variance per irrep block.

    The output covariance is block-diagonal with blocks :math:`\\sigma_\\lambda^2 I`
    on each irrep subspace. This preserves O(3) equivariance while being more
    flexible than a fully isotropic covariance.
    """

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        pool: bool = True,
    ):
        super().__init__(pool=pool)
        self.output_spec = output_spec
        self.mean_head = o3.Linear(o3.Irreps(hidden_irreps), output_spec.irreps)

        # Count irrep blocks.
        self._num_blocks = sum(mul for mul, _ in output_spec.irreps)
        self.log_var_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            o3.Irreps(f"{self._num_blocks}x0e"),
        )

    def forward_pooled(
        self, pooled_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mean_head(pooled_features)
        log_vars = self.log_var_head(pooled_features)
        return mu, log_vars
