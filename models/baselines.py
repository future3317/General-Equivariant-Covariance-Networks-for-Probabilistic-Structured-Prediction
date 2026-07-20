"""Baseline prediction heads for controlled comparisons.

These heads implement simpler uncertainty parameterizations that can be
plugged into ``StructuredProbabilisticPredictor`` together with a compatible
SPD map. They are useful as fair baselines in the dielectric and elasticity
experiments.
"""

from __future__ import annotations

import torch
from compatibility.e3nn import o3
from torch_scatter import scatter

from representations import O3IrrepsSpec


class DeterministicHead(torch.nn.Module):
    """Predict only the mean :math:`\\mu(x)`. No covariance output."""

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        pool: bool = True,
    ):
        super().__init__()
        self.output_spec = output_spec
        self.pool = pool
        self.head = o3.Linear(o3.Irreps(hidden_irreps), output_spec.irreps)

    def forward(self, node_features, batch=None):
        if self.pool:
            if batch is None:
                raise ValueError("batch is required when pool=True")
            pooled = scatter(node_features, batch, dim=0, reduce="mean")
        else:
            pooled = node_features
        return self.head(pooled)


class IsotropicCovarianceHead(torch.nn.Module):
    """Predict mean and a single isotropic variance :math:`\\sigma^2 I`."""

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_spec: O3IrrepsSpec,
        pool: bool = True,
    ):
        super().__init__()
        self.output_spec = output_spec
        self.pool = pool
        self.mean_head = o3.Linear(o3.Irreps(hidden_irreps), output_spec.irreps)
        # One invariant scalar for log(sigma^2).
        self.log_sigma2_head = o3.Linear(
            o3.Irreps(hidden_irreps), o3.Irreps("1x0e")
        )

    def forward(self, node_features, batch=None):
        if self.pool:
            if batch is None:
                raise ValueError("batch is required when pool=True")
            pooled = scatter(node_features, batch, dim=0, reduce="mean")
        else:
            pooled = node_features
        mu = self.mean_head(pooled)
        log_sigma2 = self.log_sigma2_head(pooled)
        return mu, log_sigma2


class IrrepBlockDiagonalCovarianceHead(torch.nn.Module):
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
        super().__init__()
        self.output_spec = output_spec
        self.pool = pool
        self.mean_head = o3.Linear(o3.Irreps(hidden_irreps), output_spec.irreps)

        # Count irrep blocks.
        self._num_blocks = sum(mul for mul, _ in output_spec.irreps)
        self.log_var_head = o3.Linear(
            o3.Irreps(hidden_irreps),
            o3.Irreps(f"{self._num_blocks}x0e"),
        )

    def forward(self, node_features, batch=None):
        if self.pool:
            if batch is None:
                raise ValueError("batch is required when pool=True")
            pooled = scatter(node_features, batch, dim=0, reduce="mean")
        else:
            pooled = node_features
        mu = self.mean_head(pooled)
        log_vars = self.log_var_head(pooled)
        return mu, log_vars
