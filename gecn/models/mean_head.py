"""Equivariant mean head."""

from __future__ import annotations

import torch
from e3nn import o3
from torch_scatter import scatter


class EquivariantMeanHead(torch.nn.Module):
    """Predict the mean :math:`\\mu(x) \\in V` from node features.

    If ``pool`` is ``True`` (default) the input is graph-pooled before the
    final linear projection. Set ``pool=False`` for node-level predictions.
    """

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        pool: bool = True,
    ):
        super().__init__()
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_irreps = o3.Irreps(output_irreps)
        self.pool = pool
        self.linear = o3.Linear(self.hidden_irreps, self.output_irreps)

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pool:
            if batch is None:
                raise ValueError("batch is required when pool=True")
            pooled = scatter(node_features, batch, dim=0, reduce="mean")
            return self.linear(pooled)
        return self.linear(node_features)
