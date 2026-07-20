"""Equivariant mean head."""

from __future__ import annotations

import torch
from compatibility.e3nn import o3

from models.pooling import GraphOutputHead


class EquivariantMeanHead(GraphOutputHead):
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
        super().__init__(pool=pool)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_irreps = o3.Irreps(output_irreps)
        self.linear = o3.Linear(self.hidden_irreps, self.output_irreps)

    def forward_pooled(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Project features that have already been graph pooled."""
        return self.linear(pooled_features)
