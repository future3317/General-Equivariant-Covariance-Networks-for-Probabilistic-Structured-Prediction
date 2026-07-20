"""Materialized modules produced from immutable compilation plans."""

from __future__ import annotations

from typing import Any

import torch

from representations import O3Compilation


class CompiledProbabilisticReadout(torch.nn.Module):
    """Probability-producing readout for a declared feature contract."""

    def __init__(self, compilation: O3Compilation):
        super().__init__()
        self.compilation = compilation
        self.head = compilation.build_head()
        self.spd_map = compilation.build_spd_map()
        self.distribution = compilation.build_distribution()
        self.irreps_in = compilation.seed_irreps
        self.irreps_out = compilation.mean_irreps

    def forward(
        self,
        seed_features: torch.Tensor,
        batch: torch.Tensor | None = None,
        *,
        target: torch.Tensor | None = None,
        return_scale: bool = False,
        return_precision: bool = False,
    ) -> dict[str, Any]:
        mean, parameters = self.head(seed_features, batch)
        result: dict[str, Any] = {"mu": mean, "params": parameters}
        if target is not None:
            loss, components = self.distribution(
                mean, parameters, target, self.spd_map
            )
            result["loss"] = loss
            result["components"] = components
        if return_scale:
            result["scale"] = self.spd_map(parameters)
        if return_precision:
            result["precision"] = self.spd_map.precision(parameters)
        return result
