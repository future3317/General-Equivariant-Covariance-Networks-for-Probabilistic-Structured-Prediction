"""Predictor wrapper for baseline heads that output mean and covariance params."""

from __future__ import annotations

from typing import Dict
import torch

from distributions.base import StructuredDistributionLoss
from models.backbone import EquivariantBackbone
from models.baselines import (
    DeterministicHead,
    IsotropicCovarianceHead,
    IrrepBlockDiagonalCovarianceHead,
)
from representations import O3IrrepsSpec
from spd_maps.base import SPDMap


class BaselineProbabilisticPredictor(torch.nn.Module):
    """Predictor for baseline heads that jointly output ``mu`` and params.

    Unlike ``StructuredProbabilisticPredictor`` which uses separate mean and
    covariance heads, this wrapper is intended for the simpler baseline heads in
    ``models.baselines``. It calls the baseline head, maps the covariance params
    through ``spd_map``, and optionally evaluates a distribution loss.

    Args:
        backbone: Equivariant feature extractor.
        output_spec: Specification of the output representation ``V``.
        baseline_head: A head from ``models.baselines``.
        spd_map: Map from baseline params to SPD matrices.
        distribution: Probabilistic loss. May be ``None`` for deterministic heads.
    """

    def __init__(
        self,
        backbone: EquivariantBackbone,
        output_spec: O3IrrepsSpec,
        baseline_head: DeterministicHead | IsotropicCovarianceHead | IrrepBlockDiagonalCovarianceHead,
        spd_map: SPDMap | None,
        distribution: StructuredDistributionLoss | None,
    ):
        super().__init__()
        self.backbone = backbone
        self.output_spec = output_spec
        self.baseline_head = baseline_head
        self.spd_map = spd_map
        self.distribution = distribution

    def forward(
        self,
        data,
        target: torch.Tensor | None = None,
        return_scale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        node_features, batch = self.backbone(data)
        head_output = self.baseline_head(node_features, batch)

        if isinstance(head_output, tuple):
            mu, params = head_output
        else:
            mu = head_output
            params = None

        result: Dict[str, torch.Tensor] = {"mu": mu}

        if params is not None and self.spd_map is not None:
            result["params"] = params
            if return_scale:
                result["scale"] = self.spd_map(params)

            if target is not None and self.distribution is not None:
                loss, components = self.distribution(mu, params, target, self.spd_map)
                result["loss"] = loss
                result["components"] = components
        else:
            # Deterministic case.
            if target is not None:
                mse = torch.nn.functional.mse_loss(mu, target)
                result["loss"] = mse

        return result
