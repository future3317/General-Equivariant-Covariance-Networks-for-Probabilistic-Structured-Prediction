"""Compatibility wrapper for baseline heads."""

from __future__ import annotations

import torch

from distributions.base import StructuredDistributionLoss
from models.backbone import EquivariantBackbone
from models.baselines import (
    DeterministicHead,
    IsotropicCovarianceHead,
    IrrepBlockDiagonalCovarianceHead,
)
from models.structured_predictor import StructuredProbabilisticPredictor
from representations import O3IrrepsSpec
from spd_maps.base import SPDMap


class BaselineProbabilisticPredictor(StructuredProbabilisticPredictor):
    """Predictor for baseline heads that jointly output ``mu`` and params.

    This preserves the historical constructor and default ``return_scale``
    behavior while delegating prediction assembly to the unified predictor.

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
        baseline_head: DeterministicHead
        | IsotropicCovarianceHead
        | IrrepBlockDiagonalCovarianceHead,
        spd_map: SPDMap | None,
        distribution: StructuredDistributionLoss | None,
    ):
        super().__init__(
            backbone=backbone,
            output_spec=output_spec,
            joint_head=baseline_head,
            spd_map=spd_map,
            distribution=distribution,
        )

    @property
    def baseline_head(self) -> torch.nn.Module:
        """The joint head, under the historical public attribute name."""
        return self.joint_head

    def forward(
        self,
        data,
        target: torch.Tensor | None = None,
        return_scale: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        return super().forward(
            data,
            target=target,
            return_scale=return_scale,
        )
