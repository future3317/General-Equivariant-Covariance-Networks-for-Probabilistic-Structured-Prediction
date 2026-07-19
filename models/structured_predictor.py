"""Top-level structured probabilistic predictor."""

from __future__ import annotations

from typing import Dict
import torch

from distributions.base import StructuredDistributionLoss
from models.backbone import EquivariantBackbone
from models.mean_head import EquivariantMeanHead
from models.covariance_head import (
    O3EquivariantSymmetricOperatorHead,
    O3EquivariantLowRankCovarianceHead,
)
from representations import O3IrrepsSpec
from spd_maps.base import SPDMap


class StructuredProbabilisticPredictor(torch.nn.Module):
    """Compose backbone, mean head, covariance head, SPD map, and distribution.

    The model predicts a mean :math:`\\mu(x) \\in V` and a scale matrix
    :math:`S(x) \\in \\operatorname{SPD}(V)`. The distribution loss compares
    ``mu`` and ``target`` in the output representation space ``V``.

    Args:
        backbone: Equivariant feature extractor.
        output_spec: Specification of the output representation ``V``.
        mean_head: Head mapping hidden features to ``mu``.
        covariance_head: Head mapping hidden features to SPD-map parameters.
        spd_map: Map from covariance-head parameters to SPD matrices.
        distribution: Probabilistic loss (Gaussian, Student-t, ...).
    """

    def __init__(
        self,
        backbone: EquivariantBackbone,
        output_spec: O3IrrepsSpec,
        mean_head: EquivariantMeanHead,
        covariance_head: O3EquivariantSymmetricOperatorHead | O3EquivariantLowRankCovarianceHead,
        spd_map: SPDMap,
        distribution: StructuredDistributionLoss,
    ):
        super().__init__()
        self.backbone = backbone
        self.output_spec = output_spec
        self.mean_head = mean_head
        self.covariance_head = covariance_head
        self.spd_map = spd_map
        self.distribution = distribution

    def forward(
        self,
        data,
        target: torch.Tensor | None = None,
        return_scale: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            data: PyG-like data object.
            target: Optional target values in the output representation space.
            return_scale: If True, explicitly compute and return the SPD scale
                matrix. Training only needs the parameterization and can avoid
                this extra matrix exponential.

        Returns:
            Dictionary containing ``mu`` and ``params``, plus ``scale`` when
            ``return_scale=True`` and ``loss``/``components`` when ``target`` is
            provided.
        """
        node_features, batch = self.backbone(data)
        mu = self.mean_head(node_features, batch)
        params = self.covariance_head(node_features, batch)

        result: Dict[str, torch.Tensor] = {
            "mu": mu,
            "params": params,
        }

        if target is not None:
            loss, components = self.distribution(mu, params, target, self.spd_map)
            result["loss"] = loss
            result["components"] = components

        if return_scale:
            result["scale"] = self.spd_map(params)

        return result
