"""Top-level structured probabilistic predictor."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING
import torch

from distributions.base import StructuredDistributionLoss
from models.backbone import EquivariantBackbone
from representations import O3IrrepsSpec
from spd_maps.base import SPDMap

if TYPE_CHECKING:
    from representations.compiler import O3Compilation


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
        mean_head: torch.nn.Module | None = None,
        covariance_head: torch.nn.Module | None = None,
        spd_map: SPDMap | None = None,
        distribution: StructuredDistributionLoss | None = None,
        *,
        joint_head: torch.nn.Module | None = None,
        compilation: "O3Compilation | None" = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.output_spec = output_spec
        self.mean_head = mean_head
        self.covariance_head = covariance_head
        self.joint_head = joint_head
        self.compilation = compilation
        if joint_head is None and (mean_head is None or covariance_head is None):
            raise ValueError("provide joint_head or both mean_head and covariance_head")
        if joint_head is not None and (mean_head is not None or covariance_head is not None):
            raise ValueError("joint_head cannot be combined with separate heads")
        if spd_map is None or distribution is None:
            raise ValueError("spd_map and distribution are required")
        self.spd_map = spd_map
        self.distribution = distribution

    def forward(
        self,
        data,
        target: torch.Tensor | None = None,
        return_scale: bool = False,
        return_precision: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            data: PyG-like data object.
            target: Optional target values in the output representation space.
            return_scale: If True, explicitly compute and return the SPD scale
                matrix. Training only needs the parameterization and can avoid
                this extra matrix exponential.
            return_precision: If True, return the precision matrix. Graph
                precision models assemble it without inverting covariance.

        Returns:
            Dictionary containing ``mu`` and ``params``, plus ``scale`` when
            ``return_scale=True`` and ``loss``/``components`` when ``target`` is
            provided.
        """
        node_features, batch = self.backbone(data)
        if self.joint_head is not None:
            mu, params = self.joint_head(node_features, batch)
        else:
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

        if return_precision:
            result["precision"] = self.spd_map.precision(params)

        return result
