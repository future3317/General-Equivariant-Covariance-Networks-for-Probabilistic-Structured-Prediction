"""Top-level structured probabilistic predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from distributions.base import StructuredDistributionLoss
from models.backbone import EquivariantBackbone
from models.pooling import GraphOutputHead, mean_pool
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
        if joint_head is not None and (
            mean_head is not None or covariance_head is not None
        ):
            raise ValueError("joint_head cannot be combined with separate heads")
        if (spd_map is None) != (distribution is None):
            raise ValueError("spd_map and distribution must be provided together")
        if joint_head is None and spd_map is None:
            raise ValueError("separate mean/covariance heads require an SPD map")
        self._heads_share_pooling = False
        if joint_head is None and mean_head is not None and covariance_head is not None:
            self._heads_share_pooling = bool(
                isinstance(mean_head, GraphOutputHead)
                and isinstance(covariance_head, GraphOutputHead)
                and mean_head.pool
                and covariance_head.pool
            )
        self.spd_map = spd_map
        self.distribution = distribution

    def _predict(
        self, node_features: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run exactly one configured head path."""
        if self.joint_head is not None:
            output = self.joint_head(node_features, batch)
            if isinstance(output, torch.Tensor):
                return output, None
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("joint_head must return mu or the pair (mu, params)")
            return output
        if self._heads_share_pooling:
            pooled = mean_pool(node_features, batch)
            return (
                self.mean_head.forward_pooled(pooled),
                self.covariance_head.forward_pooled(pooled),
            )
        return (
            self.mean_head(node_features, batch),
            self.covariance_head(node_features, batch),
        )

    def forward(
        self,
        data,
        target: torch.Tensor | None = None,
        return_scale: bool = False,
        return_precision: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
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
        return self.forward_from_features(
            node_features,
            batch,
            target=target,
            return_scale=return_scale,
            return_precision=return_precision,
        )

    def forward_from_features(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
        return_scale: bool = False,
        return_precision: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Evaluate the typed readout from already computed backbone features.

        This boundary supports frozen-backbone studies and mixed precision in
        which the backbone runs under autocast while operator assembly and the
        proper likelihood remain in FP32.
        """
        # Autocast is intentionally scoped to the backbone.  Enforce the
        # readout parameter dtype here as well, so callers using cached BF16
        # features cannot accidentally send reduced-precision coordinates into
        # CG projection, matrix exponential, Cholesky, or an NLL.
        readout_parameter = next(self.parameters(), None)
        if readout_parameter is not None and node_features.dtype != readout_parameter.dtype:
            node_features = node_features.to(dtype=readout_parameter.dtype)
        mu, params = self._predict(node_features, batch)
        result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"mu": mu}

        if params is None:
            if self.spd_map is not None:
                raise TypeError("a probabilistic joint_head must return (mu, params)")
            if target is not None:
                loss = torch.nn.functional.mse_loss(mu, target)
                result["loss"] = loss
                result["components"] = {"loss_fit": loss.detach()}
            return result

        if self.spd_map is None or self.distribution is None:
            raise TypeError("a joint_head returning params requires an SPD map")
        result["params"] = params

        if target is not None:
            if target.dtype != mu.dtype:
                target = target.to(dtype=mu.dtype)
            loss, components = self.distribution(mu, params, target, self.spd_map)
            result["loss"] = loss
            result["components"] = components

        if return_scale:
            result["scale"] = self.spd_map(params)

        if return_precision:
            result["precision"] = self.spd_map.precision(params)

        return result
