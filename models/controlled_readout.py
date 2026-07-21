"""Controlled readouts that isolate covariance-family comparisons."""

from __future__ import annotations

import torch

from models.pooling import GraphOutputHead


class ControlledMeanOperatorHead(torch.nn.Module):
    """Pair one shared mean readout with compiled operator parameters.

    The compiled head retains the compiler-certified covariance lifting, while
    the mean is predicted through an identical direct readout for every
    operator family. This is useful for controlled uncertainty experiments.
    """

    def __init__(
        self,
        mean_head: GraphOutputHead,
        operator_head: torch.nn.Module,
    ):
        super().__init__()
        if not hasattr(operator_head, "forward_parameters"):
            raise TypeError("operator_head must expose forward_parameters")
        self.mean_head = mean_head
        self.operator_head = operator_head
        # This projection belongs to the compiler's usual joint readout but is
        # intentionally bypassed by the controlled mean path.
        for parameter in operator_head.mean_projection.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.mean_head(node_features, batch),
            self.operator_head.forward_parameters(node_features, batch),
        )
