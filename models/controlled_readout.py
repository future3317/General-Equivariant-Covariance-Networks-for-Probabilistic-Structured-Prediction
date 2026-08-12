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
            self.forward_mean(node_features, batch),
            self.operator_head.forward_parameters(node_features, batch),
        )

    def forward_mean(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the family-independent mean path."""
        return self.mean_head(node_features, batch)

    def forward_parameters_detached_features(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate compiled operator parameters behind the faithful boundary."""
        return self.operator_head.forward_parameters_detached_features(
            node_features, batch
        )


class ControlledMeanPooledParameterHead(torch.nn.Module):
    """Pair the shared mean with a conventional pooled parameter readout."""

    def __init__(self, mean_head: GraphOutputHead, parameter_head: torch.nn.Module):
        super().__init__()
        self.mean_head = mean_head
        self.operator_head = parameter_head

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if node_features.shape[0] != batch.shape[0]:
            raise ValueError("batch must index every feature row")
        counts = torch.bincount(batch)
        pooled = node_features.new_zeros((counts.shape[0], node_features.shape[-1]))
        pooled.index_add_(0, batch, node_features)
        pooled = pooled / counts.clamp_min(1).to(node_features.dtype).unsqueeze(-1)
        return self.mean_head(node_features, batch), self.operator_head(pooled)
