"""Pooling primitives shared by graph-level prediction heads."""

from __future__ import annotations

import abc

import torch


def mean_pool(node_features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Average node features for each graph in a PyG batch.

    The implementation uses native PyTorch operations, so model code does not
    require ``torch-scatter`` merely for graph-level mean pooling.  Graph IDs
    follow the PyG convention: non-negative integers in ``[0, num_graphs)``.
    """
    if node_features.ndim < 2:
        raise ValueError("node_features must have a node and feature dimension")
    if batch.ndim != 1 or batch.shape[0] != node_features.shape[0]:
        raise ValueError("batch must be one-dimensional with one graph id per node")
    if batch.dtype != torch.long:
        raise TypeError("batch graph ids must have dtype torch.long")
    if batch.numel() == 0:
        return node_features.new_empty((0, *node_features.shape[1:]))
    if torch.any(batch < 0):
        raise ValueError("batch graph ids must be non-negative")

    counts = torch.bincount(batch)
    pooled = node_features.new_zeros((counts.shape[0], *node_features.shape[1:]))
    pooled.index_add_(0, batch, node_features)
    count_shape = (counts.shape[0],) + (1,) * (node_features.ndim - 1)
    return pooled / counts.clamp_min(1).to(node_features.dtype).reshape(count_shape)


class GraphOutputHead(torch.nn.Module, abc.ABC):
    """Base class for node-level heads with optional graph mean pooling."""

    def __init__(self, *, pool: bool = True):
        super().__init__()
        self.pool = bool(pool)

    @abc.abstractmethod
    def forward_pooled(self, pooled_features: torch.Tensor):
        """Project node features or features pooled by the caller."""
        ...

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor | None = None,
    ):
        if self.pool:
            if batch is None:
                raise ValueError("batch is required when pool=True")
            node_features = mean_pool(node_features, batch)
        return self.forward_pooled(node_features)
