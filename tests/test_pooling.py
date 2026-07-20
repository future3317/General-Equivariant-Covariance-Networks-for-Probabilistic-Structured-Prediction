"""Tests for native graph pooling shared by model heads."""

import pytest
import torch

from models.pooling import GraphOutputHead, mean_pool


class _IdentityHead(GraphOutputHead):
    def forward_pooled(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return pooled_features


def test_mean_pool_matches_manual_graph_averages():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [8.0, 10.0], [4.0, 6.0]])
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    expected = torch.tensor([[2.0, 3.0], [6.0, 8.0]])
    torch.testing.assert_close(mean_pool(features, batch), expected)


def test_mean_pool_preserves_empty_batch_shape():
    features = torch.empty(0, 7)
    pooled = mean_pool(features, torch.empty(0, dtype=torch.long))
    assert pooled.shape == (0, 7)


@pytest.mark.parametrize(
    ("features", "batch", "error"),
    [
        (torch.randn(2, 3), torch.tensor([[0, 0]]), ValueError),
        (torch.randn(2, 3), torch.tensor([0, 0], dtype=torch.int32), TypeError),
        (torch.randn(2, 3), torch.tensor([0, -1]), ValueError),
    ],
)
def test_mean_pool_rejects_invalid_batch_ids(features, batch, error):
    with pytest.raises(error):
        mean_pool(features, batch)


def test_graph_output_head_owns_the_pooling_contract():
    features = torch.tensor([[1.0], [3.0], [8.0]])
    batch = torch.tensor([0, 0, 1], dtype=torch.long)
    torch.testing.assert_close(
        _IdentityHead(pool=True)(features, batch), torch.tensor([[2.0], [8.0]])
    )
    torch.testing.assert_close(_IdentityHead(pool=False)(features), features)
    with pytest.raises(ValueError, match="batch"):
        _IdentityHead(pool=True)(features)
