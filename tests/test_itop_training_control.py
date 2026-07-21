"""Regression tests for ITOP optimization, stopping, and exact resume state."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from scripts.train_itop import (
    _capture_rng_state,
    _load_checkpoint,
    _restore_rng_state,
    _save_checkpoint,
    _set_loader_epoch,
    _update_early_stopping,
    train_epoch,
)
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


class _NonFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class _TinyCachedFeatureModel(torch.nn.Module):
    def __init__(self, failure: str | None = None):
        super().__init__()
        self.backbone = torch.nn.Identity()
        self.readout = torch.nn.Linear(3, 45)
        self.failure = failure

    def forward_from_features(
        self,
        features,
        graph_batch,
        *,
        target,
        return_scale,
    ):
        del graph_batch, return_scale
        mean = self.readout(features)
        loss = torch.nn.functional.mse_loss(mean, target)
        if self.failure == "loss":
            loss = loss * loss.new_tensor(float("nan"))
        elif self.failure == "gradient":
            loss = _NonFiniteGradient.apply(loss)
        return {
            "mu": mean,
            "loss": loss,
            "components": {
                "loss_fit": loss.detach() * 0.75,
                "loss_uncertainty": loss.detach() * 0.25,
            },
        }


def _cached_feature_batches():
    return [
        {
            "features": torch.randn(4, 3),
            "target": torch.randn(4, 45),
        },
        {
            "features": torch.randn(3, 3),
            "target": torch.randn(3, 45),
        },
    ]


def test_train_epoch_records_components_and_gradient_norms():
    model = _TinyCachedFeatureModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    statistics = train_epoch(
        model,
        _cached_feature_batches(),
        optimizer,
        torch.device("cpu"),
        frozen_backbone=False,
        use_bf16=False,
    )
    assert set(statistics) == {
        "loss",
        "gradient_norm_mean",
        "gradient_norm_max",
        "loss_fit",
        "loss_uncertainty",
    }
    assert all(np.isfinite(value) for value in statistics.values())
    assert statistics["gradient_norm_max"] >= statistics["gradient_norm_mean"]


@pytest.mark.parametrize("failure", ("loss", "gradient"))
def test_train_epoch_fails_fast_on_nonfinite_optimization(failure):
    model = _TinyCachedFeatureModel(failure=failure)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises((FloatingPointError, RuntimeError), match="non-finite"):
        train_epoch(
            model,
            _cached_feature_batches(),
            optimizer,
            torch.device("cpu"),
            frozen_backbone=False,
            use_bf16=False,
        )


def test_early_stopping_tracks_improvement_and_rejects_nan():
    best, stale = float("inf"), 0
    best, stale, improved = _update_early_stopping(3.0, best, stale)
    assert (best, stale, improved) == (3.0, 0, True)
    best, stale, improved = _update_early_stopping(3.1, best, stale)
    assert (best, stale, improved) == (3.0, 1, False)
    best, stale, improved = _update_early_stopping(2.9, best, stale)
    assert (best, stale, improved) == (2.9, 0, True)
    with pytest.raises(FloatingPointError, match="validation criterion"):
        _update_early_stopping(float("nan"), best, stale)


def test_training_sample_order_is_seed_and_epoch_addressable():
    dataset = TensorDataset(torch.arange(32))
    sampler = RandomSampler(dataset, generator=torch.Generator())
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    _set_loader_epoch(loader, seed=42, epoch=7)
    first = torch.cat([batch[0] for batch in loader])
    torch.rand(100)
    _set_loader_epoch(loader, seed=42, epoch=7)
    resumed = torch.cat([batch[0] for batch in loader])
    _set_loader_epoch(loader, seed=42, epoch=8)
    next_epoch = torch.cat([batch[0] for batch in loader])

    torch.testing.assert_close(resumed, first, rtol=0.0, atol=0.0)
    assert not torch.equal(next_epoch, first)


def test_rng_checkpoint_round_trip_is_exact(tmp_path):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = _capture_rng_state()
    checkpoint = tmp_path / "last_state.pt"
    _save_checkpoint({"rng_state": state}, checkpoint)

    expected = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored = _load_checkpoint(checkpoint)
    _restore_rng_state(restored["rng_state"])
    actual = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)
