from __future__ import annotations

import torch

from scripts.analyze_itop_family_tradeoff import (
    aggregate_paired_differences,
    paired_bootstrap_interval,
    pareto_frontier,
    validate_paired_predictions,
)


def test_paired_bootstrap_is_reproducible_and_contains_constant_delta():
    delta = torch.full((200,), -2.5, dtype=torch.float64)
    first = paired_bootstrap_interval(delta, repetitions=200, seed=9)
    second = paired_bootstrap_interval(delta, repetitions=200, seed=9)
    assert first == second
    assert first["mean"] == -2.5
    assert first["lower_95"] == -2.5
    assert first["upper_95"] == -2.5


def test_pair_validation_rejects_target_or_order_mismatch():
    full = {"target": torch.zeros(3, 2), "frame_index": torch.arange(3)}
    graph = {"target": torch.zeros(3, 2), "frame_index": torch.arange(3)}
    validate_paired_predictions(full, graph)

    graph["frame_index"] = torch.tensor([0, 2, 1])
    try:
        validate_paired_predictions(full, graph)
    except ValueError as error:
        assert "ordering" in str(error)
    else:
        raise AssertionError("mismatched ordering was accepted")


def test_pareto_frontier_minimizes_score_and_resource():
    rows = {
        "full": {"nll": 5.0, "memory_mb": 40.0},
        "graph": {"nll": 4.0, "memory_mb": 20.0},
        "small_bad": {"nll": 8.0, "memory_mb": 10.0},
    }
    assert pareto_frontier(rows, x="memory_mb", y="nll") == [
        "graph",
        "small_bad",
    ]


def test_seed_differences_are_averaged_per_sample_before_bootstrap():
    result = aggregate_paired_differences(
        [torch.ones(5), torch.full((5,), 3.0)]
    )
    assert torch.equal(result, torch.full((5,), 2.0, dtype=torch.float64))
