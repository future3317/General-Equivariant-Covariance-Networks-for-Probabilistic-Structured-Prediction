import numpy as np
import pytest

from scripts.review_evidence_audit import (
    bootstrap_mean_interval,
    paired_difference,
    validate_prediction_pair,
)


def test_paired_difference_preserves_frame_alignment():
    left = {
        "frame_index": np.array([3, 1]),
        "mean": np.zeros((2, 45)),
        "target": np.zeros((2, 45)),
    }
    right = {
        "frame_index": np.array([3, 1]),
        "mean": np.zeros((2, 45)),
        "target": np.zeros((2, 45)),
    }
    diff = paired_difference(left, right)
    assert diff.shape == (2,)
    np.testing.assert_allclose(diff, [0.0, 0.0])


def test_validate_prediction_pair_rejects_mismatched_targets():
    left = {"frame_index": np.array([0]), "target": np.zeros((1, 4))}
    right = {"frame_index": np.array([0]), "target": np.ones((1, 4))}
    with pytest.raises(ValueError, match="target"):
        validate_prediction_pair(left, right)


def test_bootstrap_mean_interval_is_deterministic_and_contains_mean():
    values = np.array([-2.0, 0.0, 2.0])
    interval = bootstrap_mean_interval(
        values, seed=7, samples=2000, confidence=0.95
    )
    assert interval[0] <= values.mean() <= interval[1]
    assert interval == bootstrap_mean_interval(
        values, seed=7, samples=2000, confidence=0.95
    )


def test_fixed_nu_grid_rejects_nonpositive_dof():
    from scripts.audit_dielectric_fixed_nu_sensitivity import validate_nu_grid

    with pytest.raises(ValueError, match="nu"):
        validate_nu_grid([0.0, 5.0])
