import pytest
import torch

from scripts.evaluate_itop_conformal import _split_indices, _validate_view_pair


def test_posthoc_split_is_deterministic_and_disjoint():
    frame_index = torch.arange(10)
    calibration, evaluation = _split_indices(
        frame_index, calibration_fraction=0.4, seed=7
    )
    calibration_again, evaluation_again = _split_indices(
        frame_index, calibration_fraction=0.4, seed=7
    )
    torch.testing.assert_close(calibration, calibration_again)
    torch.testing.assert_close(evaluation, evaluation_again)
    assert set(calibration.tolist()).isdisjoint(evaluation.tolist())
    assert sorted(calibration.tolist() + evaluation.tolist()) == list(range(10))


def test_posthoc_split_rejects_duplicate_or_invalid_inputs():
    with pytest.raises(ValueError, match="unique"):
        _split_indices(torch.tensor([1, 1, 2]), calibration_fraction=0.5, seed=0)
    with pytest.raises(ValueError, match="calibration_fraction"):
        _split_indices(torch.arange(3), calibration_fraction=0.0, seed=0)


def test_view_pair_requires_paired_side_and_top_artifacts():
    side = {
        "frame_index": torch.tensor([3, 8]),
        "view_id": torch.zeros(2, dtype=torch.long),
    }
    top = {
        "frame_index": torch.tensor([3, 8]),
        "view_id": torch.ones(2, dtype=torch.long),
    }
    contract = _validate_view_pair(side, top)
    assert contract["paired_frame_count"] == 2
    top["frame_index"] = torch.tensor([8, 3])
    with pytest.raises(ValueError, match="same frame order"):
        _validate_view_pair(side, top)
