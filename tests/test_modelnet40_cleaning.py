import numpy as np

from scripts.clean_modelnet40_cache import (
    centered_max_radius,
    clean_cache_payload,
    fit_log_radius_threshold,
)


def _cloud(radius: float) -> np.ndarray:
    return np.array(
        [[radius, 0.0, 0.0], [-radius, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )


def _split(radii: list[float], offset: float = 0.0) -> dict[str, np.ndarray]:
    count = len(radii)
    inertia = np.arange(count * 6, dtype=np.float32).reshape(count, 6) + offset
    return {
        "points": np.stack([_cloud(radius) for radius in radii]),
        "inertia": inertia,
        "labels": np.arange(count, dtype=np.int64),
    }


def test_centered_radius_is_translation_invariant():
    points = _cloud(2.0)[None]
    translated = points + np.array([[[10.0, -4.0, 3.0]]], dtype=np.float32)
    np.testing.assert_allclose(centered_max_radius(points), [2.0])
    np.testing.assert_allclose(centered_max_radius(translated), [2.0])


def test_robust_rule_separates_corrupted_scale():
    normal = [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4]
    points = np.stack([_cloud(radius) for radius in normal + [50.0]])
    threshold, _ = fit_log_radius_threshold(points, robust_z=8.0)
    assert max(normal) < threshold < 50.0


def test_cleaning_filters_both_splits_and_fits_train_only_statistics():
    payload = {
        "train": _split([0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 50.0]),
        "test": _split([1.0, 60.0], offset=1000.0),
        "stats": {"sentinel": np.array(-1.0)},
    }
    cleaned, audit = clean_cache_payload(payload, robust_z=8.0)
    assert len(cleaned["train"]["points"]) == 8
    assert len(cleaned["test"]["points"]) == 1
    assert audit["selection_uses_targets"] is False
    np.testing.assert_allclose(
        cleaned["stats"]["mean"], cleaned["train"]["inertia"].mean(axis=0)
    )
    assert not np.allclose(
        cleaned["stats"]["mean"], cleaned["test"]["inertia"].mean(axis=0)
    )
