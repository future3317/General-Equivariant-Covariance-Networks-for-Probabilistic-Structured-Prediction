import numpy as np
import torch

from scripts.audit_dielectric_paired_bootstrap import (
    cluster_bootstrap_mean_interval,
    paired_difference,
)


def _prediction(target: list[list[float]], scale: float = 1.0) -> dict[str, torch.Tensor]:
    target_tensor = torch.tensor(target, dtype=torch.float64)
    n = target_tensor.shape[0]
    return {
        "sample_id": torch.arange(n),
        "target": target_tensor,
        "mean": torch.zeros_like(target_tensor),
        "scale": torch.eye(6, dtype=torch.float64).expand(n, -1, -1) * scale,
    }


def test_paired_difference_uses_full_normalized_laws() -> None:
    left = _prediction([[1.0] + [0.0] * 5, [0.0] * 6])
    right = _prediction([[1.0] + [0.0] * 5, [0.0] * 6], scale=2.0)

    values = paired_difference(
        left, right, left_law="gaussian", right_law="student_t"
    )

    assert values.shape == (2,)
    assert np.isfinite(values).all()
    assert not np.allclose(values, 0.0)


def test_cluster_bootstrap_is_reproducible_and_finite() -> None:
    values = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    first = cluster_bootstrap_mean_interval(values, seed=7, samples=200)
    second = cluster_bootstrap_mean_interval(values, seed=7, samples=200)

    assert first == second
    assert first[0] <= values.mean() <= first[1]

    rng = np.random.default_rng(7)
    indices = rng.integers(0, values.shape[1], size=(200, values.shape[1]))
    expected_draws = values[:, indices].mean(axis=(0, 2))
    expected = (
        float(np.quantile(expected_draws, 0.025)),
        float(np.quantile(expected_draws, 0.975)),
    )
    assert np.allclose(first, expected)
