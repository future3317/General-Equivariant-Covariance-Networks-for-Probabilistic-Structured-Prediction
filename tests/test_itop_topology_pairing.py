import numpy as np

from scripts.audit_itop_topology_pairing import cluster_bootstrap_mean_interval


def test_subject_cluster_bootstrap_resamples_whole_clusters() -> None:
    values_by_seed = np.array(
        [
            [1.0, 1.0, 10.0, 10.0],
            [3.0, 3.0, 14.0, 14.0],
        ]
    )
    cluster_ids = np.array(["subject_a", "subject_a", "subject_b", "subject_b"])

    interval = cluster_bootstrap_mean_interval(
        values_by_seed,
        cluster_ids,
        seed=7,
        samples=200,
    )

    rng = np.random.default_rng(7)
    cluster_indices = rng.integers(0, 2, size=(200, 2))
    cluster_values = np.array([2.0, 12.0])
    expected_draws = cluster_values[cluster_indices].mean(axis=1)
    expected = (
        float(np.quantile(expected_draws, 0.025)),
        float(np.quantile(expected_draws, 0.975)),
    )

    assert interval == expected
