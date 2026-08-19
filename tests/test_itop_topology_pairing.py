import numpy as np
import torch

from scripts.audit_itop_topology_pairing import cluster_bootstrap_mean_interval
from distributions.student_t import student_t_log_prob_from_statistics
from scripts.review_evidence_audit import _frame_nll


def test_frame_nll_uses_student_t_scatter_not_covariance_logdet() -> None:
    dimension = 45
    scatter_logdet = torch.tensor([0.0], dtype=torch.float64)
    covariance_logdet = scatter_logdet + dimension * np.log(5.0 / 3.0)
    prediction = {
        "frame_uncertainty": covariance_logdet,
        "frame_mahalanobis2": torch.tensor([2.0], dtype=torch.float64),
    }
    expected = (-student_t_log_prob_from_statistics(
        scatter_logdet, prediction["frame_mahalanobis2"], dimension, 5.0
    )).numpy()
    np.testing.assert_allclose(_frame_nll(prediction), expected)


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
