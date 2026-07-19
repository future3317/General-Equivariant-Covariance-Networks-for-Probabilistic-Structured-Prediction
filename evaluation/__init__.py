"""Evaluation metrics, calibration diagnostics, and equivariance validators."""

from evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_r2_score,
    mahalanobis_distance_squared,
    empirical_coverage,
    negative_log_likelihood_gaussian,
    energy_score,
    covariance_relative_error,
    log_euclidean_error,
    eigenvalue_error,
    whitened_residual_covariance,
)
from evaluation.calibration import (
    mahalanobis_distances,
    calibration_error,
    qq_data,
    sharpness,
)
from evaluation.equivariance import (
    mean_equivariance_error,
    scale_equivariance_error,
    average_equivariance_error,
)

__all__ = [
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "mean_r2_score",
    "mahalanobis_distance_squared",
    "empirical_coverage",
    "negative_log_likelihood_gaussian",
    "energy_score",
    "covariance_relative_error",
    "log_euclidean_error",
    "eigenvalue_error",
    "whitened_residual_covariance",
    "mahalanobis_distances",
    "calibration_error",
    "qq_data",
    "sharpness",
    "mean_equivariance_error",
    "scale_equivariance_error",
    "average_equivariance_error",
]
