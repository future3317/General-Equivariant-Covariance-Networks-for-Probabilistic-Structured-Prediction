"""Calibration diagnostics for probabilistic predictions."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2


def mahalanobis_distances(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> np.ndarray:
    """Compute squared Mahalanobis distances for a batch.

    Args:
        pred: Mean predictions ``(N, d)``.
        target: Targets ``(N, d)``.
        scale: SPD scale matrices ``(N, d, d)``.

    Returns:
        Squared Mahalanobis distances as a NumPy array of length ``N``.
    """
    residual = target - pred
    x = torch.linalg.solve(scale, residual.unsqueeze(-1)).squeeze(-1)
    maha2 = torch.sum(residual * x, dim=-1)
    return maha2.cpu().numpy()


def calibration_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    confidence_levels: list[float] | None = None,
) -> dict[str, float]:
    """Compute expected calibration error over confidence levels.

    Args:
        pred: Mean predictions ``(N, d)``.
        target: Targets ``(N, d)``.
        scale: SPD scale matrices ``(N, d, d)``.
        confidence_levels: Confidence levels to evaluate. Defaults to
            ``[0.1, 0.2, ..., 0.9]``.

    Returns:
        Dictionary with ``ece`` (expected calibration error) and
        ``ace`` (absolute calibration error) in percentage points.
    """
    if confidence_levels is None:
        confidence_levels = [0.1 * i for i in range(1, 10)]

    d = pred.shape[-1]
    maha2 = mahalanobis_distances(pred, target, scale)

    observed = []
    for level in confidence_levels:
        threshold = chi2.ppf(level, df=float(d))
        observed.append(float(np.mean(maha2 < threshold)))

    observed = np.array(observed)
    expected = np.array(confidence_levels)

    ece = float(np.mean(observed - expected))
    ace = float(np.mean(np.abs(observed - expected)))

    return {
        "ece": ece,
        "ace": ace,
        "confidence_levels": confidence_levels,
        "observed_coverages": observed.tolist(),
    }


def qq_data(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    num_quantiles: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return theoretical and empirical quantiles for a Q-Q calibration plot.

    Args:
        pred: Mean predictions ``(N, d)``.
        target: Targets ``(N, d)``.
        scale: SPD scale matrices ``(N, d, d)``.
        num_quantiles: Number of quantiles to return.

    Returns:
        ``(theoretical_quantiles, empirical_quantiles)``.
    """
    d = pred.shape[-1]
    maha2 = mahalanobis_distances(pred, target, scale)
    empirical = np.sort(maha2)

    n = len(empirical)
    probabilities = np.linspace(0.5 / n, 1 - 0.5 / n, min(num_quantiles, n))
    theoretical = chi2.ppf(probabilities, df=float(d))
    empirical_quantiles = np.quantile(empirical, probabilities)

    return theoretical, empirical_quantiles


def sharpness(scale: torch.Tensor) -> dict[str, float]:
    """Sharpness of predictive distributions.

    Returns the mean log determinant and mean trace of the scale matrices.
    Lower values indicate sharper (more confident) predictions.
    """
    return {
        "mean_logdet": float(torch.logdet(scale).mean().item()),
        "mean_trace": float(
            torch.diagonal(scale, dim1=-2, dim2=-1).sum(-1).mean().item()
        ),
    }
