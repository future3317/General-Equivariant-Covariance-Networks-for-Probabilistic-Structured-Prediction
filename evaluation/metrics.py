"""Evaluation metrics for probabilistic structured prediction.

All functions operate on torch tensors and assume predictions and targets are
in the same output representation space. Covariance/scale matrices are assumed
SPD and are passed as the ``scale`` argument.
"""

from __future__ import annotations

import torch
from torch import Tensor


def mean_absolute_error(pred: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error, averaged over batch and feature dimensions."""
    return torch.mean(torch.abs(pred - target))


def root_mean_squared_error(pred: Tensor, target: Tensor) -> Tensor:
    """Root mean squared error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def r2_score(pred: Tensor, target: Tensor, dim: int = 0) -> Tensor:
    """Coefficient of determination :math:`R^2` along a dimension.

    Args:
        pred: Predictions of shape ``(..., d)``.
        target: Targets of the same shape.
        dim: Dimension along which to compute per-feature variance.

    Returns:
        Per-feature :math:`R^2` of shape ``(d,)``.
    """
    ss_res = torch.sum((target - pred) ** 2, dim=dim)
    ss_tot = torch.sum((target - target.mean(dim=dim, keepdim=True)) ** 2, dim=dim)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mean_r2_score(pred: Tensor, target: Tensor) -> Tensor:
    """Mean :math:`R^2` over the last (feature) dimension."""
    return r2_score(pred, target, dim=0).mean()


def mahalanobis_distance_squared(
    residual: Tensor,
    scale: Tensor,
) -> Tensor:
    """Squared Mahalanobis distance :math:`r^T S^{-1} r`.

    Args:
        residual: ``(..., d)``.
        scale: SPD matrices ``(..., d, d)``.

    Returns:
        Squared distances ``(...)``.
    """
    # Solve S x = r, then compute r^T x.
    x = torch.linalg.solve(scale, residual.unsqueeze(-1))
    return torch.sum(residual * x.squeeze(-1), dim=-1)


def empirical_coverage(
    pred: Tensor,
    target: Tensor,
    scale: Tensor,
    levels: list[float] | None = None,
    reference: str = "gaussian",
    student_t_dof: float = 5.0,
) -> dict[str, float]:
    """Empirical coverage of confidence ellipsoids at specified levels.

    Args:
        pred: Mean predictions ``(N, d)``.
        target: Targets ``(N, d)``.
        scale: SPD scale matrices ``(N, d, d)``.
        levels: List of confidence levels. Defaults to ``[0.5, 0.8, 0.9, 0.95]``.

    Returns:
        Dictionary mapping ``coverage_XX`` to empirical coverage fractions.
    """
    if levels is None:
        levels = [0.5, 0.8, 0.9, 0.95]

    from scipy.stats import chi2, f

    if reference not in {"gaussian", "student_t"}:
        raise ValueError(f"unknown calibration reference: {reference}")
    if reference == "student_t" and student_t_dof <= 0:
        raise ValueError("student_t_dof must be positive")

    d = pred.shape[-1]
    residual = target - pred
    maha2 = mahalanobis_distance_squared(residual, scale)

    result = {}
    for level in levels:
        threshold = (
            chi2.ppf(level, df=float(d))
            if reference == "gaussian"
            else float(d) * f.ppf(level, dfn=float(d), dfd=float(student_t_dof))
        )
        result[f"coverage_{int(level * 100):02d}"] = (
            (maha2 < threshold).float().mean().item()
        )
    return result


def negative_log_likelihood_gaussian(
    pred: Tensor,
    target: Tensor,
    scale: Tensor,
) -> Tensor:
    """Gaussian negative log-likelihood for a batch.

    Args:
        pred: Mean predictions ``(N, d)``.
        target: Targets ``(N, d)``.
        scale: SPD covariance matrices ``(N, d, d)``.

    Returns:
        Mean NLL over the batch.
    """
    d = pred.shape[-1]
    residual = target - pred
    maha2 = mahalanobis_distance_squared(residual, scale)
    logdet = torch.logdet(scale)
    import math

    nll = 0.5 * d * math.log(2.0 * math.pi) + 0.5 * logdet + 0.5 * maha2
    return nll.mean()


def energy_score(
    pred: Tensor,
    scale: Tensor,
    target: Tensor,
    num_samples: int = 50,
) -> Tensor:
    """Energy Score for multivariate Gaussian predictions.

    Samples from :math:`N(\\mu, S)` and approximates:

    .. math::

        ES = \\mathbb E_{Y \\sim p} \\|Y - y_{\\text{true}}\\|
             - \\frac12 \\mathbb E_{Y, Y' \\sim p} \\|Y - Y'\\|.

    Args:
        pred: Mean predictions ``(N, d)``.
        scale: SPD covariance matrices ``(N, d, d)``.
        target: Targets ``(N, d)``.
        num_samples: Number of samples drawn per prediction.

    Returns:
        Mean Energy Score over the batch.
    """
    *batch, d = pred.shape
    pred_flat = pred.reshape(-1, d)
    scale_flat = scale.reshape(-1, d, d)
    target_flat = target.reshape(-1, d)

    # S = L L^T
    L = torch.linalg.cholesky(scale_flat)
    eps = torch.randn(
        pred_flat.shape[0], num_samples, d, device=pred.device, dtype=pred.dtype
    )
    samples = pred_flat.unsqueeze(1) + torch.einsum("bij,bnj->bni", L, eps)

    # E ||Y - y_true||
    term1 = torch.norm(samples - target_flat.unsqueeze(1), dim=-1).mean(dim=1)

    # E ||Y - Y'||
    term2 = torch.norm(samples.unsqueeze(2) - samples.unsqueeze(1), dim=-1).mean(
        dim=(1, 2)
    )

    score = term1 - 0.5 * term2
    return score.mean()


def covariance_relative_error(pred_scale: Tensor, true_scale: Tensor) -> Tensor:
    """Relative Frobenius error between predicted and true covariance matrices."""
    diff_norm = torch.norm(pred_scale - true_scale, dim=(-2, -1))
    true_norm = torch.norm(true_scale, dim=(-2, -1))
    return (diff_norm / (true_norm + 1e-12)).mean()


def log_euclidean_error(pred_A: Tensor, true_A: Tensor) -> Tensor:
    """Frobenius norm error in the log-domain parameterization."""
    return torch.norm(pred_A - true_A, dim=(-2, -1)).mean()


def eigenvalue_error(pred_scale: Tensor, true_scale: Tensor) -> Tensor:
    """Mean absolute eigenvalue error."""
    pred_eig = torch.linalg.eigvalsh(pred_scale)
    true_eig = torch.linalg.eigvalsh(true_scale)
    return torch.mean(torch.abs(pred_eig - true_eig))


def whitened_residual_covariance(
    pred: Tensor,
    target: Tensor,
    scale: Tensor,
) -> Tensor:
    """Covariance of whitened residuals.

    Computes :math:`z = L^{-1} r` where :math:`S = L L^\\top`, then returns
    the trace of the empirical covariance of :math:`z`. For a well-calibrated
    Gaussian, this should be close to the dimension of the output space.
    """
    residual = target - pred
    L = torch.linalg.cholesky(scale)
    z = torch.linalg.solve_triangular(L, residual.unsqueeze(-1), upper=False).squeeze(
        -1
    )
    whitened = torch.matmul(z.unsqueeze(-1), z.unsqueeze(-2))
    return torch.trace(whitened.mean(dim=0))


def covariance_spectrum_diagnostics(
    scale: Tensor,
    *,
    log_variance_bounds: tuple[float, float] | None = None,
    boundary_fraction: float = 0.01,
) -> dict[str, float]:
    """Summarize predictive covariance spectra without changing the model.

    When a spectral window is declared, ``lower_boundary_fraction`` and
    ``upper_boundary_fraction`` measure the fraction of covariance
    eigenvalues in the outer ``boundary_fraction`` of the *log-variance*
    interval.  They are saturation diagnostics, not bound violations.
    """
    if scale.ndim != 3 or scale.shape[-1] != scale.shape[-2]:
        raise ValueError("scale must have shape (N, d, d)")
    if not 0.0 < boundary_fraction < 0.5:
        raise ValueError("boundary_fraction must lie in (0, 0.5)")

    log_eigenvalues = torch.log(torch.linalg.eigvalsh(scale))
    condition_numbers = torch.exp(
        log_eigenvalues[..., -1] - log_eigenvalues[..., 0]
    )
    diagnostics = {
        "log_eigenvalue_min": float(log_eigenvalues.min().item()),
        "log_eigenvalue_max": float(log_eigenvalues.max().item()),
        "log_eigenvalue_mean": float(log_eigenvalues.mean().item()),
        "condition_number_mean": float(condition_numbers.mean().item()),
        "condition_number_max": float(condition_numbers.max().item()),
    }
    if log_variance_bounds is None:
        return diagnostics

    lower, upper = log_variance_bounds
    if not lower < upper:
        raise ValueError("log_variance_bounds must satisfy lower < upper")
    width = upper - lower
    normalized = (log_eigenvalues - lower) / width
    tolerance = 1e-5
    diagnostics.update(
        {
            "declared_log_variance_min": float(lower),
            "declared_log_variance_max": float(upper),
            "declared_max_condition_number": float(torch.exp(torch.tensor(width)).item()),
            "lower_boundary_fraction": float(
                (normalized <= boundary_fraction).float().mean().item()
            ),
            "upper_boundary_fraction": float(
                (normalized >= 1.0 - boundary_fraction).float().mean().item()
            ),
            "bound_violation_fraction": float(
                ((normalized < -tolerance) | (normalized > 1.0 + tolerance))
                .float()
                .mean()
                .item()
            ),
        }
    )
    return diagnostics
