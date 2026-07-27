"""Proper moment and dependence diagnostics for finite predictive ensembles."""

from __future__ import annotations

import math

import torch


def combine_ensemble_moments(
    means: torch.Tensor,
    scales: torch.Tensor,
    *,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Apply the law of total covariance to ensemble members.

    ``means`` has shape ``(M, N, d)`` and ``scales`` has shape
    ``(M, N, d, d)``.  The returned total covariance is a moment object; it
    must not be mislabeled as a single Student-t density.  Proper mixture NLL
    is provided separately by :func:`ensemble_nll`.
    """
    if means.ndim != 3 or scales.ndim != 4:
        raise ValueError("expected means (M,N,d) and scales (M,N,d,d)")
    if means.shape[:2] != scales.shape[:2] or means.shape[-1] != scales.shape[-1]:
        raise ValueError("ensemble means/scales have incompatible shapes")
    if distribution not in {"gaussian", "student_t"}:
        raise ValueError("distribution must be gaussian or student_t")
    if distribution == "student_t":
        if student_t_dof <= 2:
            raise ValueError("Student-t moment covariance requires nu > 2")
        factor = float(student_t_dof) / (float(student_t_dof) - 2.0)
    else:
        factor = 1.0
    mean = means.mean(dim=0)
    aleatoric = scales.mean(dim=0) * factor
    centered = means - mean.unsqueeze(0)
    epistemic = torch.einsum("mni,mnj->nij", centered, centered) / means.shape[0]
    total = aleatoric + epistemic
    return {
        "mean": mean,
        "aleatoric_covariance": aleatoric,
        "epistemic_covariance": epistemic,
        "total_covariance": total,
    }


def sample_ensemble(
    means: torch.Tensor,
    scales: torch.Tensor,
    *,
    num_samples: int = 128,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
) -> torch.Tensor:
    """Draw samples from the equally weighted predictive mixture."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    M, N, d = means.shape
    indices = torch.randint(M, (num_samples,), device=means.device)
    selected_means = means[indices]
    selected_scales = scales[indices]
    chol = torch.linalg.cholesky(selected_scales)
    noise = torch.randn(num_samples, N, d, device=means.device, dtype=means.dtype)
    samples = selected_means + torch.einsum("snij,snj->sni", chol, noise)
    if distribution == "student_t":
        if student_t_dof <= 0:
            raise ValueError("student_t_dof must be positive")
        chi = torch.distributions.Chi2(
            torch.as_tensor(student_t_dof, dtype=means.dtype, device=means.device)
        ).sample((num_samples, N))
        samples = selected_means + (samples - selected_means) * torch.sqrt(
            float(student_t_dof) / chi
        ).unsqueeze(-1)
    elif distribution != "gaussian":
        raise ValueError("distribution must be gaussian or student_t")
    return samples


def ensemble_nll(
    means: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
    *,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
) -> torch.Tensor:
    """Exact negative log likelihood of the equally weighted member mixture."""
    M, N, d = means.shape
    if target.shape != (N, d):
        raise ValueError("target must have shape (N,d)")
    residual = target.unsqueeze(0) - means
    chol = torch.linalg.cholesky(scales)
    solved = torch.cholesky_solve(residual.unsqueeze(-1), chol).squeeze(-1)
    quadratic = (residual * solved).sum(-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    if distribution == "gaussian":
        component_log_prob = -0.5 * (d * math.log(2.0 * math.pi) + logdet + quadratic)
    elif distribution == "student_t":
        nu = float(student_t_dof)
        if nu <= 0:
            raise ValueError("student_t_dof must be positive")
        constant = (
            math.lgamma((nu + d) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * d * math.log(nu * math.pi)
        )
        component_log_prob = constant - 0.5 * logdet - 0.5 * (nu + d) * torch.log1p(
            quadratic / nu
        )
    else:
        raise ValueError("distribution must be gaussian or student_t")
    return -torch.logsumexp(component_log_prob, dim=0).mean() + math.log(M)


def variogram_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    order: float = 0.5,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Monte-Carlo variogram score, sensitive to predictive dependence."""
    if samples.ndim != 3 or target.ndim != 2 or samples.shape[1:] != target.shape:
        raise ValueError("samples must be (S,N,d) and target must be (N,d)")
    if order <= 0:
        raise ValueError("order must be positive")
    predicted = torch.abs(samples.unsqueeze(-1) - samples.unsqueeze(-2)).pow(order).mean(0)
    observed = torch.abs(target.unsqueeze(-1) - target.unsqueeze(-2)).pow(order)
    if weights is None:
        weights = torch.ones_like(observed)
    if weights.shape != observed.shape:
        raise ValueError("weights must have shape (N,d,d)")
    upper = torch.triu(torch.ones_like(observed), diagonal=1)
    return (((predicted - observed).square()) * weights * upper).sum() / target.shape[0]
