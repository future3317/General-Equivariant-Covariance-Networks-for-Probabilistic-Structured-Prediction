"""Proper moment and dependence diagnostics for finite predictive ensembles."""

from __future__ import annotations

import math

import torch

from distributions.student_t import student_t_log_prob_from_statistics


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
    student_t_dof: float | torch.Tensor = 5.0,
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
        nu = torch.as_tensor(student_t_dof, dtype=means.dtype, device=means.device)
        if bool((nu <= 0).any()):
            raise ValueError("student_t_dof must be positive")
        if nu.ndim == 0:
            chi = torch.distributions.Chi2(nu).sample((num_samples, N))
            factor = torch.sqrt(nu / chi)
        elif nu.shape == (N,):
            chi = torch.distributions.Chi2(nu).sample((num_samples,))
            factor = torch.sqrt(nu.unsqueeze(0) / chi)
        else:
            raise ValueError("sampling degrees of freedom must be scalar or shape (N,)")
        samples = selected_means + (samples - selected_means) * factor.unsqueeze(-1)
    elif distribution != "gaussian":
        raise ValueError("distribution must be gaussian or student_t")
    return samples


def energy_score_from_samples(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_chunk_size: int = 4,
    observation_chunk_size: int = 256,
) -> torch.Tensor:
    """Exact Energy Score from predictive samples ``(S,N,d)``.

    The all-pairs term is evaluated in bounded blocks. This changes only the
    reduction schedule, not the all-pairs Monte Carlo estimator.
    """
    if samples.ndim != 3 or target.shape != samples.shape[1:]:
        raise ValueError("samples must be (S,N,d) and target must be (N,d)")
    if sample_chunk_size < 1 or observation_chunk_size < 1:
        raise ValueError("Energy Score chunk sizes must be positive")
    first = torch.linalg.vector_norm(samples - target.unsqueeze(0), dim=-1).mean(0)
    num_samples, num_observations, _ = samples.shape
    pairwise_parts = []
    for observation_start in range(0, num_observations, observation_chunk_size):
        observation_stop = min(
            observation_start + observation_chunk_size, num_observations
        )
        observation_samples = samples[:, observation_start:observation_stop]
        pairwise_sum = samples.new_zeros(observation_stop - observation_start)
        for sample_start in range(0, num_samples, sample_chunk_size):
            sample_stop = min(sample_start + sample_chunk_size, num_samples)
            distances = torch.linalg.vector_norm(
                observation_samples[sample_start:sample_stop, None]
                - observation_samples[None, :],
                dim=-1,
            )
            pairwise_sum = pairwise_sum + distances.sum((0, 1))
        pairwise_parts.append(pairwise_sum / (num_samples * num_samples))
    pairwise = torch.cat(pairwise_parts)
    return (first - 0.5 * pairwise).mean()


def finite_mixture_nll(
    means: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
    *,
    distribution: str = "gaussian",
    student_t_dof: float | torch.Tensor = 5.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact finite-mixture NLL without moment matching.

    ``weights`` may be shared ``(M,)`` weights or sample-conditional ``(M,N)``
    invariant weights.  The existing ensemble API delegates here with uniform
    weights, so training and evaluation share one logsumexp implementation.
    """
    if means.ndim != 3 or scales.ndim != 4:
        raise ValueError("expected means (M,N,d) and scales (M,N,d,d)")
    M, N, d = means.shape
    if scales.shape != (M, N, d, d):
        raise ValueError("mixture means/scales have incompatible shapes")
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
        component_log_prob = student_t_log_prob_from_statistics(
            logdet,
            quadratic,
            d,
            student_t_dof,
        )
    else:
        raise ValueError("distribution must be gaussian or student_t")
    if weights is None:
        log_weights = component_log_prob.new_full((M, 1), -math.log(M))
    else:
        weights = weights.to(
            device=component_log_prob.device, dtype=component_log_prob.dtype
        )
        if weights.shape not in {(M,), (M, N)}:
            raise ValueError("weights must have shape (M,) or (M,N)")
        if bool((weights <= 0).any()):
            raise ValueError("mixture weights must be positive")
        weights = weights / weights.sum(dim=0, keepdim=True)
        log_weights = weights.log()
        if log_weights.ndim == 1:
            log_weights = log_weights[:, None]
    return -torch.logsumexp(component_log_prob + log_weights, dim=0).mean()


def ensemble_nll(
    means: torch.Tensor,
    scales: torch.Tensor,
    target: torch.Tensor,
    *,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
) -> torch.Tensor:
    """Exact NLL of an equally weighted member mixture."""
    return finite_mixture_nll(
        means,
        scales,
        target,
        distribution=distribution,
        student_t_dof=student_t_dof,
    )


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
    predicted = (
        torch.abs(samples.unsqueeze(-1) - samples.unsqueeze(-2)).pow(order).mean(0)
    )
    observed = torch.abs(target.unsqueeze(-1) - target.unsqueeze(-2)).pow(order)
    if weights is None:
        weights = torch.ones_like(observed)
    if weights.shape != observed.shape:
        raise ValueError("weights must have shape (N,d,d)")
    upper = torch.triu(torch.ones_like(observed), diagonal=1)
    return (((predicted - observed).square()) * weights * upper).sum() / target.shape[0]
