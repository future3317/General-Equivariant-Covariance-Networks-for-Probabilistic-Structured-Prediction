"""Validation-only scalar temperature calibration for predictive scales."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .metrics import mahalanobis_distance_squared


def scale_nll(
    pred: Tensor,
    target: Tensor,
    scale: Tensor,
    *,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
) -> Tensor:
    """Mean negative log likelihood for a covariance/scale matrix."""
    if distribution not in {"gaussian", "student_t"}:
        raise ValueError(f"unknown distribution: {distribution}")
    if distribution == "student_t" and student_t_dof <= 0:
        raise ValueError("student_t_dof must be positive")
    d = pred.shape[-1]
    q = mahalanobis_distance_squared(target - pred, scale)
    logdet = torch.linalg.slogdet(scale).logabsdet
    if distribution == "gaussian":
        nll = 0.5 * (d * math.log(2.0 * math.pi) + logdet + q)
    else:
        nu = float(student_t_dof)
        nll = (
            -math.lgamma((nu + d) / 2.0)
            + math.lgamma(nu / 2.0)
            + 0.5 * (d * math.log(nu * math.pi) + logdet)
            + 0.5 * (nu + d) * torch.log1p(q / nu)
        )
    return nll.mean()


def fit_temperature(
    pred: Tensor,
    target: Tensor,
    scale: Tensor,
    *,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
    min_temperature: float = 1e-3,
    max_temperature: float = 1e3,
    steps: int = 80,
) -> float:
    """Fit a positive scalar ``T`` on validation NLL with ``S' = T S``."""
    if not (0.0 < min_temperature < max_temperature):
        raise ValueError("invalid temperature bounds")
    # Prediction collection commonly runs under ``inference_mode``.  Clone
    # here so LBFGS can legally save tensors for its backward pass.
    pred = pred.detach().clone()
    target = target.detach().clone()
    scale = scale.detach().clone()
    # Optimize in log space; the clamp makes the fitted value deterministic and
    # prevents pathological extrapolation on tiny validation sets.
    log_t = torch.zeros((), dtype=scale.dtype, device=scale.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_t], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        t = log_t.exp().clamp(min_temperature, max_temperature)
        loss = scale_nll(
            pred, target, t * scale,
            distribution=distribution, student_t_dof=student_t_dof,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(min_temperature, max_temperature).item())


def apply_temperature(scale: Tensor, temperature: float) -> Tensor:
    """Return a scaled SPD matrix without mutating the input."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return scale * float(temperature)
