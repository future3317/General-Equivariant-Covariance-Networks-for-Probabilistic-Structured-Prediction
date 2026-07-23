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


def apply_block_temperature(
    scale: Tensor, block_ids: Tensor, temperatures: Tensor | list[float]
) -> Tensor:
    """Apply positive isotypic-block temperatures to an SPD scale.

    If ``F`` is diagonal and constant on every irreducible ``(l, parity)``
    block, ``F S F`` commutes with the representation action.  Consequently
    this calibration changes uncertainty sharpness without breaking the
    compiler's equivariance contract.
    """
    if scale.ndim != 3 or scale.shape[-1] != scale.shape[-2]:
        raise ValueError("scale must have shape (N, d, d)")
    block_ids = torch.as_tensor(block_ids, dtype=torch.long, device=scale.device)
    temperatures = torch.as_tensor(
        temperatures, dtype=scale.dtype, device=scale.device
    )
    if block_ids.ndim != 1 or block_ids.numel() != scale.shape[-1]:
        raise ValueError("block_ids must have one entry per output coordinate")
    if temperatures.ndim != 1 or temperatures.numel() == 0:
        raise ValueError("temperatures must be a non-empty vector")
    if bool((block_ids < 0).any()) or int(block_ids.max()) >= temperatures.numel():
        raise ValueError("block_ids contain an invalid block index")
    if bool((temperatures <= 0).any()) or not bool(torch.isfinite(temperatures).all()):
        raise ValueError("temperatures must be finite and positive")
    factors = torch.sqrt(temperatures[block_ids])
    return factors.view(1, -1, 1) * scale * factors.view(1, 1, -1)


def fit_block_temperatures(
    pred: Tensor,
    target: Tensor,
    scale: Tensor,
    block_ids: Tensor,
    *,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
    min_temperature: float = 1e-3,
    max_temperature: float = 1e3,
    steps: int = 80,
) -> list[float]:
    """Fit validation-only positive temperatures, one per isotypic block."""
    if not (0.0 < min_temperature < max_temperature):
        raise ValueError("invalid temperature bounds")
    pred = pred.detach().clone()
    target = target.detach().clone()
    scale = scale.detach().clone()
    block_ids = torch.as_tensor(block_ids, dtype=torch.long, device=scale.device)
    num_blocks = int(block_ids.max().item()) + 1
    log_t = torch.zeros(num_blocks, dtype=scale.dtype, device=scale.device)
    log_t.requires_grad_()
    optimizer = torch.optim.LBFGS(
        [log_t], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperatures = log_t.exp().clamp(min_temperature, max_temperature)
        calibrated = apply_block_temperature(scale, block_ids, temperatures)
        loss = scale_nll(
            pred,
            target,
            calibrated,
            distribution=distribution,
            student_t_dof=student_t_dof,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.detach().exp().clamp(min_temperature, max_temperature).tolist()
