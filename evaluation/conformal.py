"""Split-conformal ellipsoidal regions for compiler-produced SPD shapes.

This module is deliberately outside the compiler.  A ``shape`` matrix is
used only to rank calibration residuals; the resulting region is not a
Gaussian covariance or a Student-t scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _validate_inputs(
    means: torch.Tensor,
    shapes: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int]:
    if means.ndim != 2 or targets.ndim != 2 or shapes.ndim != 3:
        raise ValueError("expected means/targets (N,d) and shapes (N,d,d)")
    if means.shape != targets.shape:
        raise ValueError("means and targets must have the same shape")
    n, d = means.shape
    if n < 1 or shapes.shape != (n, d, d):
        raise ValueError("shapes must have shape (N,d,d) matching means")
    if not torch.isfinite(means).all() or not torch.isfinite(targets).all():
        raise ValueError("means and targets must be finite")
    if not torch.isfinite(shapes).all():
        raise ValueError("shapes must be finite")
    if not torch.allclose(
        shapes, shapes.transpose(-1, -2), rtol=1e-5, atol=1e-7
    ):
        raise ValueError("shapes must be symmetric")
    try:
        torch.linalg.cholesky(shapes)
    except RuntimeError as exc:
        raise ValueError("shapes must be positive definite") from exc
    return n, d


def _squared_shape_scores(
    means: torch.Tensor,
    shapes: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    residual = (targets - means).unsqueeze(-1)
    chol = torch.linalg.cholesky(shapes)
    whitened = torch.linalg.solve_triangular(chol, residual, upper=False)
    return whitened.square().sum(dim=(-2, -1))


@dataclass(frozen=True)
class SplitConformalRegion:
    """Fitted split-conformal ellipsoidal region.

    ``threshold`` is a squared radius applied to a compiler-produced SPD
    shape.  The region has finite-sample marginal coverage under exchangeable
    calibration/test examples; it does not provide conditional coverage.
    """

    threshold: float
    alpha: float
    calibration_size: int
    rank: int

    def contains(
        self,
        means: torch.Tensor,
        shapes: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return one membership flag per target example."""
        _validate_inputs(means, shapes, targets)
        scores = _squared_shape_scores(means, shapes, targets)
        return scores <= self.threshold

    def log_volume(self, shapes: torch.Tensor) -> torch.Tensor:
        """Return log-volume of each region for the supplied SPD shapes."""
        if shapes.ndim != 3 or shapes.shape[-1] != shapes.shape[-2]:
            raise ValueError("shapes must have shape (N,d,d)")
        if not torch.isfinite(shapes).all() or not torch.allclose(
            shapes, shapes.transpose(-1, -2), rtol=1e-5, atol=1e-7
        ):
            raise ValueError("shapes must be finite and symmetric")
        chol = torch.linalg.cholesky(shapes)
        d = shapes.shape[-1]
        logdet = 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        if math.isinf(self.threshold):
            return torch.full_like(logdet, float("inf"))
        unit_ball_log_volume = (d / 2.0) * math.log(math.pi) - math.lgamma(
            d / 2.0 + 1.0
        )
        return (
            unit_ball_log_volume
            + 0.5 * logdet
            + (d / 2.0) * math.log(self.threshold)
        )


def fit_split_conformal(
    means: torch.Tensor,
    shapes: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.1,
) -> SplitConformalRegion:
    """Fit a finite-sample split-conformal ellipsoidal region.

    The threshold is the ``ceil((n+1)(1-alpha))`` order statistic of squared
    shape-Mahalanobis scores.  If the rank is ``n+1``, the conventional
    finite-sample endpoint is ``+inf``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    n, _ = _validate_inputs(means, shapes, targets)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        threshold = float("inf")
    else:
        scores = _squared_shape_scores(means, shapes, targets)
        threshold = float(torch.kthvalue(scores, rank).values.item())
    return SplitConformalRegion(
        threshold=threshold,
        alpha=float(alpha),
        calibration_size=n,
        rank=rank,
    )


def evaluate_region(
    region: SplitConformalRegion,
    means: torch.Tensor,
    shapes: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Summarize empirical coverage and mean log-volume on an evaluation set."""
    membership = region.contains(means, shapes, targets)
    log_volume = region.log_volume(shapes)
    return {
        "coverage": float(membership.float().mean().item()),
        "mean_log_volume": float(log_volume.mean().item()),
        "n": int(membership.numel()),
    }


__all__ = ["SplitConformalRegion", "evaluate_region", "fit_split_conformal"]
