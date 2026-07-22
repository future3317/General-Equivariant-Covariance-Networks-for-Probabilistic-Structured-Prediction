"""Equivariant block-metric wrapper for SPD maps."""

from __future__ import annotations

import torch

from spd_maps.base import SPDMap


class RepresentationMetricMap(SPDMap):
    """Wrap an SPD map in an equivariant block metric.

    ``metric`` is a positive diagonal row-vector scaling.  For ``0e + 2e``
    outputs it must have the form ``[s0, s2, s2, s2, s2, s2]``; this commutes
    with every orthogonal representation matrix and therefore preserves the
    compiler's equivariance contract.  The wrapped map acts in scaled
    coordinates, while this class exposes the corresponding physical scatter.
    """

    def __init__(self, base: SPDMap, metric: torch.Tensor):
        super().__init__()
        if metric.ndim != 1 or metric.numel() < 1:
            raise ValueError("metric must be a one-dimensional non-empty vector")
        if not bool(torch.isfinite(metric).all()) or bool((metric <= 0).any()):
            raise ValueError("metric entries must be finite and positive")
        self.base = base
        self.register_buffer("metric", metric.detach().clone())
        self.register_buffer("inverse_metric", metric.detach().clone().reciprocal())
        self._log_metric = float(torch.log(metric).sum().item())

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        scaled = self.base(A)
        d_inv = self.inverse_metric.to(dtype=scaled.dtype, device=scaled.device)
        return d_inv.unsqueeze(-1) * scaled * d_inv.unsqueeze(-2)

    def logdet(self, A: torch.Tensor) -> torch.Tensor:
        return self.base.logdet(A) - 2.0 * A.new_tensor(self._log_metric)

    def precision_action(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        metric = self.metric.to(dtype=residual.dtype, device=residual.device)
        return self.base.precision_action(A, residual * metric)

    def precision(self, A: torch.Tensor) -> torch.Tensor:
        precision = self.base.precision(A)
        metric = self.metric.to(dtype=precision.dtype, device=precision.device)
        return metric.unsqueeze(-1) * precision * metric.unsqueeze(-2)

    def statistics(
        self, A: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metric = self.metric.to(dtype=residual.dtype, device=residual.device)
        logdet, quadratic = self.base.statistics(A, residual * metric)
        return logdet - 2.0 * A.new_tensor(self._log_metric), quadratic
