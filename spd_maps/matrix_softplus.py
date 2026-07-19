"""Spectral softplus SPD map with Löwner divided-difference autograd."""

from __future__ import annotations

import torch

from spd_maps.base import SPDMap, symmetrize


class _SpectralMapFunction(torch.autograd.Function):
    """Custom autograd for ``S = U diag(f(λ)) U^T`` using Löwner divided differences."""

    @staticmethod
    def forward(ctx, A, f, df, eps, delta_threshold):
        # A is assumed symmetric.
        L, Q = torch.linalg.eigh(A)
        fL = f(L) + eps
        S = torch.matmul(Q, torch.matmul(torch.diag_embed(fL), Q.transpose(-1, -2)))
        ctx.save_for_backward(Q, L)
        ctx.f = f
        ctx.df = df
        ctx.delta_threshold = delta_threshold
        return S

    @staticmethod
    def backward(ctx, grad_output):
        Q, L = ctx.saved_tensors
        grad_output = symmetrize(grad_output)

        # Project gradient to eigenbasis.
        G = torch.matmul(Q.transpose(-1, -2), torch.matmul(grad_output, Q))

        # Löwner divided-difference matrix.
        delta = L.unsqueeze(-1) - L.unsqueeze(-2)  # (..., d, d)
        lam_i = L.unsqueeze(-1)  # (..., d, 1)
        lam_j = L.unsqueeze(-2)  # (..., 1, d)
        lam_mid = 0.5 * (lam_i + lam_j)  # (..., d, d)

        df = ctx.df
        f = ctx.f
        threshold = ctx.delta_threshold

        # Off-diagonal: divided differences, broadcast over the d x d grid.
        denom = torch.where(torch.abs(delta) > threshold, delta, torch.ones_like(delta))
        F_off = (f(lam_i) - f(lam_j)) / denom

        # Near-diagonal and diagonal: derivative at midpoint / eigenvalue.
        F_near = df(lam_mid)
        F = torch.where(torch.abs(delta) > threshold, F_off, F_near)

        # Explicitly enforce the diagonal to be f'(λ_i).
        d = F.shape[-1]
        eye_mask = torch.eye(d, device=F.device, dtype=torch.bool).expand(F.shape)
        F_diag = df(L)  # (..., d)
        F = torch.where(eye_mask, F_diag.unsqueeze(-1).expand_as(F), F)

        # Gradient w.r.t. A.
        grad_input = torch.matmul(Q, torch.matmul(F * G, Q.transpose(-1, -2)))
        return grad_input, None, None, None, None


class SpectralSoftplusMap(SPDMap):
    """SPD map via spectral softplus: :math:`S = U \\operatorname{diag}(\\text{softplus}(λ)+ε) U^T`.

    This is a spectral map that produces strictly positive eigenvalues. Unlike
    the matrix exponential it is not globally surjective onto SPD (the smallest
    eigenvalue is bounded below by ``eps``), but it avoids exponential growth
    and has stable gradients near degenerate eigenvalues.
    """

    def __init__(self, eps: float = 1e-5, delta_threshold: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.delta_threshold = delta_threshold

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x)

    def _df(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        return _SpectralMapFunction.apply(A, self._f, self._df, self.eps, self.delta_threshold)

    def logdet(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        L = torch.linalg.eigvalsh(A)
        return torch.sum(torch.log(torch.nn.functional.softplus(L) + self.eps), dim=-1)

    def precision_action(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        L, Q = torch.linalg.eigh(A)
        fL = torch.nn.functional.softplus(L) + self.eps
        # Whiten residual in eigenbasis.
        z = torch.matmul(Q.transpose(-1, -2), residual.unsqueeze(-1)).squeeze(-1)
        return torch.sum((z * z) / fL, dim=-1)
