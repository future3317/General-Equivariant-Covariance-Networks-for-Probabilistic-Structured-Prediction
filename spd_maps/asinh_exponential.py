"""Full-image SPD map based on ``exp(asinh(lambda))`` spectral calculus."""

from __future__ import annotations

import torch

from spd_maps.base import SPDMap, symmetrize
from spd_maps.matrix_softplus import _SpectralMapFunction


class _AsinhExponentialStatisticsFunction(torch.autograd.Function):
    """Stable joint statistics with a repeated-eigenvalue-safe VJP."""

    @staticmethod
    def forward(ctx, A, residual, delta_threshold):
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        projected = torch.matmul(
            eigenvectors.transpose(-1, -2), residual.unsqueeze(-1)
        ).squeeze(-1)
        log_spectrum = torch.asinh(eigenvalues)
        inverse_spectrum = torch.exp(-log_spectrum)
        ctx.save_for_backward(eigenvalues, eigenvectors, projected, inverse_spectrum)
        ctx.delta_threshold = delta_threshold
        return log_spectrum.sum(-1), (projected.square() * inverse_spectrum).sum(-1)

    @staticmethod
    def backward(ctx, grad_logdet, grad_quadratic):
        eigenvalues, eigenvectors, projected, inverse_spectrum = ctx.saved_tensors
        delta = eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2)
        denominator = torch.where(
            delta.abs() > ctx.delta_threshold, delta, torch.ones_like(delta)
        )
        lambda_i = eigenvalues.unsqueeze(-1)
        lambda_j = eigenvalues.unsqueeze(-2)
        h_i = inverse_spectrum.unsqueeze(-1)
        h_j = inverse_spectrum.unsqueeze(-2)
        divided = (h_i - h_j) / denominator
        midpoint = 0.5 * (lambda_i + lambda_j)
        h_derivative = -torch.exp(-torch.asinh(midpoint)) / torch.sqrt(
            1.0 + midpoint.square()
        )
        loewner = torch.where(delta.abs() > ctx.delta_threshold, divided, h_derivative)
        diagonal_derivative = 1.0 / torch.sqrt(1.0 + eigenvalues.square())
        gradient_eigenbasis = torch.diag_embed(
            grad_logdet.unsqueeze(-1) * diagonal_derivative
        )
        gradient_eigenbasis = gradient_eigenbasis + grad_quadratic[
            ..., None, None
        ] * loewner * (projected[..., :, None] * projected[..., None, :])
        grad_A = torch.matmul(
            eigenvectors,
            torch.matmul(gradient_eigenbasis, eigenvectors.transpose(-1, -2)),
        )
        grad_residual = 2.0 * grad_quadratic[..., None] * torch.matmul(
            eigenvectors, (inverse_spectrum * projected).unsqueeze(-1)
        ).squeeze(-1)
        return 0.5 * (grad_A + grad_A.transpose(-1, -2)), grad_residual, None


class AsinhExponentialMap(SPDMap):
    r"""Map a symmetric generator by ``f(lambda)=exp(asinh(lambda))``.

    The scalar map is a smooth bijection from :math:`\mathbb R` to
    :math:`\mathbb R_{>0}`.  The custom spectral VJP uses Löwner divided
    differences and a midpoint derivative at repeated eigenvalues.
    """

    def __init__(self, delta_threshold: float = 1e-6) -> None:
        super().__init__()
        if delta_threshold <= 0.0:
            raise ValueError("delta_threshold must be positive")
        self.delta_threshold = float(delta_threshold)

    @staticmethod
    def _f(values: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.asinh(values))

    @staticmethod
    def _df(values: torch.Tensor) -> torch.Tensor:
        asinh = torch.asinh(values)
        return torch.exp(asinh) / torch.sqrt(1.0 + values.square())

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        return _SpectralMapFunction.apply(
            A, self._f, self._df, 0.0, self.delta_threshold
        )

    def logdet(self, A: torch.Tensor) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvalsh(symmetrize(A))
        return torch.asinh(eigenvalues).sum(-1)

    def precision_action(
        self, A: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return self.statistics(A, residual)[1]

    def statistics(
        self, A: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _AsinhExponentialStatisticsFunction.apply(
            symmetrize(A), residual, self.delta_threshold
        )
