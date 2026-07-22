"""Uniformly conditioned, orthogonally equivariant SPD map."""

from __future__ import annotations

import torch

from spd_maps.base import SPDMap, symmetrize
from spd_maps.matrix_softplus import _SpectralMapFunction


class _SpectralWindowStatisticsFunction(torch.autograd.Function):
    """Gaussian sufficient statistics with a finite repeated-eigenvalue VJP."""

    @staticmethod
    def forward(ctx, A, residual, log_variance_min, log_variance_max, threshold):
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        width = log_variance_max - log_variance_min
        sigmoid = torch.sigmoid(eigenvalues)
        log_variance = log_variance_min + width * sigmoid
        inverse_variance = torch.exp(-log_variance)
        projected = torch.matmul(
            eigenvectors.transpose(-1, -2), residual.unsqueeze(-1)
        ).squeeze(-1)
        ctx.save_for_backward(eigenvectors, eigenvalues, projected, inverse_variance)
        ctx.log_variance_min = log_variance_min
        ctx.log_variance_max = log_variance_max
        ctx.threshold = threshold
        return log_variance.sum(-1), (projected.square() * inverse_variance).sum(-1)

    @staticmethod
    def backward(ctx, grad_logdet, grad_quadratic):
        eigenvectors, eigenvalues, projected, inverse_variance = ctx.saved_tensors
        width = ctx.log_variance_max - ctx.log_variance_min
        sigmoid = torch.sigmoid(eigenvalues)
        log_variance_derivative = width * sigmoid * (1.0 - sigmoid)

        # Gradient of tr(log Sigma(A)): g'(A) for g(lambda)=log variance(lambda).
        diagonal = grad_logdet.unsqueeze(-1) * log_variance_derivative
        gradient_eigenbasis = torch.diag_embed(diagonal)

        # Gradient of r^T h(A) r, h(lambda)=1 / variance(lambda), using
        # Lowner divided differences.  This remains defined when eigenvalues
        # coincide, where the ordinary eigvector VJP is undefined.
        delta = eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2)
        lambda_i = eigenvalues.unsqueeze(-1)
        lambda_j = eigenvalues.unsqueeze(-2)
        denominator = torch.where(
            delta.abs() > ctx.threshold, delta, torch.ones_like(delta)
        )
        h_i = inverse_variance.unsqueeze(-1)
        h_j = inverse_variance.unsqueeze(-2)
        divided = (h_i - h_j) / denominator
        midpoint = 0.5 * (lambda_i + lambda_j)
        midpoint_sigmoid = torch.sigmoid(midpoint)
        derivative = -torch.exp(
            -(ctx.log_variance_min + width * midpoint_sigmoid)
        ) * width * midpoint_sigmoid * (1.0 - midpoint_sigmoid)
        lowner = torch.where(delta.abs() > ctx.threshold, divided, derivative)
        quadratic_eigenbasis = lowner * (
            projected.unsqueeze(-1) * projected.unsqueeze(-2)
        )
        gradient_eigenbasis = gradient_eigenbasis + grad_quadratic.unsqueeze(
            -1
        ).unsqueeze(-1) * quadratic_eigenbasis
        grad_A = torch.matmul(
            eigenvectors,
            torch.matmul(gradient_eigenbasis, eigenvectors.transpose(-1, -2)),
        )
        grad_A = symmetrize(grad_A)

        grad_residual = 2.0 * grad_quadratic.unsqueeze(-1) * torch.matmul(
            eigenvectors,
            (inverse_variance * projected).unsqueeze(-1),
        ).squeeze(-1)
        return grad_A, grad_residual, None, None, None


class SpectralWindowMap(SPDMap):
    """Map a symmetric generator to an SPD matrix with bounded spectrum.

    For ``A = Q diag(lambda) Q^T``, this map returns

    ``Q diag(exp(lower + (upper - lower) * sigmoid(lambda))) Q^T``.

    The same spectral map is used for the likelihood and returned covariance.
    Consequently, the covariance eigenvalues lie in the closed interval
    ``[exp(lower), exp(upper)]`` and the condition number is at most
    ``exp(upper - lower)``.  Spectral functional calculus also makes the map
    exactly equivariant under conjugation by every orthogonal representation.
    """

    def __init__(
        self,
        log_variance_min: float,
        log_variance_max: float,
        delta_threshold: float = 1e-6,
    ):
        super().__init__()
        if not log_variance_min < log_variance_max:
            raise ValueError("log_variance_min must be smaller than log_variance_max")
        if delta_threshold <= 0.0:
            raise ValueError("delta_threshold must be positive")
        self.log_variance_min = float(log_variance_min)
        self.log_variance_max = float(log_variance_max)
        self.delta_threshold = float(delta_threshold)

    @property
    def max_condition_number(self) -> float:
        return float(torch.exp(torch.tensor(self.log_variance_max - self.log_variance_min)))

    def _log_variance(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        width = self.log_variance_max - self.log_variance_min
        return self.log_variance_min + width * torch.sigmoid(eigenvalues)

    def _f(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._log_variance(eigenvalues))

    def _df(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        mapped = self._f(eigenvalues)
        sigmoid = torch.sigmoid(eigenvalues)
        return mapped * (self.log_variance_max - self.log_variance_min) * sigmoid * (1.0 - sigmoid)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        return _SpectralMapFunction.apply(
            A, self._f, self._df, 0.0, self.delta_threshold
        )

    def statistics(
        self, A: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        A = symmetrize(A)
        return _SpectralWindowStatisticsFunction.apply(
            A,
            residual,
            self.log_variance_min,
            self.log_variance_max,
            self.delta_threshold,
        )

    def logdet(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        return self._log_variance(torch.linalg.eigvalsh(A)).sum(-1)

    def precision_action(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        _, quadratic = self.statistics(A, residual)
        return quadratic
