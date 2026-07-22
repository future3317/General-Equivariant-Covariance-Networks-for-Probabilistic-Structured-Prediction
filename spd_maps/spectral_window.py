"""Uniformly conditioned, orthogonally equivariant SPD map."""

from __future__ import annotations

import torch

from spd_maps.base import SPDMap, symmetrize
from spd_maps.matrix_softplus import _SpectralMapFunction


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
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        log_variance = self._log_variance(eigenvalues)
        projected = torch.matmul(
            eigenvectors.transpose(-1, -2), residual.unsqueeze(-1)
        ).squeeze(-1)
        return log_variance.sum(-1), (projected.square() * torch.exp(-log_variance)).sum(-1)

    def logdet(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        return self._log_variance(torch.linalg.eigvalsh(A)).sum(-1)

    def precision_action(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        _, quadratic = self.statistics(A, residual)
        return quadratic
