"""Abstract base class for SPD maps."""

from __future__ import annotations

import abc
import torch


class SPDMap(torch.nn.Module, abc.ABC):
    """Map a symmetric operator ``A`` to an SPD matrix ``S``.

    The map must preserve orthogonal conjugation equivariance:

    .. math::

        S(\\rho(g) A \\rho(g)^\\top) = \\rho(g) S(A) \\rho(g)^\\top.

    Subclasses implement ``forward`` (the map), ``logdet`` (log determinant of
    the resulting SPD matrix), and ``precision_action`` (quadratic form with
    the precision matrix).
    """

    @abc.abstractmethod
    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Return SPD matrices ``S`` of shape ``(..., d, d)``.

        Args:
            A: Symmetric matrices of shape ``(..., d, d)``.
        """
        ...

    @abc.abstractmethod
    def logdet(self, A: torch.Tensor) -> torch.Tensor:
        """Return ``log det S(A)`` of shape ``(...)``.

        This is used directly in NLL losses and avoids recomputing ``S``.
        """
        ...

    @abc.abstractmethod
    def precision_action(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Return ``r^T S(A)^{-1} r`` of shape ``(...)``.

        Args:
            A: Symmetric matrices of shape ``(..., d, d)``.
            residual: Vectors of shape ``(..., d)``.
        """
        ...

    def precision(self, A: torch.Tensor) -> torch.Tensor:
        """Return the precision matrix, computed lazily by default."""
        return torch.linalg.inv(self.forward(A))


def symmetrize(A: torch.Tensor) -> torch.Tensor:
    """Numerical symmetrization."""
    return 0.5 * (A + A.transpose(-1, -2))
