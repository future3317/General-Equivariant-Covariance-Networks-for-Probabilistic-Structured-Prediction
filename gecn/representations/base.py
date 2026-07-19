"""Abstract base interfaces for orthogonal representations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class OrthogonalRepresentationSpec(Protocol):
    """Specification of a finite-dimensional orthogonal representation.

    The representation is a homomorphism :math:`\\rho: G \\to O(V)` where
    :math:`V` is a real Euclidean space equipped with a standard inner product.
    """

    @property
    def dim(self) -> int:
        """Dimension of the representation space V."""
        ...

    def representation_matrix(self, group_element) -> torch.Tensor:
        """Return the orthogonal matrix :math:`\\rho(g) \\in O(V)`.

        Args:
            group_element: An element of the group. For O(3) this is typically
                a rotation matrix ``R`` of shape ``(3, 3)``.

        Returns:
            Tensor of shape ``(dim, dim)``.
        """
        ...

    def symmetric_square(self) -> "SymmetricSquareSpec":
        """Return the specification of :math:`\\operatorname{Sym}^2(V)`."""
        ...


@runtime_checkable
class SymmetricSquareSpec(Protocol):
    """Specification of the symmetric square of a representation.

    This describes how :math:`\\operatorname{Sym}^2(V)` decomposes into
    irreps and provides the change-of-basis tensor that maps coefficients
    in the irrep basis to symmetric operators on ``V``.
    """

    @property
    def operator_irreps(self):
        """Irreps of the symmetric-square space."""
        ...

    @property
    def operator_dim(self) -> int:
        """Total dimension of Sym^2(V), i.e. dim*(dim+1)//2."""
        ...

    @property
    def output_dim(self) -> int:
        """Dimension of the original output representation V."""
        ...

    @property
    def basis(self) -> torch.Tensor:
        """Change-of-basis tensor of shape ``(operator_dim, output_dim, output_dim)``.

        Each slice ``basis[q]`` is a symmetric matrix on ``V``.
        """
        ...

    def assemble(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Assemble a symmetric operator from irrep coefficients.

        Args:
            coefficients: Tensor of shape ``(..., operator_dim)``.

        Returns:
            Symmetric matrices of shape ``(..., output_dim, output_dim)``.
        """
        ...

    def project(self, A: torch.Tensor) -> torch.Tensor:
        """Project a symmetric matrix to irrep coefficients.

        Args:
            A: Symmetric matrices of shape ``(..., output_dim, output_dim)``.

        Returns:
            Coefficients of shape ``(..., operator_dim)``.
        """
        ...
