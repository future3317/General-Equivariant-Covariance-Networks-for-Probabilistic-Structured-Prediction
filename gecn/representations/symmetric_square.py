"""Automatic construction of Sym^2(V) for O(3) irreps."""

from __future__ import annotations

import torch
from e3nn import o3

from gecn.representations.base import SymmetricSquareSpec


def symmetric_square_irreps(output_irreps: o3.Irreps) -> o3.Irreps:
    """Return the irreps of :math:`\\operatorname{Sym}^2(V)`.

    Args:
        output_irreps: Irreps of the output representation ``V``.

    Returns:
        Irreps of the symmetric-square space.
    """
    rtp = o3.ReducedTensorProducts("ij=ji", i=output_irreps)
    return rtp.irreps_out


class O3SymmetricOperatorBasis(torch.nn.Module):
    """Basis for equivariant symmetric operators on an O(3) representation.

    Given output irreps ``V``, this module builds the change-of-basis from
    the irrep coefficients of :math:`\\operatorname{Sym}^2(V)` to symmetric
    matrices on ``V``. The basis is obtained from ``e3nn.o3.ReducedTensorProducts``
    with the symmetry constraint ``ij=ji``.

    The basis matrices are orthonormal in the Frobenius inner product on
    ``V \\otimes V``.
    """

    def __init__(self, output_irreps: o3.Irreps):
        super().__init__()
        self.output_irreps = o3.Irreps(output_irreps)
        self._output_dim = self.output_irreps.dim

        rtp = o3.ReducedTensorProducts("ij=ji", i=self.output_irreps)
        self._operator_irreps = rtp.irreps_out
        self._operator_dim = self._operator_irreps.dim

        # e3nn returns change_of_basis with shape
        # (irreps_out.dim, irreps_in1.dim, irreps_in2.dim, ...).
        # Here it is (operator_dim, output_dim, output_dim).
        basis = rtp.change_of_basis
        if basis.shape != (self._operator_dim, self._output_dim, self._output_dim):
            # Defensive reshape in case future e3nn changes the layout.
            basis = basis.reshape(self._operator_dim, self._output_dim, self._output_dim)

        self.register_buffer("_basis", basis)

    @property
    def operator_irreps(self) -> o3.Irreps:
        return self._operator_irreps

    @property
    def operator_dim(self) -> int:
        return self._operator_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def basis(self) -> torch.Tensor:
        return self._basis

    def assemble(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Assemble symmetric operator ``A`` from irrep coefficients.

        Args:
            coefficients: ``(..., operator_dim)``.

        Returns:
            Symmetric matrices ``A`` of shape ``(..., output_dim, output_dim)``.
        """
        if coefficients.shape[-1] != self._operator_dim:
            raise ValueError(
                f"coefficients last dim {coefficients.shape[-1]} != operator_dim {self._operator_dim}"
            )
        A = torch.einsum("...q,qij->...ij", coefficients, self._basis)
        # Numerical symmetrization; basis is already symmetric but rounding can occur.
        return 0.5 * (A + A.transpose(-1, -2))

    def project(self, A: torch.Tensor) -> torch.Tensor:
        """Project a symmetric matrix to irrep coefficients via the Frobenius inner product.

        Because the basis is orthonormal, the inverse of ``assemble`` is simply
        the contraction ``coefficients[q] = <basis[q], A>_F``.

        Args:
            A: Symmetric matrices of shape ``(..., output_dim, output_dim)``.

        Returns:
            Coefficients of shape ``(..., operator_dim)``.
        """
        if A.shape[-2:] != (self._output_dim, self._output_dim):
            raise ValueError(
                f"A matrix dims {A.shape[-2:]} != ({self._output_dim}, {self._output_dim})"
            )
        # Contract over the matrix indices.
        return torch.einsum("...ij,qij->...q", A, self._basis)

    def extra_repr(self) -> str:
        return (
            f"output_irreps={self.output_irreps}, "
            f"operator_irreps={self._operator_irreps}, "
            f"output_dim={self._output_dim}, operator_dim={self._operator_dim}"
        )
