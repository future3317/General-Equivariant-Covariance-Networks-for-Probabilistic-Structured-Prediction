"""Automatic construction of Sym^2(V) for O(3) irreps."""

from __future__ import annotations

import torch
from compatibility.e3nn import o3


def symmetric_square_irreps(output_irreps: o3.Irreps) -> o3.Irreps:
    """Return the irreps of :math:`\\operatorname{Sym}^2(V)`.

    Args:
        output_irreps: Irreps of the output representation ``V``.

    Returns:
        Irreps of the symmetric-square space.
    """
    # Compute the irrep *types* analytically.  Constructing an e3nn
    # ReducedTensorProducts object for a high-multiplicity representation
    # (e.g. ITOP's ``15x1o``) materializes a 45x45 change-of-basis tensor and
    # can take minutes.  The compiler only needs the decomposition here; the
    # concrete basis is still materialized lazily by O3SymmetricOperatorBasis
    # for families that actually assemble a dense operator.
    irreps = o3.Irreps(output_irreps)
    groups: list[tuple[int, int, int]] = []
    for multiplicity, irrep in irreps:
        groups.append((int(multiplicity), int(irrep.l), int(irrep.p)))

    counts: dict[tuple[int, int], int] = {}

    def add(angular_momentum: int, parity: int, multiplicity: int) -> None:
        if multiplicity:
            key = (angular_momentum, parity)
            counts[key] = counts.get(key, 0) + multiplicity

    # Self-products: the swap symmetry of the L channel is (-1)^(2l-L).
    for multiplicity, angular_momentum, parity in groups:
        symmetric_copies = multiplicity * (multiplicity + 1) // 2
        antisymmetric_copies = multiplicity * (multiplicity - 1) // 2
        for output_l in range(0, 2 * angular_momentum + 1):
            if (2 * angular_momentum - output_l) % 2 == 0:
                add(output_l, 1, symmetric_copies)
            else:
                add(output_l, 1, antisymmetric_copies)

    # Cross-products have no exchange constraint and occur once per pair of
    # multiplicity channels.
    for index, (left_mul, left_l, left_parity) in enumerate(groups):
        for right_mul, right_l, right_parity in groups[index + 1 :]:
            for output_l in range(abs(left_l - right_l), left_l + right_l + 1):
                add(output_l, left_parity * right_parity, left_mul * right_mul)

    ordered = sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
    return o3.Irreps(
        [
            (multiplicity, (angular_momentum, parity))
            for (angular_momentum, parity), multiplicity in ordered
        ]
    )


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
            basis = basis.reshape(
                self._operator_dim, self._output_dim, self._output_dim
            )

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
