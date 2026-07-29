"""Automatic construction of the exterior square ``Lambda^2(V)`` for O(3)."""

from __future__ import annotations

import torch
from compatibility.e3nn import o3


def exterior_square_irreps(output_irreps: o3.Irreps) -> o3.Irreps:
    """Return the O(3) irrep decomposition of ``Lambda^2(V)``."""
    groups = [
        (int(multiplicity), int(irrep.l), int(irrep.p))
        for multiplicity, irrep in o3.Irreps(output_irreps)
    ]
    counts: dict[tuple[int, int], int] = {}

    def add(angular_momentum: int, parity: int, multiplicity: int) -> None:
        if multiplicity:
            counts[(angular_momentum, parity)] = (
                counts.get((angular_momentum, parity), 0) + multiplicity
            )

    for multiplicity, angular_momentum, parity in groups:
        symmetric_copies = multiplicity * (multiplicity + 1) // 2
        antisymmetric_copies = multiplicity * (multiplicity - 1) // 2
        for output_l in range(0, 2 * angular_momentum + 1):
            # The swap sign of the CG channel is (-1)^(2l-L).
            if (2 * angular_momentum - output_l) % 2 == 0:
                add(output_l, parity * parity, antisymmetric_copies)
            else:
                add(output_l, parity * parity, symmetric_copies)

    for index, (left_mul, left_l, left_parity) in enumerate(groups):
        for right_mul, right_l, right_parity in groups[index + 1 :]:
            for output_l in range(abs(left_l - right_l), left_l + right_l + 1):
                add(output_l, left_parity * right_parity, left_mul * right_mul)

    return o3.Irreps(
        [
            (multiplicity, (angular_momentum, parity))
            for (angular_momentum, parity), multiplicity in sorted(
                counts.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ]
    )


class O3SkewOperatorBasis(torch.nn.Module):
    """Orthonormal skew-matrix basis compiled from ``ij=-ji``."""

    def __init__(
        self, output_irreps: o3.Irreps, *, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.output_irreps = o3.Irreps(output_irreps)
        self._output_dim = self.output_irreps.dim
        requested_dtype = dtype or torch.get_default_dtype()
        previous_dtype = torch.get_default_dtype()
        if requested_dtype != previous_dtype:
            torch.set_default_dtype(requested_dtype)
        try:
            rtp = o3.ReducedTensorProducts("ij=-ji", i=self.output_irreps)
        finally:
            if requested_dtype != previous_dtype:
                torch.set_default_dtype(previous_dtype)
        self._operator_irreps = rtp.irreps_out
        self._operator_dim = self._operator_irreps.dim
        basis = rtp.change_of_basis.reshape(
            self._operator_dim, self._output_dim, self._output_dim
        )
        self.register_buffer("_basis", basis.to(dtype=requested_dtype))

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
        if coefficients.shape[-1] != self._operator_dim:
            raise ValueError(
                f"coefficients last dim {coefficients.shape[-1]} != operator_dim {self._operator_dim}"
            )
        matrix = torch.einsum("...q,qij->...ij", coefficients, self._basis)
        return 0.5 * (matrix - matrix.transpose(-1, -2))

    def project(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.shape[-2:] != (self._output_dim, self._output_dim):
            raise ValueError("matrix dimensions do not match output representation")
        return torch.einsum("...ij,qij->...q", matrix, self._basis)
