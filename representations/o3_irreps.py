"""Concrete O(3) orthogonal representation specification."""

from __future__ import annotations

import torch
from e3nn import o3

from representations.base import OrthogonalRepresentationSpec
from representations.symmetric_square import O3SymmetricOperatorBasis


class O3IrrepsSpec(OrthogonalRepresentationSpec):
    """O(3) representation specified by e3nn irreps.

    The representation matrices are built from the standard Wigner-D matrices
    for SO(3), extended to O(3) by the parity of each irrep.

    This is the concrete implementation used by the paper's experiments. The
    abstract interface ``OrthogonalRepresentationSpec`` leaves room for other
    groups, but only ``O3IrrepsSpec`` is fully supported in this release.
    """

    def __init__(self, irreps: o3.Irreps):
        self.irreps = o3.Irreps(irreps)
        self._dim = self.irreps.dim
        self._symmetric_square = O3SymmetricOperatorBasis(self.irreps)

    @property
    def dim(self) -> int:
        return self._dim

    def representation_matrix(self, group_element: torch.Tensor) -> torch.Tensor:
        """Build the orthogonal representation matrix of ``R ∈ O(3)``.

        Args:
            group_element: Rotation (or reflection) matrix ``R`` of shape
                ``(3, 3)`` or ``(..., 3, 3)``.

        Returns:
            Representation matrix ``ρ(R)`` of shape ``(dim, dim)`` (or batched).
        """
        R = torch.as_tensor(group_element, dtype=torch.get_default_dtype())
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"O(3) element must have shape (..., 3, 3), got {R.shape}")
        # e3nn's D_from_matrix returns the Wigner-D matrices for each irrep.
        return self.irreps.D_from_matrix(R)

    def symmetric_square(self) -> O3SymmetricOperatorBasis:
        """Return the symmetric-square basis for this representation."""
        return self._symmetric_square
