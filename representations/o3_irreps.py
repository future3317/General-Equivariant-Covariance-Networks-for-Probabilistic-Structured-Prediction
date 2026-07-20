"""Concrete O(3) orthogonal representation specification."""

from __future__ import annotations

import torch
from compatibility.e3nn import CartesianTensor, o3

from representations.base import OrthogonalRepresentationSpec
from representations.symmetric_square import (
    O3SymmetricOperatorBasis,
    symmetric_square_irreps,
)


class O3IrrepsSpec(OrthogonalRepresentationSpec):
    """O(3) representation specified by e3nn irreps.

    The representation matrices are built from the standard Wigner-D matrices
    for SO(3), extended to O(3) by the parity of each irrep.

    This is the concrete implementation used by the paper's experiments. The
    abstract interface ``OrthogonalRepresentationSpec`` leaves room for other
    groups, but only ``O3IrrepsSpec`` is fully supported in this release.
    """

    def __init__(
        self,
        irreps: o3.Irreps,
        *,
        cartesian_formula: str | None = None,
    ):
        self.irreps = o3.Irreps(irreps)
        self.cartesian_formula = cartesian_formula
        self._cartesian = (
            CartesianTensor(cartesian_formula)
            if cartesian_formula is not None
            else None
        )
        if self._cartesian is not None and o3.Irreps(self._cartesian) != self.irreps:
            raise ValueError(
                f"irreps {self.irreps} do not match Cartesian symmetry "
                f"'{cartesian_formula}' ({o3.Irreps(self._cartesian)})"
            )
        self._dim = self.irreps.dim
        self._symmetric_square_irreps = symmetric_square_irreps(self.irreps)
        self._symmetric_square: O3SymmetricOperatorBasis | None = None

    @classmethod
    def from_cartesian(cls, formula: str) -> "O3IrrepsSpec":
        """Compile a Cartesian permutation-symmetry formula to O(3) irreps."""
        cartesian = CartesianTensor(formula)
        return cls(o3.Irreps(cartesian), cartesian_formula=formula)

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
        R = torch.as_tensor(group_element)
        if not R.is_floating_point():
            R = R.to(dtype=torch.get_default_dtype())
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"O(3) element must have shape (..., 3, 3), got {R.shape}")
        # e3nn's D_from_matrix returns the Wigner-D matrices for each irrep.
        return self.irreps.D_from_matrix(R)

    def symmetric_square(self) -> O3SymmetricOperatorBasis:
        """Return the symmetric-square basis for this representation."""
        if self._symmetric_square is None:
            self._symmetric_square = O3SymmetricOperatorBasis(self.irreps)
        return self._symmetric_square

    @property
    def symmetric_square_irreps(self) -> o3.Irreps:
        """Return ``Sym^2(V)`` irreps without materializing the basis module."""
        return self._symmetric_square_irreps

    def from_cartesian_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert a Cartesian tensor to the compiled irrep coordinates."""
        if self._cartesian is None:
            raise ValueError("this output specification has no Cartesian formula")
        return self._cartesian.from_cartesian(tensor)

    def to_cartesian_tensor(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Convert compiled irrep coordinates back to a Cartesian tensor."""
        if self._cartesian is None:
            raise ValueError("this output specification has no Cartesian formula")
        return self._cartesian.to_cartesian(coefficients)
