"""Helpers to derive O(3) output irreps from Cartesian tensors."""

from __future__ import annotations

from compatibility.e3nn import CartesianTensor, o3


def rank2_symmetric_irreps() -> o3.Irreps:
    """Return irreps for a symmetric rank-2 Cartesian tensor, ``ij=ji``.

    This corresponds to the space of :math:`3 \\times 3` symmetric matrices,
    i.e. ``0e + 2e``.
    """
    return o3.Irreps(CartesianTensor("ij=ji"))


def rank4_elasticity_irreps() -> o3.Irreps:
    """Return irreps for the elastic stiffness tensor, ``ijkl=jikl=ijlk=klij``.

    The resulting irreps have dimension 21. Note: the *full* covariance of a
    21-dimensional output lives in ``Sym^2(R^21)`` and has 231 parameters; use
    a low-rank parameterization if the full model is too expensive.
    """
    return o3.Irreps(CartesianTensor("ijkl=jikl=ijlk=klij"))
