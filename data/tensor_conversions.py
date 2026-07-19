"""Coordinate conversions between Voigt/Kelvin-Mandel, Cartesian tensors, and e3nn irreps."""

from __future__ import annotations

import itertools
import torch
from e3nn import o3
from e3nn.io import CartesianTensor

from voigt_utils import (
    voigt_to_tensor,
    tensor_to_voigt,
    voigt_to_kelvin_mandel,
    kelvin_mandel_to_voigt,
)


# Cartesian tensors for common symmetries.
_CARTESIAN_RANK2 = CartesianTensor("ij=ji")
_CARTESIAN_RANK4 = CartesianTensor("ijkl=jikl=ijlk=klij")

# Voigt index for a symmetric pair (i, j).
_VOIGT_PAIR = {
    (0, 0): 0,
    (1, 1): 1,
    (2, 2): 2,
    (1, 2): 3,
    (2, 1): 3,
    (0, 2): 4,
    (2, 0): 4,
    (0, 1): 5,
    (1, 0): 5,
}

# Unique (I, J) pairs for a 21-dimensional elasticity vector.
_ELASTICITY_21_INDICES = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
]


def voigt_to_irreps(voigt: torch.Tensor) -> torch.Tensor:
    """Convert a 6D Voigt vector to ``0e + 2e`` irrep coefficients."""
    tensor = voigt_to_tensor(voigt)
    return _CARTESIAN_RANK2.from_cartesian(tensor)


def irreps_to_voigt(irreps: torch.Tensor) -> torch.Tensor:
    """Convert ``0e + 2e`` irrep coefficients to a 6D Voigt vector."""
    tensor = _CARTESIAN_RANK2.to_cartesian(irreps)
    return tensor_to_voigt(tensor)


def km_to_irreps(km: torch.Tensor) -> torch.Tensor:
    """Convert a 6D Kelvin-Mandel vector to ``0e + 2e`` irrep coefficients."""
    voigt = kelvin_mandel_to_voigt(km)
    return voigt_to_irreps(voigt)


def irreps_to_km(irreps: torch.Tensor) -> torch.Tensor:
    """Convert ``0e + 2e`` irrep coefficients to a 6D Kelvin-Mandel vector."""
    voigt = irreps_to_voigt(irreps)
    return voigt_to_kelvin_mandel(voigt)


def irreps_to_matrix_exp_voigt(irreps: torch.Tensor) -> torch.Tensor:
    """Map log-irreps to physical-space Voigt vector via matrix exponential.

    This is the inverse of the dielectric training coordinate transform: the
    model predicts a log-tensor in irrep space; this function exponentiates it
    and returns the physical tensor in Voigt notation.
    """
    log_tensor = _CARTESIAN_RANK2.to_cartesian(irreps)
    tensor = torch.linalg.matrix_exp(0.5 * (log_tensor + log_tensor.transpose(-1, -2)))
    return tensor_to_voigt(tensor)


def elasticity_21d_to_voigt6x6(vec21: torch.Tensor) -> torch.Tensor:
    """Reconstruct a symmetric 6x6 Voigt matrix from a 21D vector."""
    *batch, _ = vec21.shape
    C6 = torch.zeros(*batch, 6, 6, device=vec21.device, dtype=vec21.dtype)
    for idx, (i, j) in enumerate(_ELASTICITY_21_INDICES):
        C6[..., i, j] = vec21[..., idx]
        if i != j:
            C6[..., j, i] = vec21[..., idx]
    return C6


def voigt6x6_to_21d(C6: torch.Tensor) -> torch.Tensor:
    """Extract the 21D vector from a symmetric 6x6 Voigt matrix."""
    return torch.stack([C6[..., i, j] for i, j in _ELASTICITY_21_INDICES], dim=-1)


def voigt6x6_to_tensor(C6: torch.Tensor) -> torch.Tensor:
    """Expand a symmetric 6x6 Voigt matrix to a 3x3x3x3 elasticity tensor."""
    *batch, _, _ = C6.shape
    C = torch.zeros(*batch, 3, 3, 3, 3, device=C6.device, dtype=C6.dtype)
    for i, j, k, l in itertools.product(range(3), repeat=4):
        C[..., i, j, k, l] = C6[..., _VOIGT_PAIR[(i, j)], _VOIGT_PAIR[(k, l)]]
    return C


def tensor_to_voigt6x6(C: torch.Tensor) -> torch.Tensor:
    """Contract a 3x3x3x3 elasticity tensor to a symmetric 6x6 Voigt matrix."""
    *batch, _, _, _, _ = C.shape
    C6 = torch.zeros(*batch, 6, 6, device=C.device, dtype=C.dtype)
    for I, J in itertools.product(range(6), repeat=2):
        pairs_I = [p for p, idx in _VOIGT_PAIR.items() if idx == I]
        pairs_J = [p for p, idx in _VOIGT_PAIR.items() if idx == J]
        vals = []
        for i, j in pairs_I:
            for k, l in pairs_J:
                vals.append(C[..., i, j, k, l])
        C6[..., I, J] = torch.stack(vals, dim=-1).mean(dim=-1)
    return C6


def elasticity_21d_to_tensor(vec21: torch.Tensor) -> torch.Tensor:
    """Convert a 21D elasticity vector to a 3x3x3x3 tensor."""
    C6 = elasticity_21d_to_voigt6x6(vec21)
    return voigt6x6_to_tensor(C6)


def elasticity_tensor_to_21d(C: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3x3x3 elasticity tensor to a 21D vector."""
    C6 = tensor_to_voigt6x6(C)
    return voigt6x6_to_21d(C6)


def elasticity_21d_to_irreps(vec21: torch.Tensor) -> torch.Tensor:
    """Convert a 21D elasticity vector to e3nn irreps."""
    C = elasticity_21d_to_tensor(vec21)
    return _CARTESIAN_RANK4.from_cartesian(C)


def irreps_to_elasticity_21d(irreps: torch.Tensor) -> torch.Tensor:
    """Convert e3nn irreps to a 21D elasticity vector."""
    C = _CARTESIAN_RANK4.to_cartesian(irreps)
    return elasticity_tensor_to_21d(C)
