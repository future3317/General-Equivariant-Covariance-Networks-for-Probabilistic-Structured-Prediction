"""Tests for coordinate conversions between physical tensors and e3nn irreps."""

import torch

from data.tensor_conversions import (
    km_to_irreps,
    irreps_to_km,
    voigt_to_irreps,
    irreps_to_voigt,
    elasticity_21d_to_irreps,
    irreps_to_elasticity_21d,
    irreps_to_matrix_exp_voigt,
)
from voigt_utils import kelvin_mandel_to_voigt
from matrix_log_transform import matrix_exponential_transform


RTOL = 1e-5


def test_km_irreps_round_trip():
    """Kelvin-Mandel vector -> irreps -> Kelvin-Mandel vector."""
    km = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.5, 0.4, 0.3],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        ]
    )
    irreps = km_to_irreps(km)
    km_back = irreps_to_km(irreps)
    assert torch.allclose(km, km_back, rtol=RTOL)


def test_voigt_irreps_round_trip():
    """Voigt vector -> irreps -> Voigt vector."""
    voigt = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.5, 0.4, 0.3],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        ]
    )
    irreps = voigt_to_irreps(voigt)
    voigt_back = irreps_to_voigt(irreps)
    assert torch.allclose(voigt, voigt_back, rtol=RTOL)


def test_irreps_matrix_exp_voigt_matches_km_transform():
    """irreps_to_matrix_exp_voigt must agree with the original KM log-exp pipeline."""
    km = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.5, 0.4, 0.3],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        ]
    )
    irreps = km_to_irreps(km)
    pred_voigt = irreps_to_matrix_exp_voigt(irreps)
    target_voigt = matrix_exponential_transform(kelvin_mandel_to_voigt(km))
    assert torch.allclose(pred_voigt, target_voigt, rtol=RTOL)


def test_elasticity_21d_irreps_round_trip():
    """21D elasticity vector -> irreps -> 21D elasticity vector."""
    vec21 = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
            ],
            [
                2.0,
                1.5,
                1.0,
                0.5,
                0.4,
                0.3,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ]
    )
    irreps = elasticity_21d_to_irreps(vec21)
    vec21_back = irreps_to_elasticity_21d(irreps)
    assert torch.allclose(vec21, vec21_back, rtol=RTOL, atol=1e-6)


def test_elasticity_irreps_major_symmetry():
    """The reconstructed 21D vector must build a major-symmetric 6x6 Voigt matrix."""
    from data.tensor_conversions import elasticity_21d_to_voigt6x6

    vec21 = torch.randn(4, 21)
    C6 = elasticity_21d_to_voigt6x6(vec21)
    assert torch.allclose(C6, C6.transpose(-1, -2), rtol=RTOL)
