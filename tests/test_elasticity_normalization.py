import numpy as np
import torch

from compatibility.e3nn import o3
from data.elasticity_normalization import ElasticityTargetNormalizer
from representations import rank4_elasticity_irreps


def _normalizer() -> ElasticityTargetNormalizer:
    values = np.random.default_rng(7).normal(size=(32, 21))
    return ElasticityTargetNormalizer.fit(
        values, mode="representation_compatible"
    )


def test_representation_compatible_normalization_commutes_with_o3_action():
    normalizer = _normalizer()
    values = torch.randn(4, 21)
    rotation = o3.rand_matrix()
    irreps = o3.Irreps(rank4_elasticity_irreps())

    unnormalized = normalizer.transform(values)
    physical_rotated = normalizer.inverse(
        unnormalized @ irreps.D_from_matrix(rotation).T
    )
    lhs = normalizer.transform(physical_rotated)
    rhs = unnormalized @ irreps.D_from_matrix(rotation).T
    torch.testing.assert_close(lhs, rhs, atol=2e-5, rtol=2e-5)


def test_representation_compatible_normalization_round_trips_physical_targets():
    normalizer = _normalizer()
    values = torch.randn(5, 21)
    recovered = normalizer.inverse(normalizer.transform(values))
    torch.testing.assert_close(recovered, values, atol=2e-5, rtol=2e-5)


def test_legacy_normalization_remains_available_for_reproducibility():
    normalizer = ElasticityTargetNormalizer.fit(
        np.random.default_rng(8).normal(size=(32, 21)), mode="legacy_voigt"
    )
    values = torch.randn(3, 21)
    recovered = normalizer.inverse(normalizer.transform(values))
    torch.testing.assert_close(recovered, values, atol=2e-5, rtol=2e-5)
