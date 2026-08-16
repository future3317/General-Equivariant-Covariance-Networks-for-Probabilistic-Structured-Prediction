"""Tests for feature-gated statistical misspecification repairs."""


import torch

from compatibility.e3nn import o3
from distributions import FiniteMixtureStudentTNLL, StudentTNLL
from models.frozen_distribution_readout import (
    FrozenMultimodalStudentTMixture,
    FrozenSharedMeanStudentTMixture,
    GlobalDegreesOfFreedomReadout,
    InvariantDegreesOfFreedomReadout,
    InvariantMixtureLogitsReadout,
)
from spd_maps import MatrixExponentialMap
from spd_maps.base import SPDMap


class _ScalarSPDMap(SPDMap):
    """Flattened one-dimensional SPD fixture for O(3) head tests."""

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return params.exp().unsqueeze(-1)

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        return params.squeeze(-1)

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return residual.squeeze(-1).square() * (-params.squeeze(-1)).exp()


def test_finite_mixture_contract_uses_exact_law_aware_diagnostics():
    contract = FiniteMixtureStudentTNLL().predictive_law_contract(components=2)
    assert contract["log_prob"] == "exact_component_logsumexp"
    assert contract["sample"] == "categorical_component_student_t"
    assert contract["diagnostic_oracle"]["kind"] == "simulation_based_mixture_oracle"
    assert contract["diagnostic_oracle"]["moment_matched"] is False


def test_finite_mixture_one_component_matches_existing_student_t():
    torch.manual_seed(10)
    spd_map = MatrixExponentialMap()
    mean = torch.randn(6, 2, dtype=torch.float64)
    target = torch.randn_like(mean)
    raw = torch.randn(6, 2, 2, dtype=torch.float64)
    params = 0.5 * (raw + raw.transpose(-1, -2))
    legacy, _ = StudentTNLL(nu=5.0)(mean, params, target, spd_map)
    mixture, _ = FiniteMixtureStudentTNLL()(
        mean.unsqueeze(0).expand(2, -1, -1),
        params.unsqueeze(0).expand(2, -1, -1, -1),
        target,
        spd_map,
        weights=torch.full((2, 6), 0.5, dtype=torch.float64),
        nu=torch.full((2, 6), 5.0, dtype=torch.float64),
    )
    torch.testing.assert_close(mixture, legacy, rtol=1e-12, atol=1e-12)


def test_finite_mixture_uses_exact_normalized_logsumexp():
    spd_map = MatrixExponentialMap()
    means = torch.tensor([[[-1.0], [1.0]], [[1.0], [-1.0]]], dtype=torch.float64)
    params = torch.zeros(2, 2, 1, 1, dtype=torch.float64)
    target = torch.tensor([[0.25], [-0.5]], dtype=torch.float64)
    weights = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64)
    nus = torch.tensor([[3.0, 10.0], [10.0, 3.0]], dtype=torch.float64)
    _, diagnostics = FiniteMixtureStudentTNLL()(
        means, params, target, spd_map, weights=weights, nu=nus
    )
    expected = -torch.logsumexp(
        diagnostics["log_weights"] + diagnostics["component_log_prob"], dim=0
    ).mean()
    torch.testing.assert_close(diagnostics["loss"], expected)
    assert bool(torch.isfinite(diagnostics["loss"]))


def test_global_and_conditional_nu_are_invariant_and_have_finite_second_moment():
    torch.manual_seed(11)
    irreps = o3.Irreps("2x0e + 1x1o")
    features = torch.randn(7, irreps.dim, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    transformed = features @ irreps.D_from_matrix(rotation).T
    global_nu = GlobalDegreesOfFreedomReadout(initial=5.0).double()
    conditional_nu = InvariantDegreesOfFreedomReadout(
        irreps, minimum=2.05, initial=5.0
    ).double()
    torch.testing.assert_close(global_nu(features), global_nu(transformed))
    torch.testing.assert_close(
        conditional_nu(features), conditional_nu(transformed), rtol=1e-9, atol=1e-9
    )
    assert bool((global_nu(features) > 2.0).all())
    assert bool((conditional_nu(features) > 2.0).all())


def test_mixture_logits_are_invariant_and_multimodal_mean_is_weight_centered():
    torch.manual_seed(12)
    feature_irreps = o3.Irreps("2x0e + 2x1o")
    features = torch.randn(5, feature_irreps.dim, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    transformed = features @ feature_irreps.D_from_matrix(rotation).T
    logits = InvariantMixtureLogitsReadout(feature_irreps, components=2).double()
    torch.testing.assert_close(
        logits(features), logits(transformed), rtol=1e-9, atol=1e-9
    )

    model = FrozenMultimodalStudentTMixture(
        feature_irreps,
        parameter_irreps="1x0e",
        output_irreps="1x0e",
        spd_map=_ScalarSPDMap(),
        components=2,
    ).double()
    mean = torch.randn(5, 1, dtype=torch.float64)
    target = torch.randn_like(mean)
    params = torch.randn(5, 1, dtype=torch.float64)
    result = model(features, mean, params, target)
    weighted_mean = (result["weights"].unsqueeze(-1) * result["component_means"]).sum(0)
    torch.testing.assert_close(weighted_mean, mean, rtol=1e-8, atol=1e-8)
    assert bool(torch.isfinite(result["loss"]))


def test_shared_mean_mixture_keeps_mean_and_has_finite_component_scatter():
    torch.manual_seed(13)
    model = FrozenSharedMeanStudentTMixture(
        feature_irreps="1x0e + 1x1o",
        parameter_irreps="1x0e",
        output_irreps="1x0e",
        spd_map=_ScalarSPDMap(),
        components=2,
    )
    features = torch.randn(8, 4)
    mean = torch.randn(8, 1)
    target = torch.randn_like(mean)
    params = torch.randn(8, 1)
    result = model(features, mean, params, target)
    torch.testing.assert_close(result["component_means"], mean.unsqueeze(0).expand(2, -1, -1))
    assert bool((result["weights"] > 0).all())
    assert bool(torch.isfinite(result["component_scales"]).all())
    assert bool(torch.isfinite(result["loss"]))


def test_observation_descriptor_contract_rejects_label_derived_fields():
    from data.frozen_distribution_features import validate_observation_descriptors

    valid = {"radius_mean": torch.ones(3), "valid_depth_fraction": torch.ones(3)}
    checked = validate_observation_descriptors(valid, count=3)
    assert tuple(checked) == ("radius_mean", "valid_depth_fraction")
    for key in ("visible_fraction_diagnostic_only", "target", "label"):
        with torch.no_grad():
            try:
                validate_observation_descriptors({key: torch.ones(3)}, count=3)
            except ValueError:
                pass
            else:
                raise AssertionError(f"descriptor field {key!r} was accepted")


def test_component_valued_mixture_sampling_is_finite():
    from evaluation.ensemble import sample_finite_mixture

    means = torch.zeros(2, 6, 2)
    scales = torch.eye(2).reshape(1, 1, 2, 2).expand(2, 6, 2, 2).clone()
    weights = torch.tensor([[0.8] * 6, [0.2] * 6])
    nu = torch.tensor([[3.0] * 6, [7.0] * 6])
    samples = sample_finite_mixture(
        means,
        scales,
        num_samples=16,
        weights=weights,
        student_t_dof=nu,
    )
    assert samples.shape == (16, 6, 2)
    assert bool(torch.isfinite(samples).all())
