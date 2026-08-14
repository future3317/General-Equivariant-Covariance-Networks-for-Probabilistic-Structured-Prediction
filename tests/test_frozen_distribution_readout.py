import pytest
import torch

from compatibility.e3nn import o3
from distributions import GaussianNLL, StudentTNLL
from models.frozen_distribution_readout import (
    FrozenConditionalStudentT,
    FrozenMeanScatterElliptical,
    FrozenMeanScatterStudentT,
    FrozenSymmetricStudentTMixture,
    FrozenUncertaintyBranchConditionalStudentT,
)
from equivcompiler import CenteredSpectralWindowCovariance, FeatureSpec, plan_readout
from spd_maps import IsotropicMap, MatrixExponentialMap


@pytest.mark.parametrize(
    ("distribution", "objective_type"),
    (("gaussian", GaussianNLL), ("student_t", StudentTNLL)),
)
def test_frozen_elliptical_readout_reuses_existing_objective(
    distribution, objective_type
):
    torch.manual_seed(1)
    model = FrozenMeanScatterElliptical(
        "2x0e",
        "0e",
        IsotropicMap(dim=3),
        distribution=distribution,
        student_t_dof=5.0,
    ).double()
    features = torch.randn(9, 2, dtype=torch.float64)
    mean = torch.randn(9, 3, dtype=torch.float64)
    target = torch.randn_like(mean)

    result = model(features, mean, target)
    params = model.parameter_projection(features)
    expected, _ = model.objective(mean, params, target, model.spd_map)

    assert isinstance(model.objective, objective_type)
    assert torch.isfinite(result["loss"])
    torch.testing.assert_close(result["loss"], expected)
    torch.testing.assert_close(result["params"], params)


def test_frozen_elliptical_readout_rejects_invalid_distribution_contract():
    with pytest.raises(ValueError, match="unsupported elliptical distribution"):
        FrozenMeanScatterElliptical(
            "0e", "0e", IsotropicMap(dim=1), distribution="laplace"
        )
    with pytest.raises(ValueError, match="nu > 2"):
        FrozenMeanScatterElliptical(
            "0e",
            "0e",
            IsotropicMap(dim=1),
            distribution="student_t",
            student_t_dof=2.0,
        )


def test_existing_frozen_student_t_readout_remains_compatible():
    model = FrozenMeanScatterStudentT("0e", "0e", IsotropicMap(dim=1))
    assert isinstance(model.objective, StudentTNLL)
    assert model.schema()["kind"] == "single_elliptical_fixed_nu_student_t"


def test_conditional_nu_is_invariant_and_above_finite_covariance_threshold():
    torch.manual_seed(2)
    irreps = o3.Irreps("2x0e + 1x1o")
    model = FrozenConditionalStudentT(irreps, MatrixExponentialMap()).double()
    features = torch.randn(7, irreps.dim, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    transformed = features @ irreps.D_from_matrix(rotation).T
    nu = model.nu_readout(features)
    transformed_nu = model.nu_readout(transformed)
    torch.testing.assert_close(nu, transformed_nu)
    assert bool((nu > 2.0).all())

    mean = torch.randn(7, 3, dtype=torch.float64)
    target = torch.randn_like(mean)
    raw = torch.randn(7, 3, 3, dtype=torch.float64)
    params = 0.5 * (raw + raw.transpose(-1, -2))
    result = model(features, mean, params, target)
    torch.testing.assert_close(result["params"], params)


def test_symmetric_mixture_means_are_equivariant_and_preserve_frozen_mean():
    torch.manual_seed(3)
    feature_irreps = o3.Irreps("2x0e + 2x1o")
    output_irreps = o3.Irreps("1o")
    model = FrozenSymmetricStudentTMixture(
        feature_irreps,
        output_irreps,
        MatrixExponentialMap(),
    ).double()
    features = torch.randn(5, feature_irreps.dim, dtype=torch.float64)
    mean = torch.randn(5, output_irreps.dim, dtype=torch.float64)
    target = torch.randn_like(mean)
    raw = torch.randn(5, 3, 3, dtype=torch.float64)
    params = 0.5 * (raw + raw.transpose(-1, -2))
    result = model(features, mean, params, target)
    torch.testing.assert_close(result["component_means"].mean(0), mean)
    torch.testing.assert_close(
        result["weights"], torch.full((2, 5), 0.5, dtype=torch.float64)
    )
    assert torch.isfinite(result["loss"])

    rotation = o3.rand_matrix(dtype=torch.float64)
    transformed_features = features @ feature_irreps.D_from_matrix(rotation).T
    transformed_delta = model.offset_readout(transformed_features)
    expected_delta = model.offset_readout(features) @ rotation.T
    torch.testing.assert_close(transformed_delta, expected_delta, atol=1e-9, rtol=1e-9)


def test_mixture_loss_has_finite_offset_gradients():
    torch.manual_seed(5)
    model = FrozenSymmetricStudentTMixture(
        "3x0e + 1x1o",
        "1o",
        MatrixExponentialMap(),
        offset_initialization=1e-2,
    )
    features = torch.randn(12, 6)
    mean = torch.randn(12, 3)
    target = torch.randn(12, 3)
    params = torch.zeros(12, 3, 3)
    result = model(features, mean, params, target)
    result["loss"].backward()
    gradients = [parameter.grad for parameter in model.offset_readout.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
def test_uncertainty_branch_starts_as_frozen_params_and_keeps_mean_out_of_graph():
    torch.manual_seed(7)
    feature_irreps = o3.Irreps("2x0e + 1x1o")
    parameter_irreps = o3.Irreps("1x0e + 1x2e")
    model = FrozenUncertaintyBranchConditionalStudentT(
        feature_irreps, parameter_irreps, plan_readout(FeatureSpec.from_irreps(feature_irreps), output='1o', covariance=CenteredSpectralWindowCovariance(), distribution='student_t', student_t_dof=5.0).compilation.build_spd_map()
    ).double()
    features = torch.randn(6, feature_irreps.dim, dtype=torch.float64)
    base_params = torch.randn(6, parameter_irreps.dim, dtype=torch.float64)
    mean = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(mean)
    result = model(features, mean, base_params, target)
    torch.testing.assert_close(result["params"], base_params)
    assert not result["mean"].requires_grad
    result["loss"].backward()
    assert mean.grad is None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_uncertainty_branch_residual_is_equivariant():
    torch.manual_seed(8)
    feature_irreps = o3.Irreps("2x0e + 1x1o")
    model = FrozenUncertaintyBranchConditionalStudentT(
        "2x0e + 1x1o", "1x0e + 1x2e", plan_readout(FeatureSpec.from_irreps(feature_irreps), output="1o", covariance=CenteredSpectralWindowCovariance(), distribution="student_t", student_t_dof=5.0).compilation.build_spd_map()
    ).double()
    features = torch.randn(4, feature_irreps.dim, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    transformed = features @ feature_irreps.D_from_matrix(rotation).T
    residual = model.residual_projection(features)
    transformed_residual = model.residual_projection(transformed)
    expected = residual @ model.parameter_irreps.D_from_matrix(rotation).T
    torch.testing.assert_close(transformed_residual, expected, atol=1e-9, rtol=1e-9)
