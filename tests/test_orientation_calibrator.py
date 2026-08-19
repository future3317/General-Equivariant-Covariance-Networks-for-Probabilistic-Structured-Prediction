import torch

from compatibility.e3nn import o3
from models import EquivariantIsospectralOrientationCalibrator


def test_eioc_is_identity_at_zero_initialization_and_preserves_spectrum():
    torch.manual_seed(5)
    calibrator = EquivariantIsospectralOrientationCalibrator(
        "4x0e + 2x1o + 2x2e", "0e + 2e"
    )
    features = torch.randn(6, o3.Irreps("4x0e + 2x1o + 2x2e").dim)
    raw = torch.randn(6, 6, 6)
    scale = raw @ raw.transpose(-1, -2) + 0.2 * torch.eye(6)
    calibrated = calibrator(features, scale)
    torch.testing.assert_close(calibrated, scale, atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        for parameter in calibrator.coefficient_head.parameters():
            parameter.normal_(std=0.01)
    calibrated = calibrator(features, scale)
    torch.testing.assert_close(
        torch.linalg.eigvalsh(calibrated),
        torch.linalg.eigvalsh(scale),
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.all(torch.linalg.eigvalsh(calibrated) > 0)


def test_eioc_compiles_expected_skew_representation():
    calibrator = EquivariantIsospectralOrientationCalibrator(
        "2x0e + 2x1o + 2x2e", "0e + 2e"
    )
    assert str(calibrator.generator_irreps) == "1x1e+1x2e+1x3e"
    assert calibrator.operator_basis.operator_dim == 15
    assert calibrator.coefficient_head.depth == 1


def test_eioc_uses_target_directed_lifting_for_reachable_skew_channels():
    calibrator = EquivariantIsospectralOrientationCalibrator(
        "3x0e + 3x2e + 1x4e", "0e + 2e"
    )
    assert calibrator.coefficient_head.plan.target_irreps == o3.Irreps(
        "1x1e + 1x2e + 1x3e"
    )


def test_eioc_generator_transforms_by_conjugation():
    torch.manual_seed(8)
    hidden = o3.Irreps("2x0e + 2x1o + 2x2e")
    output = o3.Irreps("0e + 2e")
    calibrator = EquivariantIsospectralOrientationCalibrator(hidden, output, zero_init=False)
    features = torch.randn(3, hidden.dim)
    rotation = o3.rand_matrix()
    transformed_features = features @ hidden.D_from_matrix(rotation).T
    generator = calibrator.generator(features)
    transformed_generator = calibrator.generator(transformed_features)
    representation = output.D_from_matrix(rotation)
    expected = representation @ generator @ representation.T
    torch.testing.assert_close(transformed_generator, expected, atol=2e-5, rtol=2e-5)
