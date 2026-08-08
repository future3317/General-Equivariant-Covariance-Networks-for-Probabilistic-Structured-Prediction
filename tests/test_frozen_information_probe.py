import torch

from compatibility.e3nn import o3
from data.frozen_distribution_features import invariant_irrep_summary
from scripts.run_itop_information_probe import _fit_ridge, _predict


def test_invariant_irrep_summary_is_rotation_invariant():
    irreps = o3.Irreps("2x0e + 2x1o + 1x2e")
    features = torch.randn(8, irreps.dim, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    transformed = features @ irreps.D_from_matrix(rotation).T
    torch.testing.assert_close(
        invariant_irrep_summary(features, irreps),
        invariant_irrep_summary(transformed, irreps),
        rtol=1e-10,
        atol=1e-10,
    )


def test_ridge_selection_uses_validation_error():
    train_x = torch.arange(20, dtype=torch.float64).reshape(-1, 1)
    train_y = 3.0 * train_x[:, 0] + 2.0
    validation_x = torch.tensor([[20.0], [21.0]], dtype=torch.float64)
    validation_y = 3.0 * validation_x[:, 0] + 2.0
    model = _fit_ridge(train_x, train_y, validation_x, validation_y)
    prediction = _predict(model, validation_x)
    assert torch.mean((prediction - validation_y).square()) < 1e-5
