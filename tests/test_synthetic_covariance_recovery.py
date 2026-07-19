"""Tests for the synthetic covariance-recovery experiment."""

import pytest
import torch

from gecn.experiments.synthetic_covariance_recovery import (
    SyntheticBackbone,
    SyntheticDataset,
    _make_data_object,
    evaluate,
)
from gecn import (
    O3IrrepsSpec,
    MatrixExponentialMap,
    GaussianNLL,
    EquivariantMeanHead,
    O3EquivariantSymmetricOperatorHead,
    StructuredProbabilisticPredictor,
)


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_synthetic_dataset_generates_spd(output_irreps):
    """Generated covariance matrices must be strictly SPD and finite."""
    ds = SyntheticDataset(output_irreps, num_samples=32, input_dim=8, seed=0)
    x, y, mu, A, S = ds.generate()

    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(A).all()
    assert torch.isfinite(S).all()

    eigs = torch.linalg.eigvalsh(S)
    assert eigs.min().item() > 0


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_synthetic_model_forward_backward(output_irreps):
    """The full predictor can be built and trained for one step."""
    output_spec = O3IrrepsSpec(output_irreps)
    backbone = SyntheticBackbone(input_dim=8, hidden_irreps="16x0e + 8x1o + 4x2e")
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3EquivariantSymmetricOperatorHead(backbone.irreps_out, output_spec, pool=True)

    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=MatrixExponentialMap(),
        distribution=GaussianNLL(),
    )

    ds = SyntheticDataset(output_irreps, num_samples=16, input_dim=8, seed=1)
    x, y, mu_true, A_true, S_true = ds.generate()
    data = _make_data_object(x)

    result = model(data, target=y)
    assert "loss" in result
    assert "mu" in result
    assert "scale" in result
    assert result["mu"].shape == (16, output_spec.dim)
    assert result["scale"].shape == (16, output_spec.dim, output_spec.dim)

    result["loss"].backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("output_irreps", ["1o", "0e + 2e"])
def test_evaluate_metrics_finite(output_irreps):
    """Evaluation metrics must all be finite."""
    ds = SyntheticDataset(output_irreps, num_samples=16, input_dim=8, seed=2)
    x, y, mu_true, A_true, S_true = ds.generate()

    # Use the ground truth as predictions for a sanity check.
    metrics = evaluate(mu_true, A_true, y, mu_true, A_true, S_true)
    for name, value in metrics.items():
        assert isinstance(value, float), name
        assert torch.isfinite(torch.tensor(value)), name
