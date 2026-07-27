import torch

from evaluation.ensemble import (
    combine_ensemble_moments,
    ensemble_nll,
    sample_ensemble,
    variogram_score,
)


def test_total_covariance_separates_aleatoric_and_epistemic_terms():
    means = torch.tensor([[[0.0, 0.0]], [[2.0, 0.0]]])
    scales = torch.eye(2).reshape(1, 1, 2, 2).expand(2, 1, 2, 2).clone()
    result = combine_ensemble_moments(means, scales)
    torch.testing.assert_close(result["mean"], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(result["aleatoric_covariance"], torch.eye(2).unsqueeze(0))
    expected_epistemic = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    torch.testing.assert_close(result["epistemic_covariance"], expected_epistemic)
    torch.testing.assert_close(result["total_covariance"], torch.tensor([[[2.0, 0.0], [0.0, 1.0]]]))


def test_mixture_nll_and_variogram_are_finite():
    torch.manual_seed(2)
    means = torch.randn(3, 5, 2)
    scales = torch.eye(2).reshape(1, 1, 2, 2).expand(3, 5, 2, 2).clone()
    target = torch.randn(5, 2)
    assert torch.isfinite(ensemble_nll(means, scales, target))
    samples = sample_ensemble(means, scales, num_samples=32)
    score = variogram_score(samples, target)
    assert torch.isfinite(score) and score >= 0
