import torch

from evaluation.ensemble import (
    combine_ensemble_moments,
    ensemble_nll,
    finite_mixture_nll,
    sample_ensemble,
    variogram_score,
)


def test_total_covariance_separates_aleatoric_and_epistemic_terms():
    means = torch.tensor([[[0.0, 0.0]], [[2.0, 0.0]]])
    scales = torch.eye(2).reshape(1, 1, 2, 2).expand(2, 1, 2, 2).clone()
    result = combine_ensemble_moments(means, scales)
    torch.testing.assert_close(result["mean"], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(
        result["aleatoric_covariance"], torch.eye(2).unsqueeze(0)
    )
    expected_epistemic = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    torch.testing.assert_close(result["epistemic_covariance"], expected_epistemic)
    torch.testing.assert_close(
        result["total_covariance"], torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    )


def test_mixture_nll_and_variogram_are_finite():
    torch.manual_seed(2)
    means = torch.randn(3, 5, 2)
    scales = torch.eye(2).reshape(1, 1, 2, 2).expand(3, 5, 2, 2).clone()
    target = torch.randn(5, 2)
    assert torch.isfinite(ensemble_nll(means, scales, target))
    samples = sample_ensemble(means, scales, num_samples=32)
    score = variogram_score(samples, target)
    assert torch.isfinite(score) and score >= 0


def test_ensemble_nll_delegates_to_uniform_finite_mixture():
    torch.manual_seed(8)
    means = torch.randn(2, 7, 3, dtype=torch.float64)
    scales = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3).expand(2, 7, 3, 3)
    target = torch.randn(7, 3, dtype=torch.float64)
    expected = finite_mixture_nll(
        means,
        scales,
        target,
        distribution="student_t",
        student_t_dof=5.0,
    )
    actual = ensemble_nll(
        means,
        scales,
        target,
        distribution="student_t",
        student_t_dof=5.0,
    )
    torch.testing.assert_close(actual, expected)


def test_finite_mixture_accepts_invariant_sample_weights():
    means = torch.tensor([[[0.0]], [[2.0]]])
    scales = torch.ones(2, 1, 1, 1)
    target = torch.tensor([[0.1]])
    left = finite_mixture_nll(
        means,
        scales,
        target,
        distribution="student_t",
        weights=torch.tensor([[0.9], [0.1]]),
    )
    right = finite_mixture_nll(
        means,
        scales,
        target,
        distribution="student_t",
        weights=torch.tensor([[0.1], [0.9]]),
    )
    assert left < right
