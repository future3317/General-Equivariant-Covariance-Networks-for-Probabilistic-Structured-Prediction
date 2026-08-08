import torch

from evaluation.ensemble import (
    combine_ensemble_moments,
    energy_score_from_samples,
    ensemble_nll,
    finite_mixture_nll,
    sample_ensemble,
    variogram_score,
)


def test_chunked_energy_score_matches_dense_value_and_gradient():
    torch.manual_seed(7)
    target = torch.randn(5, 3, dtype=torch.float64)
    chunked_samples = torch.randn(7, 5, 3, dtype=torch.float64, requires_grad=True)
    dense_samples = chunked_samples.detach().clone().requires_grad_(True)

    chunked = energy_score_from_samples(
        chunked_samples,
        target,
        sample_chunk_size=2,
        observation_chunk_size=3,
    )
    first = torch.linalg.vector_norm(dense_samples - target.unsqueeze(0), dim=-1).mean(
        0
    )
    pairwise = torch.linalg.vector_norm(
        dense_samples[:, None] - dense_samples[None, :], dim=-1
    ).mean((0, 1))
    dense = (first - 0.5 * pairwise).mean()

    chunked_gradient = torch.autograd.grad(chunked, chunked_samples)[0]
    dense_gradient = torch.autograd.grad(dense, dense_samples)[0]
    torch.testing.assert_close(chunked, dense, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(chunked_gradient, dense_gradient, rtol=1e-12, atol=1e-12)


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
