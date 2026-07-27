import torch

from experiments.synthetic_repeated_covariance_recovery import run


def test_synthetic_covariance_recovery_protocol_is_finite():
    result = run(contexts=4, replicates=2, steps=2, seed=3, device="cpu")
    assert result["covariance_relative_error"] >= 0
    assert result["eigenbasis_offdiagonal_error"] >= 0
    assert torch.isfinite(torch.tensor(result["true_log_eigenvalue_mae"]))
