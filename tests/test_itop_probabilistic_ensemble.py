import torch

from scripts.evaluate_itop_probabilistic_ensemble import _evaluate_split


def test_probabilistic_ensemble_uses_exact_mixture_and_separates_spread():
    target = torch.zeros(4, 45)
    scale = torch.eye(45).repeat(4, 1, 1)
    records = []
    for offset in (-0.1, 0.0, 0.1):
        records.append(
            {
                "mean": torch.full((4, 45), offset),
                "target": target,
                "scale": scale,
                "frame_index": torch.arange(4),
                "view_id": torch.zeros(4, dtype=torch.long),
            }
        )
    metrics, artifact = _evaluate_split(
        records,
        split="test",
        device=torch.device("cpu"),
        samples=16,
    )
    assert metrics["three_member_ensemble"]["nll_semantics"] == (
        "equal_weight_exact_finite_student_t_logsumexp"
    )
    assert len(metrics["two_member_subsets"]) == 3
    assert "mixture_projection_pit" in metrics["three_member_ensemble"]
    assert artifact["between_member_model_function_spread"].shape == (4, 45, 45)
    assert artifact["within_member_predictive_covariance"].shape == (4, 45, 45)
