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
    assert metrics["ensemble"]["nll_semantics"] == (
        "equal_weight_exact_finite_student_t_logsumexp"
    )
    assert len(metrics["two_member_subsets"]) == 3
    assert "mixture_projection_pit" in metrics["ensemble"]
    assert artifact["between_member_model_function_spread"].shape == (4, 45, 45)
    assert artifact["within_member_predictive_covariance"].shape == (4, 45, 45)


def test_probabilistic_ensemble_supports_two_member_pilot():
    target = torch.zeros(3, 45)
    scale = torch.eye(45).repeat(3, 1, 1)
    records = [
        {
            "mean": torch.full((3, 45), offset),
            "target": target,
            "scale": scale,
            "frame_index": torch.arange(3),
            "view_id": torch.zeros(3, dtype=torch.long),
        }
        for offset in (-0.1, 0.1)
    ]
    metrics, artifact = _evaluate_split(
        records,
        split="pilot",
        device=torch.device("cpu"),
        samples=16,
    )
    assert metrics["ensemble"]["members"] == 2
    assert len(metrics["individual_members"]) == 2
    assert len(metrics["two_member_subsets"]) == 1
    assert artifact["component_means"].shape == (2, 3, 45)
