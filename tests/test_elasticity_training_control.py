from __future__ import annotations

from argparse import Namespace

from data.elasticity_dataset import deterministic_subset_indices
from scripts.run_elasticity_study import (
    assign_devices,
    pilot_gate,
    stage_budget,
    study_arm_arguments,
)
from scripts.train_elasticity import (
    build_elasticity_model,
    elasticity_arm_configuration,
)


def _args(**overrides):
    values = {
        "arm": "full_student_t",
        "hidden_dim": 8,
        "lmax": 2,
        "num_layers": 1,
        "num_basis": 4,
        "rank": 2,
        "parameter_budget": 256,
        "student_t_dof": 5.0,
        "atom_features": "manual",
        "tp_backend": "e3nn",
        "cueq_method": "naive",
    }
    values.update(overrides)
    return Namespace(**values)


def test_seeded_subset_indices_are_reproducible_and_seed_sensitive():
    first = deterministic_subset_indices(100, 12, seed=42)
    repeated = deterministic_subset_indices(100, 12, seed=42)
    changed = deterministic_subset_indices(100, 12, seed=43)

    assert first == repeated
    assert first != changed
    assert first == sorted(first)
    assert len(set(first)) == 12


def test_elasticity_arm_configuration_is_minimal_and_explicit():
    assert elasticity_arm_configuration("deterministic") == {
        "objective": "deterministic",
        "covariance": None,
    }
    assert elasticity_arm_configuration("low_rank_student_t") == {
        "objective": "student_t",
        "covariance": "low_rank",
    }
    assert elasticity_arm_configuration("full_student_t") == {
        "objective": "student_t",
        "covariance": "full",
    }


def test_model_builder_uses_mean_only_control_and_compiled_probabilistic_arms():
    deterministic, deterministic_schema = build_elasticity_model(
        _args(arm="deterministic")
    )
    low_rank, low_rank_schema = build_elasticity_model(
        _args(arm="low_rank_student_t")
    )
    full, full_schema = build_elasticity_model(_args(arm="full_student_t"))

    assert deterministic.distribution is None
    assert deterministic.spd_map is None
    assert deterministic_schema["kind"] == "deterministic_mean"

    assert low_rank.distribution.__class__.__name__ == "StudentTNLL"
    assert low_rank_schema["family"]["kind"] == "low_rank_plus_isotropic"
    assert low_rank_schema["family"]["parameter_count"] == 21 * 2 + 1

    assert full.distribution.__class__.__name__ == "StudentTNLL"
    assert full_schema["family"]["kind"] == "full_covariance"
    assert full_schema["family"]["parameter_count"] > low_rank_schema["family"][
        "parameter_count"
    ]
    assert full_schema["covariance_representation"]["highest_angular_momentum"] == 8


def test_study_arm_arguments_change_only_the_declared_family():
    assert study_arm_arguments("deterministic") == ["--arm", "deterministic"]
    assert study_arm_arguments("low_rank_student_t") == [
        "--arm",
        "low_rank_student_t",
    ]
    assert study_arm_arguments("full_student_t") == ["--arm", "full_student_t"]


def test_pilot_gate_requires_all_arms_loss_improvement_and_resource_contract():
    passing = {
        arm: {
            "required_artifacts_complete": True,
            "finite": True,
            "first_validation_criterion": 5.0,
            "best_validation_criterion": 4.0,
            "peak_allocated_gib": 2.0,
            "schema_valid": True,
        }
        for arm in ("deterministic", "low_rank_student_t", "full_student_t")
    }
    assert pilot_gate(passing)["passed"] is True

    passing["full_student_t"]["best_validation_criterion"] = 5.0
    failed = pilot_gate(passing)
    assert failed["passed"] is False
    assert "validation criterion did not improve" in failed["reasons"][0]


def test_device_assignment_is_deterministic_and_balanced():
    jobs = [(42, arm) for arm in ("deterministic", "low_rank_student_t", "full_student_t")]
    assert assign_devices(jobs, ["cuda:0", "cuda:1"]) == [
        (42, "deterministic", "cuda:0"),
        (42, "low_rank_student_t", "cuda:1"),
        (42, "full_student_t", "cuda:0"),
    ]


def test_stage_budget_has_fast_defaults_and_allows_stricter_formal_cap():
    assert stage_budget("pilot", None, None) == (6, 2)
    assert stage_budget("formal", None, None) == (30, 5)
    assert stage_budget("formal", 12, 3) == (12, 3)
