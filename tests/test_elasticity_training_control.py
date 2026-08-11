from __future__ import annotations

from argparse import Namespace

from data.elasticity_dataset import deterministic_subset_indices
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
