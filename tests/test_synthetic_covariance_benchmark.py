"""Regression gates for the controlled statistical-closure benchmark."""

import pytest
import torch

from experiments.synthetic_covariance_benchmark import _cases, run_case, run_pair


@pytest.mark.parametrize("family", tuple(_cases()))
def test_compiled_family_recovery_protocol_is_finite(family):
    output, specification = _cases()[family]
    result = run_case(
        family,
        output,
        specification,
        contexts=4,
        replicates=2,
        test_contexts=3,
        steps=1,
        seed=7,
        device="cpu",
    )
    assert result["canonical_reachable"]
    assert result["active_reachable"]
    assert result["parameter_count"] > 0
    for key in (
        "test_nll",
        "covariance_relative_error",
        "equivariance_max_abs",
        "basis_change_nll_abs_error",
    ):
        assert torch.isfinite(torch.tensor(result[key])), key


def test_cross_family_recovery_keeps_teacher_and_learner_provenance():
    cases = _cases()
    output, teacher_spec = cases["full"]
    learner_output, learner_spec = cases["low_rank"]
    assert output == learner_output
    result = run_pair(
        "full",
        "low_rank",
        output,
        teacher_spec,
        learner_spec,
        contexts=4,
        replicates=2,
        test_contexts=3,
        steps=1,
        seed=11,
        device="cpu",
    )
    assert result["teacher_family"] == "full"
    assert result["learner_family"] == "low_rank"
    assert result["family_match"] is False
    assert result["teacher_parameter_count"] > result["parameter_count"]
    assert torch.isfinite(torch.tensor(result["covariance_relative_error"]))
