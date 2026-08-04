"""Regression gates for the controlled statistical-closure benchmark."""

import pytest
import torch

from experiments.synthetic_covariance_benchmark import _cases, run_case


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
