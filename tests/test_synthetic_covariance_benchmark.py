"""Regression gates for the controlled statistical-closure benchmark."""

import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

from experiments.synthetic_covariance_benchmark import (
    _cases,
    run_case,
    run_independent_pair,
    run_pair,
)


def _make_oracle_artifact(tmp_path: Path, family: str, seed: int = 7) -> dict:
    code = (
        "import json,sys; from pathlib import Path; "
        "from experiments.independent_teacher_oracle import "
        "OracleProtocol,build_oracle_dataset,write_oracle_artifact; "
        "p=OracleProtocol(train_contexts=4,train_replicates=3,"
        "validation_contexts=3,validation_replicates=3,"
        "test_contexts=4,test_replicates=4,calibration_draws=128,"
        "calibration_trials=32); "
        "d=build_oracle_dataset(sys.argv[2],int(sys.argv[3]),p); "
        "a=write_oracle_artifact(d,Path(sys.argv[1]),"
        "{'commit':'test','dirty':False}); print(json.dumps(a))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), family, str(seed)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


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


def test_independent_cross_family_learners_share_exact_teacher_artifact(tmp_path):
    artifact = _make_oracle_artifact(tmp_path, "full")
    rows = [
        run_independent_pair(
            artifact,
            learner,
            steps=1,
            patience=1,
            device="cpu",
        )
        for learner in ("full", "low_rank", "isotypic_block")
    ]
    assert {row["oracle_npz_sha256"] for row in rows} == {
        artifact["npz_sha256"]
    }
    assert {row["teacher_family"] for row in rows} == {"full"}
    assert {row["selection_split"] for row in rows} == {"validation"}


def test_independent_matched_row_records_validation_selection_and_gates(tmp_path):
    artifact = _make_oracle_artifact(tmp_path, "isotypic_block", seed=11)
    row = run_independent_pair(
        artifact,
        "isotypic_block",
        steps=2,
        patience=1,
        device="cpu",
    )
    assert row["role"] == "matched_recovery"
    assert row["selected_epoch"] <= row["last_epoch"]
    assert row["primary_operator"] == "scatter"
    assert set(row["gate"]) == {
        "numeric",
        "recovery",
        "coverage",
        "provenance",
        "overall",
    }
    assert row["finite"]


def test_independent_graph_artifact_rejects_non_graph_learner(tmp_path):
    artifact = _make_oracle_artifact(tmp_path, "graph_precision")
    with pytest.raises(ValueError, match="representation contract"):
        run_independent_pair(
            artifact,
            "full",
            steps=1,
            patience=1,
            device="cpu",
        )


def test_independent_evaluation_does_not_build_an_equivariance_graph(tmp_path):
    artifact = _make_oracle_artifact(tmp_path, "isotypic_block", seed=19)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_independent_pair(
            artifact,
            "isotypic_block",
            steps=1,
            patience=1,
            device="cpu",
        )
    messages = [str(item.message) for item in caught]
    assert not any("requires_grad=True" in message for message in messages)


def test_independent_cli_writes_formal_contract_and_row_accounting(tmp_path):
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "experiments.synthetic_covariance_benchmark",
        "--teacher-backend",
        "independent_numpy",
        "--families",
        "full,low_rank,isotypic_block,graph_precision",
        "--seeds",
        "0",
        "--contexts",
        "3",
        "--replicates",
        "2",
        "--validation-contexts",
        "2",
        "--validation-replicates",
        "2",
        "--test-contexts",
        "3",
        "--test-replicates",
        "2",
        "--calibration-draws",
        "64",
        "--calibration-trials",
        "16",
        "--steps",
        "1",
        "--patience",
        "1",
        "--device",
        "cpu",
        "--oracle-dir",
        str(tmp_path / "oracles"),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["kind"] == "independent_numpy_teacher_scatter_recovery"
    assert result["contract"]["teacher_side"] == "numpy_scipy_only"
    assert result["contract"]["learner_side"] == "public_compiler"
    assert result["formal_gate"]["required_matched_rows"] == 4
    assert len(result["matched_rows"]) == 4
    assert len(result["diagnostic_rows"]) == 6
    for row in result["matched_rows"]:
        checkpoint = Path(row["checkpoint_path"])
        predictions = Path(row["prediction_path"])
        assert checkpoint.is_file()
        assert predictions.is_file()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row[
            "checkpoint_sha256"
        ]
        assert hashlib.sha256(predictions.read_bytes()).hexdigest() == row[
            "prediction_sha256"
        ]
        assert row["history"]
