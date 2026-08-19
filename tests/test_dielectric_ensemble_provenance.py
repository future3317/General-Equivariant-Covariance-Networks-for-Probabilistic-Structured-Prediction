"""Regression tests for the dielectric ensemble provenance gate."""

import json

from scripts.audit_dielectric_ensemble_provenance import audit_ensemble
from scripts.run_dielectric_deep_ensemble import DEFAULT_ENSEMBLE_SEEDS


def _write_member(root, seed, *, contract_hash="contract", commit="abc"):
    member = root / f"seed_{seed}"
    member.mkdir()
    run_spec = {
        "version": 1,
        "model": {
            "distribution": "student_t",
            "student_t_dof": 5.0,
            "covariance_parameterization": "centered_spectral_window",
        },
        "training_stage": "joint",
        "compilation": {"group": "O(3)"},
        "inference_contract_hash": contract_hash,
        "inference_contract": {"backend": "compiled_o3"},
        "provenance": {
            "source": {"commit": commit, "dirty": False},
            "dataset": {"dataset_hash": "dataset"},
        },
    }
    (member / "run_spec.json").write_text(json.dumps(run_spec))
    (member / "mean").mkdir()
    (member / "covariance").mkdir()
    return member


def test_ensemble_provenance_gate_accepts_three_matching_compiled_members(tmp_path):
    for seed in (42, 43, 44):
        _write_member(tmp_path, seed)
    metrics = {
        "kind": "finite_student_t_deep_ensemble_with_validation_calibration",
        "density_semantics": "equally_weighted_member_mixture",
        "members": [str(tmp_path / f"seed_{seed}") for seed in (42, 43, 44)],
        "fit_split": "validation",
        "eval_split": "test",
        "inference_contract_hash": "contract",
    }
    (tmp_path / "ensemble_3member_metrics.json").write_text(json.dumps(metrics))

    report = audit_ensemble(tmp_path, (42, 43, 44))

    assert report["eligible"] is True
    assert report["member_count"] == 3
    assert report["density_semantics"] == "equally_weighted_member_mixture"
    assert report["compiled_group"] == "O(3)"


def test_ensemble_provenance_gate_rejects_mixed_contracts(tmp_path):
    for seed in (42, 43, 44):
        _write_member(tmp_path, seed, contract_hash="contract" if seed != 44 else "other")

    report = audit_ensemble(tmp_path, (42, 43, 44))

    assert report["eligible"] is False
    assert "inference_contract_hash" in report["failures"]


def test_deep_ensemble_default_matches_the_three_member_contract():
    assert DEFAULT_ENSEMBLE_SEEDS == (42, 43, 44)
