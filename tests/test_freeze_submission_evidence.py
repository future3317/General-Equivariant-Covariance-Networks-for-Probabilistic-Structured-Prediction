import json
from pathlib import Path

import pytest
import torch

from scripts.freeze_submission_evidence import (
    build_headline_table,
    summarize_law_arm,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _diagnostics(nll: float, energy: float) -> dict:
    return {
        "test": {
            "nll": nll,
            "energy_score": energy,
            "elliptical_falsification": {
                "whitened_second_moment_defect": 2.0,
                "radial_pit": {"ks": 0.1},
                "radius_direction_dependence": {
                    "max_abs_spearman": 0.2,
                    "max_statistic_permutation_pvalue": 0.005,
                },
            },
        },
        "val": {"nll": nll - 0.1},
    }


def _write_arm(run: Path, *, arm: str, seed: int, checkpoint: str) -> None:
    run.mkdir(parents=True)
    _write_json(
        run / "diagnostics.json",
        _diagnostics(-2.0 - seed / 100.0, 0.4),
    )
    _write_json(
        run / "protocol.json",
        {
            "variant": arm,
            "seed": seed,
            "frozen": {
                "cache_metadata_sha256": "cache-id",
                "operator_checkpoint": checkpoint,
            },
            "selection": {"selected_epoch": 3, "best_validation_nll": -2.1},
        },
    )
    _write_json(
        run / "manifest.json",
        {"selected_epoch": 3, "best_validation_nll": -2.1},
    )
    _write_json(run / "environment.json", {"source": {"commit": "abc123"}})
    prediction = {
        "mean": torch.zeros(10, 2),
        "target": torch.zeros(10, 2),
        "scale": torch.eye(2).expand(10, 2, 2).clone(),
        "nu": torch.full((10,), 5.0),
    }
    torch.save(prediction, run / "predictions_test.pt")


def test_summarize_law_arm_preserves_checkpoint_and_headline_metrics(tmp_path):
    run = tmp_path / "conditional_scale_seed_42"
    _write_arm(run, arm="conditional_scale", seed=42, checkpoint="full42.pt")

    record = summarize_law_arm(run, arm="conditional_scale", seed=42)

    assert record["seed"] == 42
    assert record["operator_checkpoint"] == "full42.pt"
    assert record["test"]["nll"] == pytest.approx(1.8378770664)
    assert record["test"]["coverage90"] == pytest.approx(1.0)
    assert record["test"]["mace"] > 0.0
    assert record["test"]["radial_ks"] == pytest.approx(0.1)
    assert record["test"]["radius_direction_pvalue"] == pytest.approx(0.005)


def test_build_headline_table_rejects_mismatched_operator_checkpoints(tmp_path):
    formal = tmp_path / "formal"
    matched = tmp_path / "matched"
    for seed in (42, 43, 44):
        _write_arm(
            formal / f"seed_{seed}",
            arm="fixed",
            seed=seed,
            checkpoint=f"full{seed}.pt",
        )
        for arm in ("global_nu", "conditional_scale", "conditional_nu"):
            _write_arm(
                matched / arm / f"seed_{seed}",
                arm=arm,
                seed=seed,
                checkpoint=f"wrong{seed}.pt" if seed == 43 else f"full{seed}.pt",
            )

    with pytest.raises(ValueError, match="operator checkpoint mismatch"):
        build_headline_table(formal, matched)
