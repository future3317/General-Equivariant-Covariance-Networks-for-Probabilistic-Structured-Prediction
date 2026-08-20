import numpy as np

from scripts.audit_itop_topology_null import (
    _missing_required_files,
    _split_seed,
    _summarize_effects,
)


def test_legacy_true_run_only_allows_missing_provenance(tmp_path) -> None:
    required = {
        "args.json",
        "environment.json",
        "compilation.json",
        "history.json",
        "metrics.json",
        "predictions_side.pt",
        "predictions_top.pt",
        "best_model.pt",
        "train.log",
        "feature_cache.json",
    }
    for name in required:
        (tmp_path / name).touch()

    assert _missing_required_files(tmp_path, allow_missing_provenance=True) == [
        "provenance.json"
    ]


def test_legacy_args_default_split_seed_to_model_seed() -> None:
    assert _split_seed({"seed": 42}) == 42


def test_summarize_effects_reports_descriptive_topology_spread() -> None:
    summary = _summarize_effects(np.array([1.0, 2.0, 4.0, 8.0]))

    assert summary == {
        "null_minus_true_mean": 3.75,
        "null_minus_true_sample_std": np.std([1.0, 2.0, 4.0, 8.0], ddof=1),
        "median": 3.0,
        "iqr": 3.25,
        "minimum": 1.0,
        "maximum": 8.0,
        "all_positive": True,
    }
