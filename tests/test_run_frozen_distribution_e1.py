import json
import sys

import torch

from compatibility.e3nn import o3
from equivcompiler import FullCovariance
from representations import O3IrrepsSpec
from scripts.itop_reproducibility import sha256_file
from scripts.run_frozen_distribution_e1 import _parse_diagnostic_splits, main


def test_diagnostic_split_selector_defaults_to_all_and_accepts_subset():
    assert _parse_diagnostic_splits(None) == ("train", "val", "test")
    assert _parse_diagnostic_splits(("val", "test")) == ("val", "test")


def test_fixed_e1_runner_publishes_complete_artifacts(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    run = tmp_path / "run"
    cache.mkdir()
    output = O3IrrepsSpec("1o")
    family = FullCovariance().compile(output)
    feature_irreps = "1x1o+1x0e+1x2e"
    splits = {}
    for offset, (split, count) in enumerate((("train", 24), ("val", 21), ("test", 23))):
        generator = torch.Generator().manual_seed(20 + offset)
        payload = {
            "features": torch.randn(count, 9, generator=generator),
            "mean": torch.randn(count, 3, generator=generator),
            "params": 0.1 * torch.randn(count, 6, generator=generator),
            "target": torch.randn(count, 3, generator=generator),
            "sample_id": torch.arange(count) + 100 * offset,
        }
        path = cache / f"{split}.pt"
        torch.save(payload, path)
        splits[split] = {"count": count, "sha256": sha256_file(path)}
    metadata = {
        "schema_version": 1,
        "source_checkpoint": {"path": "synthetic", "sha256": "test"},
        "feature_irreps": feature_irreps,
        "output_irreps": "1o",
        "parameter_irreps": str(family.parameter_irreps),
        "parameter_count": family.parameter_count,
        "operator_family": family.as_dict(),
        "spd_map": {"kind": "matrix_exp", "representation_metric": "none"},
        "student_t_dof": 5.0,
        "splits": splits,
    }
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_distribution_e1",
            "--cache_dir",
            str(cache),
            "--run_dir",
            str(run),
            "--variant",
            "fixed",
            "--device",
            "cpu",
            "--batch_size",
            "8",
        ],
    )
    main()
    required = {
        "best_model.pt",
        "history.json",
        "protocol.json",
        "diagnostics.json",
        "environment.json",
        "manifest.json",
        "predictions_train.pt",
        "predictions_val.pt",
        "predictions_test.pt",
    }
    assert required.issubset(path.name for path in run.iterdir())
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["selection"]["selected_epoch"] == 0
    assert protocol["frozen"]["mean"] and protocol["frozen"]["features"]
    diagnostics = json.loads((run / "diagnostics.json").read_text(encoding="utf-8"))
    assert (
        diagnostics["test"]["nll_semantics"] == "exact_single_student_t_log_likelihood"
    )


def test_repair_variants_publish_exact_family_artifacts(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    output = O3IrrepsSpec("1o")
    family = FullCovariance().compile(output)
    feature_irreps = "1x1o+1x0e+1x2e"
    projection = o3.Linear(
        o3.Irreps(feature_irreps), o3.Irreps(str(family.parameter_irreps))
    )
    projection_path = cache / "operator_projection.pt"
    torch.save(projection.state_dict(), projection_path)
    splits = {}
    for offset, (split, count) in enumerate((('train', 24), ('val', 21), ('test', 23))):
        generator = torch.Generator().manual_seed(120 + offset)
        payload = {
            "features": torch.randn(count, 9, generator=generator),
            "mean": torch.randn(count, 3, generator=generator),
            "params": 0.1 * torch.randn(count, 6, generator=generator),
            "target": torch.randn(count, 3, generator=generator),
            "sample_id": torch.arange(count) + 100 * offset,
        }
        path = cache / f"{split}.pt"
        torch.save(payload, path)
        splits[split] = {"count": count, "sha256": sha256_file(path)}
    metadata = {
        "schema_version": 1,
        "source_checkpoint": {"path": "synthetic", "sha256": "test"},
        "feature_irreps": feature_irreps,
        "output_irreps": "1o",
        "parameter_irreps": str(family.parameter_irreps),
        "parameter_count": family.parameter_count,
        "operator_family": family.as_dict(),
        "spd_map": {"kind": "matrix_exp", "representation_metric": "none"},
        "student_t_dof": 5.0,
        "operator_projection": {
            "path": projection_path.name,
            "sha256": sha256_file(projection_path),
        },
        "splits": splits,
    }
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for variant in (
        "global_nu",
        "conditional_scale",
        "conditional_nu",
        "shared_mean_mixture",
    ):
        run = tmp_path / variant
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_frozen_distribution_e1",
                "--cache_dir",
                str(cache),
                "--run_dir",
                str(run),
                "--variant",
                variant,
                "--device",
                "cpu",
                "--batch_size",
                "8",
                "--max_epochs",
                "1",
            ],
        )
        main()
        diagnostics = json.loads(
            (run / "diagnostics.json").read_text(encoding="utf-8")
        )
        expected = (
            "exact_finite_mixture_logsumexp"
            if variant == "shared_mean_mixture"
            else "exact_single_student_t_log_likelihood"
        )
        assert diagnostics["test"]["nll_semantics"] == expected
