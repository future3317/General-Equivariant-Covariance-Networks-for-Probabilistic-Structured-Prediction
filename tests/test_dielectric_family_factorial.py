import argparse
import json

import pytest
import torch

from equivcompiler import FullCovariance
from representations import O3IrrepsSpec
from scripts.evaluate_dielectric_family_factorial import aggregate_factorial
from scripts.itop_reproducibility import sha256_file
from scripts.run_dielectric_family_factorial import (
    DISTRIBUTIONS,
    FAMILIES,
    FAMILY_PARAMETER_COUNTS,
    _run_arm,
    build_factorial_model,
    covariance_policy,
)


@pytest.mark.parametrize(
    ("family", "expected_count"),
    tuple(
        (family, count)
        for family, count in {
            "isotropic": 1,
            "block": 2,
            "low_rank": 13,
            "full": 21,
        }.items()
    ),
)
@pytest.mark.parametrize("distribution", ("gaussian", "student_t"))
def test_factorial_arms_compile_requested_family_and_law(
    family, expected_count, distribution
):
    metadata = {
        "feature_irreps": "3x0e+3x2e+1x4e",
        "output_irreps": "1x0e+1x2e",
        "student_t_dof": 5.0,
        "spd_map": {"representation_metric": "none"},
    }
    model, compilation = build_factorial_model(
        metadata, family, distribution, torch.device("cpu")
    )

    assert FAMILY_PARAMETER_COUNTS[family] == expected_count
    assert compilation.covariance_parameter_count == expected_count
    report = compilation.as_dict()
    assert report["objective"]["name"] == f"multivariate_{distribution}"
    assert report["execution_fidelity"]["exactness"] == "exact_for_active_family"
    assert compilation.operator_family.as_dict()["certificates"]["positivity"] == "spd"
    assert model.schema()["objective"] == f"exact_{distribution}_log_likelihood"


def test_factorial_policy_rejects_unknown_family():
    with pytest.raises(ValueError, match="unsupported factorial family"):
        covariance_policy("graph")


def test_smoke_factorial_writes_and_audits_all_arms(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    family = FullCovariance().compile(O3IrrepsSpec("0e+2e"))
    splits = {}
    for offset, (split, count) in enumerate(
        (("train", 24), ("val", 21), ("test", 23))
    ):
        generator = torch.Generator().manual_seed(100 + offset)
        payload = {
            "features": torch.randn(count, 27, generator=generator),
            "mean": torch.randn(count, 6, generator=generator),
            "params": torch.randn(count, 21, generator=generator),
            "target": torch.randn(count, 6, generator=generator),
            "sample_id": torch.arange(count),
        }
        path = cache / f"{split}.pt"
        torch.save(payload, path)
        splits[split] = {"count": count, "sha256": sha256_file(path)}
    metadata = {
        "schema_version": 1,
        "source_checkpoint": {"path": "synthetic", "sha256": "test"},
        "feature_irreps": "3x0e+3x2e+1x4e",
        "output_irreps": "0e+2e",
        "parameter_irreps": str(family.parameter_irreps),
        "parameter_count": family.parameter_count,
        "operator_family": family.as_dict(),
        "spd_map": {"kind": "matrix_exp", "representation_metric": "none"},
        "student_t_dof": 5.0,
        "splits": splits,
    }
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    root = tmp_path / "runs"
    args = argparse.Namespace(
        cache_dir=cache,
        output_root=root,
        batch_size=8,
        num_workers=0,
        max_epochs=1,
        patience=1,
        lr=5e-4,
        weight_decay=1e-5,
        stage="smoke",
    )
    for requested_family in FAMILIES:
        for distribution in DISTRIBUTIONS:
            _run_arm(
                args,
                family=requested_family,
                distribution=distribution,
                seed=42,
                device=torch.device("cpu"),
            )

    result = aggregate_factorial(root, stage="smoke", seeds=(42,))
    assert len(result["rows"]) == 8
    assert all(result["common_frozen_artifacts"].values())

    metrics_path = root / "full" / "student_t" / "seed_42" / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        aggregate_factorial(root, stage="smoke", seeds=(42,))
