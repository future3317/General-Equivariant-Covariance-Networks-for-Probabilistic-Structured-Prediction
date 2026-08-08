"""Matched E2 probes for error information in frozen H and raw observations."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
from scipy.stats import spearmanr

from data.frozen_distribution_features import (
    invariant_irrep_summary,
    load_frozen_distribution_cache,
)
from data.observation_descriptors import point_cloud_observation_descriptors
from evaluation import r2_score, risk_coverage_auc
from scripts.itop_reproducibility import (
    atomic_write_json,
    sha256_file,
    source_provenance,
)

PROBE_KINDS = ("H_only", "raw_observation_only", "H_plus_raw_observation")
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def _align(values: dict[str, torch.Tensor], sample_id: torch.Tensor) -> dict:
    order = torch.argsort(values["sample_id"])
    sorted_id = values["sample_id"][order]
    locations = torch.searchsorted(sorted_id, sample_id)
    if bool((locations >= len(sorted_id)).any()) or not torch.equal(
        sorted_id[locations], sample_id
    ):
        raise ValueError("raw descriptors and frozen cache sample IDs do not align")
    return {
        name: value[order[locations]]
        for name, value in values.items()
        if name != "sample_id"
    }


def _target(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    residual = payload["target"].double() - payload["mean"].double()
    return residual.reshape(-1, 15, 3).square().sum(-1).mean(-1)


def _fit_ridge(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
) -> dict[str, Any]:
    train_x = train_x.double()
    validation_x = validation_x.double()
    train_y = train_y.double()
    validation_y = validation_y.double()
    x_mean = train_x.mean(0)
    x_scale = train_x.std(0, unbiased=False).clamp_min(1e-8)
    y_mean = train_y.mean()
    y_scale = train_y.std(unbiased=False).clamp_min(1e-8)
    normalized_train = (train_x - x_mean) / x_scale
    normalized_validation = (validation_x - x_mean) / x_scale
    normalized_target = (train_y - y_mean) / y_scale
    gram = normalized_train.T @ normalized_train
    right = normalized_train.T @ normalized_target
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    candidates = []
    for alpha in RIDGE_GRID:
        weight = torch.linalg.solve(gram + alpha * identity, right)
        prediction = (normalized_validation @ weight) * y_scale + y_mean
        candidates.append(
            {
                "alpha": alpha,
                "validation_mse": float((prediction - validation_y).square().mean()),
                "weight": weight,
            }
        )
    selected = min(candidates, key=lambda item: item["validation_mse"])
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "weight": selected["weight"],
        "selected_alpha": selected["alpha"],
        "validation_mse": selected["validation_mse"],
        "validation_grid": [
            {"alpha": item["alpha"], "mse": item["validation_mse"]}
            for item in candidates
        ],
    }


def _predict(model: dict[str, Any], features: torch.Tensor) -> torch.Tensor:
    normalized = (features.double() - model["x_mean"]) / model["x_scale"]
    return (normalized @ model["weight"]) * model["y_scale"] + model["y_mean"]


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.double()
    target = target.double()
    spearman = float(spearmanr(prediction.numpy(), target.numpy()).statistic)
    return {
        "mse": float((prediction - target).square().mean()),
        "rmse": float((prediction - target).square().mean().sqrt()),
        "mae": float((prediction - target).abs().mean()),
        "r2": float(r2_score(prediction, target, dim=0)),
        "spearman": spearman if math.isfinite(spearman) else 0.0,
        "risk_coverage_auc": float(risk_coverage_auc(prediction, target)),
        "oracle_risk_coverage_auc": float(risk_coverage_auc(target, target)),
    }


def _bootstrap_increment(
    h_prediction: torch.Tensor,
    combined_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int = 20260808,
    repetitions: int = 1000,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    differences = []
    for _ in range(repetitions):
        index = torch.randint(len(target), (len(target),), generator=generator)
        h_mse = (h_prediction[index] - target[index]).square().mean()
        combined_mse = (combined_prediction[index] - target[index]).square().mean()
        differences.append(combined_mse - h_mse)
    values = torch.stack(differences).double()
    return {
        "metric": "MSE(H+raw)-MSE(H)",
        "mean": float(values.mean()),
        "ci95_low": float(torch.quantile(values, 0.025)),
        "ci95_high": float(torch.quantile(values, 0.975)),
        "repetitions": repetitions,
        "seed": seed,
    }


def _save_tensor(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--side_train_geometry", type=Path, required=True)
    parser.add_argument("--side_test_geometry", type=Path, required=True)
    parser.add_argument("--top_test_geometry", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E2 probe: {args.output_dir}")

    metadata, datasets = load_frozen_distribution_cache(args.cache_dir)
    payloads = {name: dataset.payload for name, dataset in datasets.items()}
    full_descriptors = {
        "side_train": point_cloud_observation_descriptors(
            args.side_train_geometry, view_id=0
        ),
        "test": point_cloud_observation_descriptors(args.side_test_geometry, view_id=0),
        "ood": point_cloud_observation_descriptors(args.top_test_geometry, view_id=1),
    }
    raw_names = sorted(
        name
        for name in full_descriptors["side_train"]
        if name not in {"sample_id", "visible_fraction_diagnostic_only"}
    )
    split_source = {
        "train": "side_train",
        "val": "side_train",
        "test": "test",
        "ood": "ood",
    }
    features: dict[str, dict[str, torch.Tensor]] = {}
    targets = {}
    for split, payload in payloads.items():
        aligned = _align(full_descriptors[split_source[split]], payload["sample_id"])
        raw = torch.stack([aligned[name] for name in raw_names], dim=-1)
        h = invariant_irrep_summary(payload["features"], metadata["feature_irreps"])
        features[split] = {
            "H_only": h,
            "raw_observation_only": raw,
            "H_plus_raw_observation": torch.cat((h, raw), dim=-1),
        }
        targets[split] = _target(payload)

    args.output_dir.mkdir(parents=True)
    models = {}
    predictions: dict[str, dict[str, torch.Tensor]] = {split: {} for split in payloads}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for kind in PROBE_KINDS:
        model = _fit_ridge(
            features["train"][kind],
            targets["train"],
            features["val"][kind],
            targets["val"],
        )
        models[kind] = model
        metrics[kind] = {}
        for split in payloads:
            prediction = _predict(model, features[split][kind]).cpu()
            predictions[split][kind] = prediction
            metrics[kind][split] = _metrics(prediction, targets[split])

    bootstrap = {
        split: _bootstrap_increment(
            predictions[split]["H_only"],
            predictions[split]["H_plus_raw_observation"],
            targets[split],
        )
        for split in ("test", "ood")
    }
    model_path = args.output_dir / "probe_models.pt"
    prediction_path = args.output_dir / "probe_predictions.pt"
    _save_tensor(models, model_path)
    _save_tensor(
        {
            "sample_id": {
                split: payload["sample_id"] for split, payload in payloads.items()
            },
            "target_residual_severity": targets,
            "predictions": predictions,
        },
        prediction_path,
    )
    protocol = {
        "schema_version": 1,
        "study": "E2 ITOP information-sufficiency probe",
        "hypothesis": (
            "raw inference-time observation geometry adds residual-severity "
            "information beyond legal frozen-H invariants"
        ),
        "primary_target": "mean_j ||target_j-frozen_mean_j||^2",
        "probe_family": "standardized linear ridge with matched validation grid",
        "ridge_grid": list(RIDGE_GRID),
        "selection": "Side validation MSE only",
        "evaluation": {"test": "Side IID", "ood": "Top cross-view only"},
        "feature_schemas": {
            "H_only": "0e values and per-copy norms of non-scalar typed H irreps",
            "raw_observation_only": raw_names,
            "H_plus_raw_observation": "concatenation of the preceding inputs",
            "forbidden": [
                "ground-truth joint visibility",
                "pose labels as inputs",
                "label-derived descriptors",
            ],
        },
        "split_records": metadata["splits"],
        "cache_metadata_sha256": sha256_file(args.cache_dir / "metadata.json"),
        "source": source_provenance(Path(__file__).resolve().parents[1]),
    }
    result = {
        "protocol": protocol,
        "selected": {
            kind: {
                "alpha": models[kind]["selected_alpha"],
                "validation_mse": models[kind]["validation_mse"],
                "input_dimension": int(features["train"][kind].shape[-1]),
                "parameter_count": int(features["train"][kind].shape[-1] + 1),
                "validation_grid": models[kind]["validation_grid"],
            }
            for kind in PROBE_KINDS
        },
        "metrics": metrics,
        "incremental_bootstrap": bootstrap,
        "artifacts": {
            model_path.name: sha256_file(model_path),
            prediction_path.name: sha256_file(prediction_path),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    atomic_write_json(result, args.output_dir / "probe_results.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
