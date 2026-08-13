"""Audit and summarize the ITOP reviewer controls from saved artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import torch
from e3nn import o3

from scripts.train_itop import _build_model

REQUIRED = (
    "args.json",
    "environment.json",
    "compilation.json",
    "history.json",
    "metrics.json",
    "predictions_side.pt",
    "predictions_top.pt",
    "best_model.pt",
    "train.log",
)
METRICS = (
    "nll",
    "energy_score_m",
    "joint_mace",
    "frame_risk_coverage_auc_cm",
)


def _read_run(path: Path) -> dict:
    missing = [name for name in REQUIRED if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing {missing}")
    args = json.loads((path / "args.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    finite = True
    minimum_eigenvalue = {}
    model = None
    for split in ("side", "top"):
        artifact = torch.load(
            path / f"predictions_{split}.pt", map_location="cpu", weights_only=True
        )
        for name, value in artifact.items():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                finite = False
                raise FloatingPointError(f"{path}/{split}:{name} is non-finite")
        if "scale" in artifact:
            scale = artifact["scale"].to(torch.float64)
            minimum_eigenvalue[split] = float(torch.linalg.eigvalsh(scale).amin())
        else:
            if model is None:
                model, _ = _build_model(argparse.Namespace(**args))
                model.spd_map = model.spd_map.to(torch.float64)
            minimum = math.inf
            params = artifact["params"].to(torch.float64)
            for chunk in params.split(64):
                precision = model.spd_map.precision(chunk)
                minimum = min(
                    minimum,
                    float(torch.linalg.eigvalsh(precision).amax().reciprocal()),
                )
            minimum_eigenvalue[split] = minimum
    return {
        "path": str(path),
        "args": args,
        "metrics": metrics,
        "finite": finite,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


def _coverage(metrics: dict, level: float) -> float:
    marginal = metrics["per_joint_marginal_coverage"]
    index = marginal["levels"].index(level)
    return mean(marginal["coverage_by_level_and_joint"][index])


def _run_metrics(run: dict) -> dict:
    result = {}
    for split in ("side", "top"):
        values = {name: float(run["metrics"][split][name]) for name in METRICS}
        values["coverage90"] = _coverage(run["metrics"][split], 0.9)
        values["coverage95"] = _coverage(run["metrics"][split], 0.95)
        values["minimum_eigenvalue"] = run["minimum_eigenvalue"][split]
        result[split] = values
    return result


def _aggregate(runs: list[dict]) -> dict:
    records = [_run_metrics(run) for run in runs]
    output = {"seeds": [run["args"]["seed"] for run in runs], "per_seed": records}
    for split in ("side", "top"):
        output[split] = {}
        for metric in (*METRICS, "coverage90", "coverage95", "minimum_eigenvalue"):
            values = [record[split][metric] for record in records]
            output[split][metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
    return output


def _rotation_audit(run: dict, feature_cache: Path, *, rotations: int) -> dict:
    namespace = argparse.Namespace(**run["args"])
    model, compilation = _build_model(namespace)
    if compilation is not None:
        raise ValueError("fixed-coordinate audit must bypass the compiler")
    checkpoint = torch.load(
        Path(run["path"]) / "best_model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    features = torch.load(feature_cache, map_location="cpu", weights_only=True)[
        "features"
    ][:512]
    batch = torch.arange(features.shape[0])
    feature_irreps = model.backbone.irreps_out
    output_irreps = model.output_spec.irreps
    with torch.no_grad():
        base = model.forward_from_features(features, batch, return_scale=True)
    errors = []
    base_scale = base["scale"].to(torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260813)
        for _ in range(rotations):
            rotation = o3.rand_matrix()
            feature_action = feature_irreps.D_from_matrix(rotation)
            output_action = output_irreps.D_from_matrix(rotation).to(torch.float64)
            with torch.no_grad():
                transformed = model.forward_from_features(
                    features @ feature_action.T, batch, return_scale=True
                )
            expected = output_action @ base_scale @ output_action.T
            numerator = torch.linalg.matrix_norm(
                transformed["scale"].to(torch.float64) - expected,
                ord="fro",
                dim=(-2, -1),
            )
            denominator = torch.linalg.matrix_norm(
                expected, ord="fro", dim=(-2, -1)
            )
            errors.extend((numerator / denominator.clamp_min(1e-15)).tolist())
    return {
        "sample_count": features.shape[0],
        "rotation_count": rotations,
        "relative_frobenius_mean": mean(errors),
        "relative_frobenius_max": max(errors),
        "all_finite": all(math.isfinite(value) for value in errors),
    }


def _parse_runs(values: list[str]) -> list[Path]:
    return [Path(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", action="append", required=True)
    parser.add_argument("--shuffled", action="append", required=True)
    parser.add_argument("--no-edge", required=True)
    parser.add_argument("--fixed-coordinate", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--rotations", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    true_runs = [_read_run(path) for path in _parse_runs(args.true)]
    shuffled_runs = [_read_run(path) for path in _parse_runs(args.shuffled)]
    no_edge = _read_run(Path(args.no_edge))
    fixed = _read_run(Path(args.fixed_coordinate))
    split_seeds = {run["args"].get("split_seed", run["args"]["seed"]) for run in shuffled_runs}
    if split_seeds != {42}:
        raise ValueError(f"shuffled runs do not share split seed 42: {split_seeds}")
    output = {
        "protocol": {
            "selection": "Side validation only; Top evaluation only",
            "true_graph_effective_split_seeds": [
                run["args"].get("split_seed", run["args"]["seed"])
                for run in true_runs
            ],
            "shuffled_graph_split_seed": 42,
            "fixed_nu": 5.0,
            "all_predictions_finite": all(
                run["finite"] for run in (*true_runs, *shuffled_runs, no_edge, fixed)
            ),
        },
        "true_graph": _aggregate(true_runs),
        "shuffled_graph": _aggregate(shuffled_runs),
        "no_edge_seed42": _run_metrics(no_edge),
        "fixed_coordinate_seed42": _run_metrics(fixed),
        "fixed_coordinate_rotation_audit": _rotation_audit(
            fixed, Path(args.feature_cache), rotations=args.rotations
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
