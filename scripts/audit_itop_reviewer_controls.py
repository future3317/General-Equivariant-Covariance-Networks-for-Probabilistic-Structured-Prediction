"""Audit and summarize the ITOP reviewer controls from saved artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import torch
from e3nn import o3

from evaluation import empirical_coverage
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


def _rotation_summary(records: list[dict[str, float]], *, rotations: int) -> dict:
    """Aggregate coordinate-stress observations without hiding failures."""
    if not records:
        raise ValueError("rotation audit produced no observations")
    required = {"equivariance_error", "nll_delta", "coverage90_delta", "coverage95_delta"}
    if any(set(record) != required | {"is_reflection"} for record in records):
        raise ValueError("rotation audit records have an unexpected schema")
    errors = torch.tensor([record["equivariance_error"] for record in records], dtype=torch.float64)
    nll = torch.tensor([record["nll_delta"] for record in records], dtype=torch.float64)
    coverage90 = torch.tensor([record["coverage90_delta"] for record in records], dtype=torch.float64)
    coverage95 = torch.tensor([record["coverage95_delta"] for record in records], dtype=torch.float64)
    reflection_count = sum(bool(record["is_reflection"]) for record in records)
    return {
        "transform_count": len(records),
        "rotation_count": len(records) - reflection_count,
        "reflection_count": reflection_count,
        "equivariance_error_mean": float(errors.mean()),
        "equivariance_error_max": float(errors.max()),
        "nll_delta_mean": float(nll.mean()),
        "nll_delta_std": float(nll.std(unbiased=False)),
        "nll_delta_max_abs": float(nll.abs().max()),
        "coverage90_delta_mean": float(coverage90.mean()),
        "coverage90_delta_max_abs": float(coverage90.abs().max()),
        "coverage95_delta_mean": float(coverage95.mean()),
        "coverage95_delta_max_abs": float(coverage95.abs().max()),
        "all_finite": bool(torch.isfinite(torch.cat((errors, nll, coverage90, coverage95))).all()),
        "requested_transform_count": rotations,
    }


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
    cache = torch.load(feature_cache, map_location="cpu", weights_only=True)
    features = cache["features"][:512]
    if "target" not in cache:
        raise ValueError("rotation audit feature cache must contain targets")
    targets = cache["target"][: features.shape[0]]
    batch = torch.arange(features.shape[0])
    feature_irreps = model.backbone.irreps_out
    output_irreps = model.output_spec.irreps
    with torch.no_grad():
        base = model.forward_from_features(
            features, batch, target=targets, return_scale=True
        )
    errors = []
    records = []
    base_scale = base["scale"].to(torch.float64)
    base_mean = base["mu"].to(torch.float64)
    base_target = targets.to(torch.float64)
    base_nll = float(base["loss"].item())
    base_coverage = empirical_coverage(
        base_mean,
        base_target,
        base_scale,
        reference="student_t" if run["args"]["model"].endswith("student_t") else "gaussian",
        student_t_dof=float(run["args"].get("student_t_dof", 5.0)),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260813)
        for _ in range(rotations):
            rotation = o3.rand_matrix()
            is_reflection = len(records) % 2 == 1
            if is_reflection:
                reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
                rotation = reflection @ rotation
            feature_action = feature_irreps.D_from_matrix(rotation)
            output_action = output_irreps.D_from_matrix(rotation).to(torch.float64)
            with torch.no_grad():
                transformed = model.forward_from_features(
                    features @ feature_action.T,
                    batch,
                    target=base_target @ output_action.T,
                    return_scale=True,
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
            equivariance_error = numerator / denominator.clamp_min(1e-15)
            errors.extend(equivariance_error.tolist())
            transformed_scale = transformed["scale"].to(torch.float64)
            transformed_coverage = empirical_coverage(
                transformed["mu"].to(torch.float64),
                base_target @ output_action.T,
                transformed_scale,
                reference="student_t" if run["args"]["model"].endswith("student_t") else "gaussian",
                student_t_dof=float(run["args"].get("student_t_dof", 5.0)),
            )
            records.append(
                {
                    "equivariance_error": float(equivariance_error.mean()),
                    "nll_delta": float(transformed["loss"].item() - base_nll),
                    "coverage90_delta": float(
                        transformed_coverage["coverage_90"] - base_coverage["coverage_90"]
                    ),
                    "coverage95_delta": float(
                        transformed_coverage["coverage_95"] - base_coverage["coverage_95"]
                    ),
                    "is_reflection": is_reflection,
                }
            )
    summary = _rotation_summary(records, rotations=rotations)
    return {"sample_count": features.shape[0], **summary}


def _parse_runs(values: list[str]) -> list[Path]:
    return [Path(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", action="append", required=True)
    parser.add_argument("--shuffled", action="append", required=True)
    parser.add_argument("--no-edge", required=True)
    parser.add_argument("--fixed-coordinate", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--rotations", type=int, default=300)
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
