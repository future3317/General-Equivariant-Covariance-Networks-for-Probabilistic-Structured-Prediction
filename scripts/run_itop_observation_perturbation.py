"""E2 diagnostic: push frozen ITOP means through input perturbations."""

from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import Dataset
from tqdm import tqdm

from compatibility.torch_geometric import PyGDataLoader
from data.itop_dataset import ITOPDepthDataset, itop_paths
from data.observation_perturbations import (
    DepthPerturbationScale,
    perturb_depth_observation,
)
from evaluation import risk_coverage_auc
from scripts.itop_reproducibility import (
    atomic_write_json,
    sha256_file,
    source_provenance,
)
from scripts.train_itop import _build_model, _forward

PERTURBATIONS = ("missing_block", "point_dropout", "depth_noise")


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().cpu()
    return {
        "mean": float(values.mean()),
        "q10": float(torch.quantile(values, 0.10)),
        "median": float(torch.quantile(values, 0.50)),
        "q90": float(torch.quantile(values, 0.90)),
    }


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    value = float(
        spearmanr(
            left.detach().double().cpu().numpy(),
            right.detach().double().cpu().numpy(),
        ).statistic
    )
    return value if math.isfinite(value) else 0.0


def _training_scale(
    depth_path: Path,
    geometry_path: Path,
    *,
    sample_count: int = 256,
) -> tuple[DepthPerturbationScale, dict[str, Any]]:
    frame_indices = np.load(geometry_path / "frame_indices.npy", mmap_mode="r")
    positions = np.linspace(0, len(frame_indices) - 1, sample_count).astype(np.int64)
    selected_frames = np.asarray(frame_indices[positions], dtype=np.int64)
    missing_fractions = []
    quantization_values = []
    with h5py.File(depth_path, "r") as source:
        for frame_index in selected_frames:
            depth = np.asarray(source["data"][int(frame_index)], dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0.0)
            missing_fractions.append(1.0 - float(valid.mean()))
            quantization_values.append(depth[::8, ::8][valid[::8, ::8]])
    sampled_depth = np.unique(np.concatenate(quantization_values))
    increments = np.diff(sampled_depth)
    increments = increments[increments > np.finfo(np.float32).eps]
    if increments.size == 0:
        raise ValueError("Side-train depths do not expose a positive quantization step")
    quantization_step = float(np.quantile(increments, 0.10))
    missing_fraction = float(np.median(missing_fractions))
    scale = DepthPerturbationScale(
        missing_fraction=missing_fraction,
        depth_noise_std=quantization_step / math.sqrt(12.0),
    )
    return scale, {
        "source": "Side-train observations only",
        "sample_count": sample_count,
        "frame_index_min": int(selected_frames.min()),
        "frame_index_max": int(selected_frames.max()),
        "median_invalid_depth_fraction": missing_fraction,
        "sampled_depth_quantization_step_m": quantization_step,
        "uniform_quantization_noise_std_m": scale.depth_noise_std,
        "policy": {
            "missing_block_area_fraction": missing_fraction,
            "independent_point_dropout_probability": missing_fraction,
            "depth_noise": "Gaussian with Side-train quantization-bin standard deviation",
        },
    }


class _PerturbedDepthDataset(Dataset):
    def __init__(
        self,
        base: ITOPDepthDataset,
        *,
        kind: str,
        scale: DepthPerturbationScale,
        repeat: int,
        seed: int,
    ) -> None:
        self.base = base
        self.kind = kind
        self.scale = scale
        self.repeat = repeat
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int):
        frame_index = int(self.base.indices[item])
        kind_index = PERTURBATIONS.index(self.kind)
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.repeat, frame_index, kind_index]
            )
        )
        depth = perturb_depth_observation(
            self.base.depth_map(item),
            kind=self.kind,
            scale=self.scale,
            rng=rng,
        )
        record = self.base.sample_record_from_depth(item, depth)
        return self.base.data_from_record(record)


@torch.inference_mode()
def _predict_repeats(
    model,
    base: ITOPDepthDataset,
    *,
    kind: str,
    scale: DepthPerturbationScale,
    repeats: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    draws = []
    reference_ids = None
    model.eval()
    for repeat in range(repeats):
        dataset = _PerturbedDepthDataset(
            base,
            kind=kind,
            scale=scale,
            repeat=repeat,
            seed=seed,
        )
        loader = PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # This diagnostic repeatedly constructs short-lived loaders. Pin
            # and persistent worker threads add no reuse and can outlive a
            # repeat when HDF5 workers are torn down.
            pin_memory=False,
            persistent_workers=False,
        )
        means = []
        sample_ids = []
        for batch in tqdm(loader, desc=f"{kind} repeat {repeat + 1}/{repeats}"):
            batch = batch.to(device, non_blocking=True)
            result = _forward(
                model,
                batch,
                target=None,
                return_scale=False,
                use_bf16=use_bf16,
            )
            mean = result["mu"].float()
            if not bool(torch.isfinite(mean).all()):
                raise FloatingPointError(f"non-finite perturbed mean for {kind}")
            means.append(mean.cpu())
            sample_ids.append(
                (batch.view_id.long() * (1 << 32) + batch.frame_index.long()).cpu()
            )
        current_ids = torch.cat(sample_ids)
        if reference_ids is None:
            reference_ids = current_ids
        elif not torch.equal(reference_ids, current_ids):
            raise ValueError("perturbation repeats changed sample order")
        draws.append(torch.cat(means))
    if reference_ids is None:
        raise ValueError("perturbation loader was empty")
    return torch.stack(draws), reference_ids


def _pushforward(draws: torch.Tensor) -> dict[str, torch.Tensor]:
    center = draws.mean(0)
    residual = draws - center.unsqueeze(0)
    covariance = torch.einsum("rni,rnj->nij", residual, residual) / (
        draws.shape[0] - 1
    )
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    return {
        "mean_draws": draws,
        "mean": center,
        "covariance": covariance,
        "trace": diagonal.sum(-1),
        "per_joint_variance": diagonal.reshape(-1, 15, 3).sum(-1),
    }


def _aligned_prediction(path: Path, sample_id: torch.Tensor) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.equal(payload["sample_id"].long(), sample_id.long()):
        raise ValueError(f"prediction IDs do not align with perturbations: {path}")
    return payload


def _diagnostics(
    pushforward: dict[str, torch.Tensor],
    fixed: dict[str, torch.Tensor],
    mixtures: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    target = fixed["target"].reshape(-1, 15, 3)
    mean = fixed["mean"].reshape(-1, 15, 3)
    joint_error2 = (target - mean).square().sum(-1)
    frame_error = joint_error2.mean(-1)
    predictive_covariance = (5.0 / 3.0) * fixed["scale"]
    predictive_diagonal = torch.diagonal(
        predictive_covariance, dim1=-2, dim2=-1
    )
    predictive_trace = predictive_diagonal.sum(-1)
    predictive_joint = predictive_diagonal.reshape(-1, 15, 3).sum(-1)
    trace = pushforward["trace"]
    joint_variance = pushforward["per_joint_variance"]
    result: dict[str, Any] = {
        "pushforward_trace_m2": _quantiles(trace),
        "frozen_mean_shift_squared_m2": _quantiles(
            (pushforward["mean"] - fixed["mean"]).square().sum(-1)
        ),
        "frame_relations": {
            "error_severity_spearman": _spearman(trace, frame_error),
            "learned_predictive_covariance_trace_spearman": _spearman(
                trace, predictive_trace
            ),
            "risk_coverage_auc_m2": float(risk_coverage_auc(trace, frame_error)),
            "oracle_risk_coverage_auc_m2": float(
                risk_coverage_auc(frame_error, frame_error)
            ),
        },
        "joint_relations": {
            "squared_error_spearman": _spearman(
                joint_variance.flatten(), joint_error2.flatten()
            ),
            "learned_predictive_variance_spearman": _spearman(
                joint_variance.flatten(), predictive_joint.flatten()
            ),
            "risk_coverage_auc_m2": float(
                risk_coverage_auc(joint_variance.flatten(), joint_error2.flatten())
            ),
        },
        "mixture_relations": {},
    }
    for mixture in mixtures:
        delta = mixture["delta"].float()
        component_distance2 = 4.0 * delta.square().sum(-1)
        cholesky = torch.linalg.cholesky(fixed["scale"].float())
        solved = torch.cholesky_solve(delta.unsqueeze(-1), cholesky).squeeze(-1)
        component_mahalanobis2 = 4.0 * (delta * solved).sum(-1)
        result["mixture_relations"][mixture["seed_name"]] = {
            "component_distance_squared_spearman": _spearman(
                trace, component_distance2
            ),
            "component_mahalanobis_squared_spearman": _spearman(
                trace, component_mahalanobis2
            ),
            "per_joint_component_distance_squared_spearman": _spearman(
                joint_variance.flatten(),
                (4.0 * delta.reshape(-1, 15, 3).square().sum(-1)).flatten(),
            ),
        }
    return result


def _save_tensor(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--side_train_geometry", type=Path, required=True)
    parser.add_argument("--side_train_depth", type=Path, required=True)
    parser.add_argument("--fixed_run", type=Path, required=True)
    parser.add_argument("--mixture_run", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E2 diagnostic: {args.output_dir}")
    if args.repeats < 2:
        raise ValueError("pushforward covariance requires at least two repeats")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_kind") != "deterministic":
        raise ValueError("E2 perturbations require a deterministic checkpoint")
    training_args = Namespace(**checkpoint["args"])
    model, plan = _build_model(training_args)
    if plan is not None:
        raise RuntimeError("deterministic checkpoint unexpectedly compiled a scatter")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device)

    scale, scale_record = _training_scale(
        args.side_train_depth,
        args.side_train_geometry,
    )
    bases = {}
    for view in ("side", "top"):
        depth_path, labels_path = itop_paths(Path(training_args.data_dir), view, "test")
        bases[view] = ITOPDepthDataset(
            depth_path,
            labels_path,
            view=view,
            num_points=int(training_args.num_points),
            num_neighbors=int(training_args.num_neighbors),
        )

    output: dict[str, Any] = {}
    diagnostics = {}
    for split, view in (("test", "side"), ("ood", "top")):
        output[split] = {}
        fixed = None
        mixtures = None
        for kind in PERTURBATIONS:
            draws, sample_id = _predict_repeats(
                model,
                bases[view],
                kind=kind,
                scale=scale,
                repeats=args.repeats,
                seed=args.seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                use_bf16=training_args.backbone_precision == "bf16",
            )
            if fixed is None:
                fixed = _aligned_prediction(
                    args.fixed_run / f"predictions_{split}.pt", sample_id
                )
                mixtures = []
                for run in args.mixture_run:
                    mixture = _aligned_prediction(
                        run / f"predictions_{split}.pt", sample_id
                    )
                    protocol = json.loads((run / "protocol.json").read_text())
                    mixture["seed_name"] = f"seed_{protocol['seed']}"
                    mixtures.append(mixture)
            pushforward = _pushforward(draws)
            output[split][kind] = {"sample_id": sample_id, **pushforward}
            diagnostics.setdefault(split, {})[kind] = _diagnostics(
                pushforward,
                fixed,
                mixtures,
            )

    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "pushforward_predictions.pt"
    _save_tensor(output, prediction_path)
    protocol = {
        "schema_version": 1,
        "study": "E2 ITOP observation perturbation pushforward covariance",
        "role": "input-sensitivity diagnostic only; not predictive, calibrated, or epistemic covariance",
        "hypothesis": (
            "input-only depth perturbations expose observation ambiguity that the "
            "current frozen uncertainty path may not track"
        ),
        "selection": "no fitted parameters; perturbation scale from Side train only",
        "evaluation": {"test": "Side IID", "ood": "Top cross-view only"},
        "perturbations": list(PERTURBATIONS),
        "repeats": args.repeats,
        "seed": args.seed,
        "scale": scale_record,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
        "fixed_run": str(args.fixed_run.resolve()),
        "mixture_runs": [str(path.resolve()) for path in args.mixture_run],
        "prediction_artifact": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
        },
        "diagnostics": diagnostics,
        "source": source_provenance(Path(__file__).resolve().parents[1]),
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "backbone_precision": training_args.backbone_precision,
        },
    }
    atomic_write_json(protocol, args.output_dir / "diagnostics.json")
    print(json.dumps({"scale": scale_record, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
