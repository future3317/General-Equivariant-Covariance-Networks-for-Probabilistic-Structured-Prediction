"""Task-neutral E1 training from immutable frozen-H,mu cache artifacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch

from compatibility.e3nn import o3
from data.frozen_distribution_features import frozen_distribution_loaders
from distributions import StudentTNLL
from equivcompiler import (
    CenteredSpectralWindowCovariance,
    FeatureSpec,
    FullCovariance,
    SpectralWindowCovariance,
    plan_readout,
)
from evaluation import (
    covariance_spectrum_diagnostics,
    elliptical_falsification,
    energy_score,
    energy_score_from_samples,
    falsification_decision,
    mixture_projection_pit,
    sample_ensemble,
)
from models.frozen_distribution_readout import (
    FrozenConditionalStudentT,
    FrozenMeanScatterStudentT,
    FrozenSymmetricStudentTMixture,
)
from scripts.itop_reproducibility import (
    atomic_write_json,
    sha256_file,
    source_provenance,
)
from spd_maps import RepresentationMetricMap

DISTRIBUTION_VARIANTS = ("fixed", "conditional_nu", "symmetric_mixture")
SPECTRAL_VARIANTS = ("centered_e4", "centered_e8", "matrix_exp")
VARIANTS = DISTRIBUTION_VARIANTS + SPECTRAL_VARIANTS


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _covariance_policy(metadata: dict, variant: str):
    source = metadata["spd_map"]
    if variant == "centered_e4":
        return CenteredSpectralWindowCovariance(-2.0, 2.0, -8.0, 8.0)
    if variant == "centered_e8":
        return CenteredSpectralWindowCovariance(-4.0, 4.0, -8.0, 8.0)
    if variant == "matrix_exp":
        return FullCovariance()
    kind = source["kind"]
    if kind == "matrix_exp":
        return FullCovariance()
    if kind == "spectral_window":
        return SpectralWindowCovariance(
            source["log_variance_min"], source["log_variance_max"]
        )
    if kind == "centered_spectral_window":
        return CenteredSpectralWindowCovariance(
            source["shape_min"],
            source["shape_max"],
            source["volume_min"],
            source["volume_max"],
        )
    raise ValueError(f"unsupported cached SPD map: {kind}")


def _build_spd_map(metadata: dict, variant: str, device: torch.device):
    output_dimension = o3.Irreps(metadata["output_irreps"]).dim
    expected = output_dimension * (output_dimension + 1) // 2
    if int(metadata["parameter_count"]) != expected:
        raise ValueError("the first E1 phase is preregistered for Full coordinates")
    plan = plan_readout(
        FeatureSpec.from_irreps(metadata["feature_irreps"], scope="global"),
        output=metadata["output_irreps"],
        covariance=_covariance_policy(metadata, variant),
        distribution="student_t",
        student_t_dof=float(metadata["student_t_dof"]),
        output_scope="global",
    )
    compilation = plan.compilation
    if compilation.covariance_parameter_count != int(metadata["parameter_count"]):
        raise RuntimeError("reconstructed operator parameter count changed")
    spd_map = compilation.build_spd_map().to(device)
    metric = metadata["spd_map"].get("representation_metric", "none")
    if metric == "block_auto":
        values = [metadata["spd_map"]["metric_scalar"]] + [
            metadata["spd_map"]["metric_l2"]
        ] * (compilation.output_spec.dim - 1)
        spd_map = RepresentationMetricMap(
            spd_map, torch.tensor(values, dtype=torch.float32, device=device)
        ).to(device)
    elif metric != "none":
        raise ValueError(f"unsupported cached representation metric: {metric}")
    return spd_map, compilation


def _build_model(metadata: dict, variant: str, spd_map, device: torch.device):
    if variant == "fixed":
        return None
    if variant == "conditional_nu":
        return FrozenConditionalStudentT(metadata["feature_irreps"], spd_map).to(device)
    if variant == "symmetric_mixture":
        return FrozenSymmetricStudentTMixture(
            metadata["feature_irreps"],
            metadata["output_irreps"],
            spd_map,
            student_t_dof=float(metadata["student_t_dof"]),
        ).to(device)
    model = FrozenMeanScatterStudentT(
        metadata["feature_irreps"],
        metadata["parameter_irreps"],
        spd_map,
        student_t_dof=float(metadata["student_t_dof"]),
    ).to(device)
    projection_record = metadata["operator_projection"]
    projection_path = Path(metadata["cache_dir"]) / projection_record["path"]
    if sha256_file(projection_path) != projection_record["sha256"]:
        raise ValueError("cached operator projection hash mismatch")
    state = torch.load(projection_path, map_location=device, weights_only=True)
    model.parameter_projection.load_state_dict(state, strict=True)
    return model


def _forward(model, variant: str, batch: dict, spd_map, objective: StudentTNLL):
    if variant == "fixed":
        loss, components = objective(
            batch["mean"], batch["params"], batch["target"], spd_map
        )
        return {"loss": loss, "params": batch["params"], **components}
    if variant in {"conditional_nu", "symmetric_mixture"}:
        return model(
            batch["features"],
            batch["mean"],
            batch["params"],
            batch["target"],
        )
    return model(batch["features"], batch["mean"], batch["target"])


@torch.inference_mode()
def _loss(model, variant: str, loader, device, spd_map, objective) -> float:
    if model is not None:
        model.eval()
    total = 0.0
    count = 0
    for raw in loader:
        batch = _to_device(raw, device)
        result = _forward(model, variant, batch, spd_map, objective)
        if not bool(torch.isfinite(result["loss"])):
            raise FloatingPointError("non-finite E1 validation loss")
        size = int(batch["target"].shape[0])
        total += float(result["loss"]) * size
        count += size
    return total / count


def _train_epoch(model, variant: str, loader, optimizer, device, spd_map, objective):
    model.train()
    total = 0.0
    count = 0
    norm_total = 0.0
    norm_max = 0.0
    steps = 0
    for raw in loader:
        batch = _to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        result = _forward(model, variant, batch, spd_map, objective)
        loss = result["loss"]
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("non-finite E1 training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        size = int(batch["target"].shape[0])
        total += float(loss.detach()) * size
        count += size
        value = float(norm.detach())
        norm_total += value
        norm_max = max(norm_max, value)
        steps += 1
    return {
        "loss": total / count,
        "gradient_norm_mean": norm_total / steps,
        "gradient_norm_max": norm_max,
    }


@torch.inference_mode()
def _predict(model, variant: str, loader, device, spd_map, objective):
    if model is not None:
        model.eval()
    records: dict[str, list[torch.Tensor]] = {}
    total = 0.0
    count = 0
    for raw in loader:
        batch = _to_device(raw, device)
        result = _forward(model, variant, batch, spd_map, objective)
        size = int(batch["target"].shape[0])
        total += float(result["loss"]) * size
        count += size
        common = {
            "mean": batch["mean"],
            "target": batch["target"],
            "sample_id": batch["sample_id"],
        }
        if variant == "symmetric_mixture":
            output = {
                **common,
                "component_means": result["component_means"],
                "component_scales": result["component_scales"],
                "weights": result["weights"],
                "delta": result["delta"],
            }
        else:
            params = result["params"]
            output = {**common, "params": params, "scale": spd_map(params)}
            if variant == "conditional_nu":
                output["nu"] = result["nu"]
        for key, value in output.items():
            records.setdefault(key, []).append(value.detach().cpu())
    concatenated = {}
    for key, values in records.items():
        dimension = (
            1 if key in {"component_means", "component_scales", "weights"} else 0
        )
        concatenated[key] = torch.cat(values, dim=dimension)
    return total / count, concatenated


def _diagnostics(
    prediction: dict,
    variant: str,
    nll: float,
    dof: float,
    seed: int,
    spd_map,
):
    torch.manual_seed(seed + 7001)
    if variant == "symmetric_mixture":
        projection = mixture_projection_pit(
            prediction["component_means"],
            prediction["component_scales"],
            prediction["target"],
            student_t_dof=dof,
            weights=prediction["weights"],
            seed=seed + 7002,
        )
        samples = sample_ensemble(
            prediction["component_means"],
            prediction["component_scales"],
            num_samples=128,
            distribution="student_t",
            student_t_dof=dof,
        )
        return {
            "nll": nll,
            "nll_semantics": "exact_finite_mixture_logsumexp",
            "energy_score": float(
                energy_score_from_samples(samples, prediction["target"]).item()
            ),
            "mixture_projection_pit": projection,
            "single_ellipse_whitening_not_applicable": True,
        }
    nu: float | torch.Tensor = prediction.get("nu", dof)
    elliptical = elliptical_falsification(
        prediction["mean"],
        prediction["target"],
        prediction["scale"],
        reference="student_t",
        student_t_dof=nu,
        seed=seed + 7002,
    )
    if isinstance(nu, torch.Tensor):
        samples = sample_ensemble(
            prediction["mean"].unsqueeze(0),
            prediction["scale"].unsqueeze(0),
            num_samples=128,
            distribution="student_t",
            student_t_dof=nu,
        )
        score = energy_score_from_samples(samples, prediction["target"])
    else:
        score = energy_score(
            prediction["mean"],
            prediction["scale"],
            prediction["target"],
            num_samples=128,
            distribution="student_t",
            student_t_dof=nu,
        )
    result = {
        "nll": nll,
        "nll_semantics": "exact_single_student_t_log_likelihood",
        "energy_score": float(score.item()),
        "elliptical_falsification": elliptical,
        "decision": falsification_decision(elliptical),
        "spectrum": covariance_spectrum_diagnostics(prediction["scale"]),
    }
    generator_map = (
        spd_map.base if isinstance(spd_map, RepresentationMetricMap) else spd_map
    )
    if hasattr(generator_map, "_transform_parameters") and hasattr(
        generator_map, "delegate"
    ):
        generator = generator_map._transform_parameters(prediction["params"])
        delegate = generator_map.delegate
        if hasattr(delegate, "shape_min") and hasattr(delegate, "shape_max"):
            dimension = generator.shape[-1]
            raw_volume = torch.diagonal(generator, dim1=-2, dim2=-1).sum(-1) / dimension
            centered = generator - raw_volume[..., None, None] * torch.eye(
                dimension, dtype=generator.dtype
            )
            shape_position = torch.sigmoid(torch.linalg.eigvalsh(centered))
            volume_position = torch.sigmoid(raw_volume)
            result["spectral_logit_saturation"] = {
                "shape_lower_1pct_fraction": float(
                    (shape_position <= 0.01).float().mean()
                ),
                "shape_upper_1pct_fraction": float(
                    (shape_position >= 0.99).float().mean()
                ),
                "shape_position_min": float(shape_position.min()),
                "shape_position_max": float(shape_position.max()),
                "volume_lower_1pct_fraction": float(
                    (volume_position <= 0.01).float().mean()
                ),
                "volume_upper_1pct_fraction": float(
                    (volume_position >= 0.99).float().mean()
                ),
                "declared_shape_bounds": [delegate.shape_min, delegate.shape_max],
                "declared_volume_bounds": [delegate.volume_min, delegate.volume_max],
            }
    return result


def _save_checkpoint(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite E1 run: {args.run_dir}")
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    _set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    metadata, loaders = frozen_distribution_loaders(
        args.cache_dir,
        seed=args.seed,
        epoch=0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metadata = dict(metadata)
    metadata["cache_dir"] = str(args.cache_dir.resolve())
    spd_map, compilation = _build_spd_map(metadata, args.variant, device)
    model = _build_model(metadata, args.variant, spd_map, device)
    objective = StudentTNLL(nu=float(metadata["student_t_dof"]))
    args.run_dir.mkdir(parents=True)
    history = []
    best = float("inf")
    stale = 0
    selected_epoch = 0
    best_path = args.run_dir / "best_model.pt"
    if model is None:
        best = _loss(None, args.variant, loaders["val"], device, spd_map, objective)
        history = [
            {
                "epoch": 0,
                "train": None,
                "validation_nll": best,
                "learning_rate": None,
                "baseline_reproduction": True,
            }
        ]
        atomic_write_json(history, args.run_dir / "history.json")
        _save_checkpoint({"variant": args.variant, "model_state": {}}, best_path)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        for epoch in range(1, args.max_epochs + 1):
            _, epoch_loaders = frozen_distribution_loaders(
                args.cache_dir,
                seed=args.seed,
                epoch=epoch,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            train = _train_epoch(
                model,
                args.variant,
                epoch_loaders["train"],
                optimizer,
                device,
                spd_map,
                objective,
            )
            validation = _loss(
                model,
                args.variant,
                loaders["val"],
                device,
                spd_map,
                objective,
            )
            scheduler.step(validation)
            record = {
                "epoch": epoch,
                "train": train,
                "validation_nll": validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            atomic_write_json(history, args.run_dir / "history.json")
            _save_checkpoint(
                {
                    "variant": args.variant,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                args.run_dir / "last_state.pt",
            )
            if validation < best:
                best = validation
                stale = 0
                selected_epoch = epoch
                _save_checkpoint(
                    {
                        "variant": args.variant,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                    },
                    best_path,
                )
            else:
                stale += 1
            if stale >= args.patience:
                break
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    predictions = {}
    diagnostics = {}
    artifact_hashes = {}
    for split, loader in loaders.items():
        if split == "train":
            _, evaluation_loaders = frozen_distribution_loaders(
                args.cache_dir,
                seed=args.seed,
                epoch=0,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            loader = evaluation_loaders[split]
        nll, prediction = _predict(
            model, args.variant, loader, device, spd_map, objective
        )
        predictions[split] = prediction
        diagnostics[split] = _diagnostics(
            prediction,
            args.variant,
            nll,
            float(metadata["student_t_dof"]),
            args.seed,
            spd_map,
        )
        path = args.run_dir / f"predictions_{split}.pt"
        _save_checkpoint(prediction, path)
        artifact_hashes[path.name] = sha256_file(path)
    protocol = {
        "schema_version": 1,
        "study": "E1 frozen-H,mu distribution family",
        "variant": args.variant,
        "hypothesis": (
            "the centered spectral restriction is a principal cause of dielectric failure"
            if args.variant in SPECTRAL_VARIANTS
            else "conditional radial flexibility or K=2 topology improves held-out proper scores "
            "without changing frozen H, mean, or the first-stage shared scatter"
        ),
        "intervention": (
            model.schema()
            if model is not None
            else {
                "kind": "released_fixed_nu_single_student_t_baseline",
                "degrees_of_freedom": float(metadata["student_t_dof"]),
            }
        ),
        "frozen": {
            "source_checkpoint": metadata["source_checkpoint"],
            "cache_metadata_sha256": sha256_file(args.cache_dir / "metadata.json"),
            "mean": True,
            "features": True,
            "shared_scatter": args.variant in DISTRIBUTION_VARIANTS,
        },
        "splits": metadata["splits"],
        "selection": {
            "metric": "validation_nll",
            "selected_epoch": selected_epoch,
            "best_validation_nll": best,
            "ood_used_for_selection": False,
        },
        "seed": args.seed,
        "optimizer": {
            "kind": "AdamW" if model is not None else "none_baseline_evaluation",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": "ReduceLROnPlateau(factor=0.5,patience=2)",
            "early_stopping_patience": args.patience,
        },
        "precision": {
            "features": "FP32 cached",
            "operator_and_nll": "FP32",
            "tf32": False,
        },
        "distribution_schema": (
            model.schema()
            if model is not None
            else {
                "kind": "single_elliptical_fixed_nu_student_t",
                "degrees_of_freedom": float(metadata["student_t_dof"]),
            }
        ),
        "operator_schema": compilation.operator_family.as_dict(),
        "spd_map_intervention": {
            "variant": args.variant,
            "only_changed_factor_in_spectral_control": args.variant
            in SPECTRAL_VARIANTS,
        },
        "compilation": compilation.as_dict(),
        "nll_semantics": (
            "exact_finite_mixture_logsumexp"
            if args.variant == "symmetric_mixture"
            else "exact_single_student_t_log_likelihood"
        ),
    }
    artifact_hashes[best_path.name] = sha256_file(best_path)
    artifact_hashes["history.json"] = sha256_file(args.run_dir / "history.json")
    last_state = args.run_dir / "last_state.pt"
    if last_state.is_file():
        artifact_hashes[last_state.name] = sha256_file(last_state)
    atomic_write_json(protocol, args.run_dir / "protocol.json")
    atomic_write_json(diagnostics, args.run_dir / "diagnostics.json")
    environment = {
        "source": source_provenance(Path(__file__).resolve().parents[1]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "pid": os.getpid(),
    }
    atomic_write_json(environment, args.run_dir / "environment.json")
    artifact_hashes.update(
        {
            "protocol.json": sha256_file(args.run_dir / "protocol.json"),
            "diagnostics.json": sha256_file(args.run_dir / "diagnostics.json"),
            "environment.json": sha256_file(args.run_dir / "environment.json"),
        }
    )
    atomic_write_json(
        {
            "selected_epoch": selected_epoch,
            "best_validation_nll": best,
            "artifact_sha256": artifact_hashes,
        },
        args.run_dir / "manifest.json",
    )
    print(
        json.dumps({"run_dir": str(args.run_dir), "diagnostics": diagnostics}, indent=2)
    )


if __name__ == "__main__":
    main()
