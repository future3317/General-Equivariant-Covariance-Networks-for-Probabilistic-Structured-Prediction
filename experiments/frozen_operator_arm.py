"""Task-neutral training and evaluation for one frozen-mean operator arm."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evaluation import (
    covariance_spectrum_diagnostics,
    elliptical_falsification,
    energy_score,
    falsification_decision,
)
from scripts.itop_reproducibility import atomic_write_json


@dataclass(frozen=True)
class FrozenOperatorArmSpec:
    """Optimization controls for one validation-selected frozen operator head."""

    run_dir: Path
    seed: int
    max_epochs: int = 60
    patience: int = 5
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer controls")


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _mean_loss(model, loader: Iterable, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for raw in loader:
            batch = _to_device(raw, device)
            result = model(batch["features"], batch["mean"], batch["target"])
            loss = result["loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite frozen-operator validation loss")
            size = int(batch["target"].shape[0])
            total += float(loss) * size
            count += size
    if count == 0:
        raise ValueError("empty frozen-operator loader")
    return total / count


def _train_epoch(model, loader: Iterable, optimizer, device: torch.device) -> dict:
    model.train()
    total = 0.0
    count = 0
    gradient_total = 0.0
    gradient_max = 0.0
    steps = 0
    for raw in loader:
        batch = _to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch["features"], batch["mean"], batch["target"])
        loss = result["loss"]
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("non-finite frozen-operator training loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        size = int(batch["target"].shape[0])
        total += float(loss.detach()) * size
        count += size
        gradient_value = float(gradient.detach())
        gradient_total += gradient_value
        gradient_max = max(gradient_max, gradient_value)
        steps += 1
    if count == 0 or steps == 0:
        raise ValueError("empty frozen-operator training loader")
    return {
        "loss": total / count,
        "gradient_norm_mean": gradient_total / steps,
        "gradient_norm_max": gradient_max,
    }


def train_frozen_operator_arm(
    model: torch.nn.Module,
    *,
    train_loader_for_epoch: Callable[[int], Iterable],
    validation_loader: Iterable,
    device: torch.device,
    spec: FrozenOperatorArmSpec,
    checkpoint_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one operator projection and select a checkpoint by validation NLL."""

    if spec.run_dir.exists() and any(spec.run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite arm: {spec.run_dir}")
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    history: list[dict[str, Any]] = []
    best_validation_nll = float("inf")
    selected_epoch = 0
    stale = 0
    best_path = spec.run_dir / "best_model.pt"
    last_path = spec.run_dir / "last_model.pt"
    for epoch in range(1, spec.max_epochs + 1):
        train = _train_epoch(model, train_loader_for_epoch(epoch), optimizer, device)
        validation_nll = _mean_loss(model, validation_loader, device)
        scheduler.step(validation_nll)
        row = {
            "epoch": epoch,
            "train": train,
            "validation_nll": validation_nll,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        atomic_write_json(history, spec.run_dir / "history.json")
        common = {
            **dict(checkpoint_metadata),
            "epoch": epoch,
            "model_state": model.state_dict(),
        }
        _atomic_torch_save(
            {**common, "optimizer_state": optimizer.state_dict()}, last_path
        )
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            selected_epoch = epoch
            stale = 0
            _atomic_torch_save(common, best_path)
        else:
            stale += 1
        if stale >= spec.patience:
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return {
        "history": history,
        "selection_split": "validation",
        "selected_epoch": selected_epoch,
        "best_validation_nll": best_validation_nll,
    }


@torch.inference_mode()
def evaluate_frozen_operator_arm(
    model: torch.nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    distribution: str,
    student_t_dof: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Evaluate one ordinary elliptical arm without changing its checkpoint."""

    if distribution not in {"gaussian", "student_t"}:
        raise ValueError(f"unsupported distribution: {distribution}")
    model.eval()
    records: dict[str, list[torch.Tensor]] = {}
    total = 0.0
    count = 0
    for raw in loader:
        batch = _to_device(raw, device)
        result = model(batch["features"], batch["mean"], batch["target"])
        if not bool(torch.isfinite(result["loss"])):
            raise FloatingPointError("non-finite frozen-operator evaluation loss")
        size = int(batch["target"].shape[0])
        total += float(result["loss"]) * size
        count += size
        output = {
            "mean": batch["mean"],
            "target": batch["target"],
            "sample_id": batch["sample_id"],
            "params": result["params"],
            "scale": model.spd_map(result["params"]),
        }
        for key, value in output.items():
            records.setdefault(key, []).append(value.detach().cpu())
    if count == 0:
        raise ValueError("empty frozen-operator evaluation loader")
    prediction = {key: torch.cat(values) for key, values in records.items()}
    if not all(bool(torch.isfinite(value).all()) for value in prediction.values()):
        raise FloatingPointError("non-finite frozen-operator prediction artifact")
    minimum_eigenvalue = float(
        torch.linalg.eigvalsh(prediction["scale"].double()).min()
    )
    if minimum_eigenvalue <= 0.0:
        raise FloatingPointError("frozen-operator scale is not strictly SPD")
    reference = "student_t" if distribution == "student_t" else "gaussian"
    dof = student_t_dof if distribution == "student_t" else 5.0
    elliptical = elliptical_falsification(
        prediction["mean"],
        prediction["target"],
        prediction["scale"],
        reference=reference,
        student_t_dof=dof,
        seed=seed + 7002,
    )
    score = energy_score(
        prediction["mean"],
        prediction["scale"],
        prediction["target"],
        num_samples=128,
        distribution=reference,
        student_t_dof=dof,
    )
    metrics = {
        "nll": total / count,
        "nll_semantics": f"exact_single_{distribution}_log_likelihood",
        "energy_score": float(score),
        "minimum_scale_eigenvalue_fp64": minimum_eigenvalue,
        "elliptical_falsification": elliptical,
        "decision": falsification_decision(elliptical),
        "spectrum": covariance_spectrum_diagnostics(prediction["scale"]),
    }
    return metrics, prediction
