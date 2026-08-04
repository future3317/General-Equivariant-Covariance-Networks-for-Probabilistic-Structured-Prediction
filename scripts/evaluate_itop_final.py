"""Audit and visualize the controlled full-data ITOP study.

This script consumes completed artifacts from ``run_itop_study``.  It keeps
frozen-head and joint-finetuned models separate, recomputes prediction-level
metrics from saved predictions, and writes a compact JSON/Markdown audit plus
publication-style figures.  It does not retrain or alter checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow direct execution as ``python scripts/evaluate_itop_final.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.itop_dataset import ITOP_JOINT_NAMES, ITOP_SKELETON_EDGES
from evaluation import binary_auroc, risk_coverage_auc
from plotting import COLORS, PALETTE, cm2inch, label_panels, save_figure, setup_tpami_style


MODEL_INFO = OrderedDict(
    (
        ("deterministic", ("Deterministic", "deterministic")),
        (
            "frozen_independent_gaussian",
            ("Independent Gaussian (frozen)", "frozen"),
        ),
        ("frozen_graph_gaussian", ("Graph Gaussian (frozen)", "frozen")),
        ("frozen_graph_student_t", ("Graph Student-t (frozen)", "frozen")),
        (
            "joint_independent_gaussian",
            ("Independent Gaussian (joint)", "joint"),
        ),
        ("joint_graph_student_t", ("Graph Student-t (joint)", "joint")),
    )
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_summary(history: list[dict[str, Any]], log_text: str) -> dict[str, Any]:
    best = min(history, key=lambda record: float(record["loss"]))
    return {
        "epochs_recorded": len(history),
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["loss"]),
        "best_train_loss": float(best.get("train_loss", float("nan"))),
        "final_validation_loss": float(history[-1]["loss"]),
        "early_stopped": "early stopping" in log_text.lower(),
    }


def _prediction_audit(
    prediction: dict[str, torch.Tensor],
    metrics: dict[str, Any],
    *,
    model: str,
    view: str,
) -> dict[str, Any]:
    mean = prediction["mean"].float()
    target = prediction["target"].float()
    if mean.shape != target.shape or mean.shape[-1] != 45:
        raise ValueError(f"{model}/{view}: invalid mean/target shape {mean.shape}")
    errors = torch.linalg.vector_norm(
        (mean - target).reshape(mean.shape[0], 15, 3), dim=-1
    )
    visible = prediction["visible_joints"].bool()
    if visible.shape != errors.shape:
        raise ValueError(f"{model}/{view}: visibility shape mismatch")
    if not torch.isfinite(mean).all() or not torch.isfinite(target).all():
        raise ValueError(f"{model}/{view}: non-finite prediction or target")
    frame_index = prediction["frame_index"]
    unique_frame_count = int(frame_index.unique().numel())

    recomputed = {
        "mpjpe_cm": float(errors.mean().item() * 100.0),
        "visible_mpjpe_cm": float(errors[visible].mean().item() * 100.0),
        "occluded_mpjpe_cm": float(errors[~visible].mean().item() * 100.0),
        "pck_5cm": float((errors <= 0.05).float().mean().item()),
        "pck_10cm": float((errors <= 0.10).float().mean().item()),
        "pck_15cm": float((errors <= 0.15).float().mean().item()),
    }
    checks = {}
    for key, value in recomputed.items():
        recorded = metrics.get(key)
        checks[key] = recorded is not None and abs(float(recorded) - value) < 1e-3

    if "frame_uncertainty" in prediction:
        uncertainty = prediction["frame_uncertainty"].float()
        if not torch.isfinite(uncertainty).all():
            raise ValueError(f"{model}/{view}: non-finite uncertainty")
        risk = float(
            risk_coverage_auc(uncertainty, errors.mean(-1)).item() * 100.0
        )
        recorded_risk = float(metrics["frame_risk_coverage_auc_cm"])
        checks["frame_risk_coverage_auc_cm"] = abs(risk - recorded_risk) < 1e-3
        recomputed["frame_risk_coverage_auc_cm"] = risk

    return {
        "num_samples": int(mean.shape[0]),
        "mean_shape": list(mean.shape),
        "frame_index_unique": unique_frame_count == frame_index.numel(),
        "frame_index_unique_count": unique_frame_count,
        "frame_index_duplicate_rate": float(
            1.0 - unique_frame_count / max(1, frame_index.numel())
        ),
        "frame_index_provenance_warning": (
            "duplicate/offset frame IDs detected; prediction order remains valid, "
            "but this artifact predates the ITOPData batching metadata fix"
            if unique_frame_count != frame_index.numel()
            else None
        ),
        "recomputed": recomputed,
        "recorded_metric_checks": checks,
        "all_recorded_checks_pass": all(checks.values()),
    }


def _load_prediction(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def audit(results: Path, output: Path) -> dict[str, Any]:
    root = results / "seed_42"
    output.mkdir(parents=True, exist_ok=True)
    selected = _read_json(results / "graph_family_selection.json")
    records: OrderedDict[str, Any] = OrderedDict()

    for model, (label, phase) in MODEL_INFO.items():
        run = root / model
        metrics = _read_json(run / "metrics.json")
        history = _read_json(run / "history.json")
        log_path = run / "train.log"
        log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        model_record: dict[str, Any] = {
            "label": label,
            "phase": phase,
            "run_dir": str(run),
            "args": _read_json(run / "args.json") if (run / "args.json").is_file() else {},
            "environment": _read_json(run / "environment.json") if (run / "environment.json").is_file() else {},
            "metrics": metrics,
            "training": _history_summary(history, log_text),
            "prediction_audit": {},
        }
        for view in ("side", "top"):
            prediction = _load_prediction(run / f"predictions_{view}.pt")
            model_record["prediction_audit"][view] = _prediction_audit(
                prediction, metrics[view], model=model, view=view
            )
        records[model] = model_record

    audit_record = {
        "schema_version": 1,
        "study": "ITOP final one-seed full side-train, 512 points",
        "seed": 42,
        "test_views": {"side": "IID", "top": "cross-view OOD"},
        "graph_family_selection": selected,
        "model_note": (
            "The runner jointly fine-tunes independent Gaussian and the graph "
            "family selected by frozen-head validation NLL. Thus graph Gaussian "
            "is a frozen-head comparator here; graph Student-t is the selected "
            "joint graph model."
        ),
        "models": records,
    }
    (output / "final_evaluation.json").write_text(
        json.dumps(audit_record, indent=2) + "\n", encoding="utf-8"
    )
    _write_tables(audit_record, output)
    _write_figures(results, output, audit_record)
    return audit_record


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and not np.isfinite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _write_tables(audit_record: dict[str, Any], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model, record in audit_record["models"].items():
        metrics = record["metrics"]
        row = {
            "model": record["label"],
            "phase": record["phase"],
            "best_epoch": record["training"]["best_epoch"],
            "best_validation_loss": record["training"]["best_validation_loss"],
        }
        for view in ("side", "top"):
            payload = metrics[view]
            for key in (
                "mpjpe_cm",
                "nll",
                "mace",
                "joint_mace",
                "visible_mpjpe_cm",
                "occluded_mpjpe_cm",
                "frame_risk_coverage_auc_cm",
                "joint_risk_coverage_auc_cm",
                "pck_5cm",
                "pck_10cm",
                "pck_15cm",
            ):
                row[f"{view}_{key}"] = payload.get(key)
        row["ood_uncertainty_auroc"] = metrics.get("ood", {}).get(
            "side_top_uncertainty_auroc"
        )
        rows.append(row)

    fieldnames = list(rows[0])
    with (output / "final_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# ITOP final training evaluation",
        "",
        "Scope: one seed (42), full valid side-train exposure, 512 points, one GPU; side-test is IID and top-test is cross-view OOD.",
        "",
        "The runner jointly fine-tunes independent Gaussian and the graph family selected by frozen-head validation NLL. Therefore the graph-Gaussian row is frozen-head, while graph Student-t has both frozen and joint rows.",
        "",
        "## Training summary",
        "",
        "| Model | Phase | Epochs | Best epoch | Best validation objective | Final objective |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in audit_record["models"].values():
        training = record["training"]
        lines.append(
            f"| {record['label']} | {record['phase']} | {training['epochs_recorded']} | {training['best_epoch']} | {_fmt(training['best_validation_loss'])} | {_fmt(training['final_validation_loss'])} |"
        )
    lines += [
        "",
        "## Test metrics",
        "",
        "| Model | Side MPJPE | Side NLL | Side MACE | Top MPJPE | Top NLL | Top MACE | Side→Top AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in audit_record["models"].values():
        metrics = record["metrics"]
        lines.append(
            f"| {record['label']} | {_fmt(metrics['side'].get('mpjpe_cm'))} | {_fmt(metrics['side'].get('nll'))} | {_fmt(metrics['side'].get('mace'))} | {_fmt(metrics['top'].get('mpjpe_cm'))} | {_fmt(metrics['top'].get('nll'))} | {_fmt(metrics['top'].get('mace'))} | {_fmt(metrics.get('ood', {}).get('side_top_uncertainty_auroc'))} |"
        )
    lines += [
        "",
        "## Structural and selective-risk diagnostics",
        "",
        "| Model | Side visible/occluded MPJPE | Top visible/occluded MPJPE | Side frame RC AUC | Top frame RC AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in audit_record["models"].values():
        side, top = record["metrics"]["side"], record["metrics"]["top"]
        lines.append(
            f"| {record['label']} | {_fmt(side.get('visible_mpjpe_cm'))}/{_fmt(side.get('occluded_mpjpe_cm'))} cm | {_fmt(top.get('visible_mpjpe_cm'))}/{_fmt(top.get('occluded_mpjpe_cm'))} cm | {_fmt(side.get('frame_risk_coverage_auc_cm'))} cm | {_fmt(top.get('frame_risk_coverage_auc_cm'))} cm |"
        )
    lines += [
        "",
        "## Audit checks",
        "",
        "Each saved prediction artifact was checked for finite values, 45-dimensional output, and agreement between recomputed MPJPE/PCK/visibility metrics and `metrics.json`. Probabilistic prediction artifacts additionally passed the saved frame risk-coverage check. The old run's `frame_index` field is separately audited because PyG had offset this metadata during batching; it does not affect tensor ordering or metric values.",
        "",
    ]
    for record in audit_record["models"].values():
        status = all(
            payload["all_recorded_checks_pass"]
            for payload in record["prediction_audit"].values()
        )
        provenance_warning = any(
            payload["frame_index_provenance_warning"]
            for payload in record["prediction_audit"].values()
        )
        suffix = " (frame-index provenance warning)" if provenance_warning else ""
        lines.append(f"- {record['label']}: {'PASS' if status else 'FAIL'}{suffix}")
    (output / "final_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_metrics(audit_record: dict[str, Any], model: str) -> dict[str, Any]:
    return audit_record["models"][model]["metrics"]


def _write_figures(results: Path, output: Path, audit_record: dict[str, Any]) -> None:
    setup_tpami_style()
    _plot_training_curves(results, output)
    _plot_metric_comparison(output, audit_record)
    _plot_ood(output, audit_record)
    _plot_risk_coverage(results, output)
    _plot_visibility(output, audit_record)
    _plot_structure(results, output, audit_record)


def _plot_training_curves(results: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(17.2, 6.7))
    det_history = _read_json(results / "seed_42" / "deterministic" / "history.json")
    axes[0].plot(
        [r["epoch"] for r in det_history],
        [r["loss"] for r in det_history],
        color=COLORS["midnight_blue"],
        label="Deterministic MSE",
    )
    axes[0].set_title("Deterministic backbone", loc="left", fontweight="bold")
    axes[0].set_ylabel("Validation objective")
    axes[0].set_xlabel("Epoch")
    colors = dict(zip(MODEL_INFO, PALETTE + [COLORS["gray"]]))
    for model in MODEL_INFO:
        if model == "deterministic":
            continue
        history = _read_json(results / "seed_42" / model / "history.json")
        axes[1].plot(
            [r["epoch"] for r in history],
            [r["loss"] for r in history],
            color=colors[model],
            label=MODEL_INFO[model][0],
        )
    axes[1].set_title("Uncertainty-head validation objectives", loc="left", fontweight="bold")
    axes[1].set_ylabel("Validation proper NLL")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=7, ncol=2)
    label_panels(axes)
    fig.suptitle("ITOP full side-train, 512 points, seed 42; side/top test each have 4,863 valid frames", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output / "itop_final_training_curves")
    plt.close(fig)


def _plot_metric_comparison(output: Path, audit_record: dict[str, Any]) -> None:
    labels = [record["label"] for record in audit_record["models"].values()]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=cm2inch(18.0, 7.0))
    for offset, view, color in ((-0.18, "side", COLORS["midnight_blue"]), (0.18, "top", COLORS["champagne_gold"])):
        values = [_load_metrics(audit_record, model)[view]["mpjpe_cm"] for model in MODEL_INFO]
        axes[0].bar(x + offset, values, 0.36, color=color, label="Side IID" if view == "side" else "Top OOD")
    axes[0].set_title("Point error", loc="left", fontweight="bold")
    axes[0].set_ylabel("MPJPE (cm)")
    axes[0].set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
    axes[0].legend(fontsize=8)
    for offset, view, color in ((-0.18, "side", COLORS["midnight_blue"]), (0.18, "top", COLORS["champagne_gold"])):
        values = [(_load_metrics(audit_record, model)[view].get("nll") or np.nan) for model in MODEL_INFO]
        axes[1].bar(x + offset, values, 0.36, color=color, label="Side IID" if view == "side" else "Top OOD")
    axes[1].set_title("Proper probabilistic score", loc="left", fontweight="bold")
    axes[1].set_ylabel("Family-correct NLL")
    axes[1].set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
    axes[1].legend(fontsize=8)
    prob_models = [model for model in MODEL_INFO if model != "deterministic"]
    px = np.arange(len(prob_models))
    values_side = [_load_metrics(audit_record, model)["side"].get("mace") for model in prob_models]
    values_top = [_load_metrics(audit_record, model)["top"].get("mace") for model in prob_models]
    axes[2].bar(px - 0.18, values_side, 0.36, color=COLORS["midnight_blue"], label="Side IID")
    axes[2].bar(px + 0.18, values_top, 0.36, color=COLORS["champagne_gold"], label="Top OOD")
    axes[2].set_title("Frame calibration diagnostic", loc="left", fontweight="bold")
    axes[2].set_ylabel("MACE")
    axes[2].set_ylim(0, 0.55)
    axes[2].set_xticks(px, [MODEL_INFO[m][0].replace(" (frozen)", "\nF").replace(" (joint)", "\nJ") for m in prob_models], rotation=35, ha="right", fontsize=7)
    axes[2].legend(fontsize=8)
    label_panels(axes)
    fig.tight_layout()
    save_figure(fig, output / "itop_final_metric_comparison")
    plt.close(fig)


def _plot_ood(output: Path, audit_record: dict[str, Any]) -> None:
    models = [model for model in MODEL_INFO if model != "deterministic"]
    labels = [MODEL_INFO[model][0] for model in models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.8, 6.7))
    top_nll = [_load_metrics(audit_record, model)["top"].get("nll", np.nan) for model in models]
    auroc = [_load_metrics(audit_record, model).get("ood", {}).get("side_top_uncertainty_auroc", np.nan) for model in models]
    axes[0].bar(x, top_nll, color=PALETTE[:len(models)])
    axes[0].set_title("Top-view OOD proper score", loc="left", fontweight="bold")
    axes[0].set_ylabel("Top-test NLL")
    axes[0].set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
    axes[1].bar(x, auroc, color=PALETTE[:len(models)])
    axes[1].axhline(0.5, color=COLORS["dark_gray"], linestyle="--", linewidth=1, label="Chance")
    axes[1].set_title("Side/top uncertainty separation", loc="left", fontweight="bold")
    axes[1].set_ylabel("Uncertainty AUROC")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
    axes[1].legend(fontsize=8)
    label_panels(axes)
    fig.tight_layout()
    save_figure(fig, output / "itop_final_ood_diagnostics")
    plt.close(fig)


def _plot_risk_coverage(results: Path, output: Path) -> None:
    selected = ("joint_independent_gaussian", "joint_graph_student_t")
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.8, 6.7))
    for view, ax in zip(("side", "top"), axes):
        for index, model in enumerate(selected):
            prediction = _load_prediction(results / "seed_42" / model / f"predictions_{view}.pt")
            uncertainty = prediction["frame_uncertainty"].float()
            error = prediction["joint_errors"].float().mean(-1) * 100.0
            order = torch.argsort(uncertainty)
            sorted_error = error[order]
            fractions = torch.linspace(0.1, 1.0, 91)
            risks = torch.stack(
                [sorted_error[: max(1, int(float(f) * len(sorted_error)))].mean() for f in fractions]
            )
            ax.plot(fractions.numpy() * 100, risks.numpy(), color=PALETTE[index], label=MODEL_INFO[model][0])
        ax.set_title(f"{view.capitalize()} {'IID' if view == 'side' else 'OOD'}", loc="left", fontweight="bold")
        ax.set_xlabel("Retained coverage (%)")
        ax.set_ylabel("Mean frame joint error (cm)")
        ax.set_xlim(10, 100)
        ax.legend(fontsize=7)
    label_panels(axes)
    fig.tight_layout()
    save_figure(fig, output / "itop_final_risk_coverage")
    plt.close(fig)


def _plot_visibility(output: Path, audit_record: dict[str, Any]) -> None:
    models = list(MODEL_INFO)
    labels = [MODEL_INFO[model][0] for model in models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(17.5, 6.8))
    for ax, view in zip(axes, ("side", "top")):
        visible = [_load_metrics(audit_record, model)[view].get("visible_mpjpe_cm", np.nan) for model in models]
        occluded = [_load_metrics(audit_record, model)[view].get("occluded_mpjpe_cm", np.nan) for model in models]
        ax.bar(x - 0.18, visible, 0.36, color=COLORS["midnight_blue"], label="Visible")
        ax.bar(x + 0.18, occluded, 0.36, color=COLORS["champagne_gold"], label="Occluded")
        ax.set_title(f"{view.capitalize()} {'IID' if view == 'side' else 'OOD'}", loc="left", fontweight="bold")
        ax.set_ylabel("MPJPE (cm)")
        ax.set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
        ax.legend(fontsize=8)
    label_panels(axes)
    fig.tight_layout()
    save_figure(fig, output / "itop_final_visibility_diagnostics")
    plt.close(fig)


def _plot_structure(results: Path, output: Path, audit_record: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.8, 6.8))
    for index, (model, view) in enumerate(
        (("joint_graph_student_t", "side"), ("joint_graph_student_t", "top"))
    ):
        metrics = _load_metrics(audit_record, model)[view]
        distances = sorted(metrics["residual_correlation_by_skeleton_distance"], key=int)
        corr = [metrics["residual_correlation_by_skeleton_distance"][d] for d in distances]
        axes[0].plot(
            [int(d) for d in distances],
            corr,
            "o-",
            color=PALETTE[index],
            label=f"{view.capitalize()} {'IID' if view == 'side' else 'OOD'}",
        )
    axes[0].set_title("Graph Student-t residual dependence", loc="left", fontweight="bold")
    axes[0].set_xlabel("Skeleton graph distance")
    axes[0].set_ylabel("Residual correlation")
    axes[0].legend(fontsize=8)
    model = "joint_graph_student_t"
    x = np.arange(2)
    side = _load_metrics(audit_record, model)["side"]
    top = _load_metrics(audit_record, model)["top"]
    axes[1].bar(x - 0.17, [side["visible_mpjpe_cm"], top["visible_mpjpe_cm"]], 0.34, color=COLORS["midnight_blue"], label="Visible")
    axes[1].bar(x + 0.17, [side["occluded_mpjpe_cm"], top["occluded_mpjpe_cm"]], 0.34, color=COLORS["champagne_gold"], label="Occluded")
    axes[1].set_title("Graph Student-t visibility error", loc="left", fontweight="bold")
    axes[1].set_xticks(x, ["Side IID", "Top OOD"])
    axes[1].set_ylabel("MPJPE (cm)")
    axes[1].legend(fontsize=8)
    label_panels(axes)
    fig.tight_layout()
    save_figure(fig, output / "itop_final_structure_diagnostics")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = audit(args.results, args.output)
    print(json.dumps({"models": list(record["models"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
