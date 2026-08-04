"""Generate publication figures for the controlled ITOP development panel.

The panel is intentionally small (one seed, 1/16 side-train, 256 points).  The
figures therefore document the compiler execution path and diagnostics rather
than claiming a final pose-estimation benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.itop_dataset import ITOP_JOINT_NAMES, ITOP_SKELETON_EDGES
from plotting import (
    COLORS,
    PALETTE,
    cm2inch,
    label_panels,
    save_figure,
    setup_tpami_style,
)

METHODS = {
    "deterministic": "Deterministic",
    "joint_independent_gaussian": "Independent Gaussian",
    "joint_graph_student_t": "Graph Student-t",
}


def _load(root: Path, model: str, name: str):
    return json.loads((root / "seed_42" / model / name).read_text(encoding="utf-8"))


def _subtitle() -> str:
    return "ITOP 1/16 development panel; one seed; Ntrain=2,487, Nside-test=Ntop-test=4,863"


def plot_training_curves(root: Path, out: Path) -> None:
    setup_tpami_style()
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.5, 6.7))
    det = _load(root, "deterministic", "history.json")
    axes[0].plot([x["epoch"] for x in det], [x["loss"] for x in det], color=COLORS["midnight_blue"], label="Deterministic MSE")
    axes[0].set_title("Deterministic training", loc="left", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_xlabel("Epoch")
    for model, color in (("joint_independent_gaussian", COLORS["champagne_gold"]), ("joint_graph_student_t", COLORS["midnight_blue"])):
        h = _load(root, model, "history.json")
        axes[1].plot([x["epoch"] for x in h], [x["loss"] for x in h], color=color, label=METHODS[model])
    axes[1].set_title("Joint uncertainty training", loc="left", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Validation proper NLL")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=8)
    label_panels(axes, x=-0.10, y=1.08, fontsize=10)
    fig.suptitle(_subtitle(), fontsize=8, y=0.995)
    fig.subplots_adjust(top=0.83)
    fig.tight_layout()
    save_figure(fig, out / "itop_training_curves")
    plt.close(fig)


def plot_metric_comparison(root: Path, out: Path) -> None:
    setup_tpami_style()
    metrics = {m: _load(root, m, "metrics.json") for m in METHODS}
    fig, axes = plt.subplots(1, 3, figsize=cm2inch(17.5, 6.8))
    x = np.arange(2)
    width = 0.25
    for i, (model, color) in enumerate(zip(METHODS, PALETTE[:3])):
        vals = [metrics[model][view]["mpjpe_cm"] for view in ("side", "top")]
        axes[0].bar(x + (i - 1) * width, vals, width, label=METHODS[model], color=color)
    axes[0].set_xticks(x, ["Side IID", "Top OOD"])
    axes[0].set_ylabel("MPJPE (cm)")
    axes[0].set_title("Point error", loc="left", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=7)

    for i, (model, color) in enumerate(zip(("joint_independent_gaussian", "joint_graph_student_t"), PALETTE[1:3])):
        vals = [metrics[model][view]["nll"] for view in ("side", "top")]
        axes[1].bar(x + (i - 0.5) * width, vals, width, label=METHODS[model], color=color)
    axes[1].set_xticks(x, ["Side IID", "Top OOD"])
    axes[1].set_ylabel("Student-t/Gaussian NLL")
    axes[1].set_title("Proper probabilistic loss", loc="left", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=7)

    vals = [metrics[m]["side"]["mace"] for m in ("joint_independent_gaussian", "joint_graph_student_t")]
    vals_top = [metrics[m]["top"]["mace"] for m in ("joint_independent_gaussian", "joint_graph_student_t")]
    xx = np.arange(2)
    axes[2].bar(xx - width / 2, vals, width, color=COLORS["midnight_blue"], label="Side IID")
    axes[2].bar(xx + width / 2, vals_top, width, color=COLORS["champagne_gold"], label="Top OOD")
    axes[2].set_xticks(xx, ["Independent", "Graph Student-t"])
    axes[2].set_ylabel("MACE")
    axes[2].set_ylim(0, 0.55)
    axes[2].set_title("Calibration diagnostic", loc="left", fontsize=10, fontweight="bold")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=8)
    label_panels(axes, x=-0.08, y=1.08, fontsize=10)
    fig.suptitle(_subtitle(), fontsize=8, y=0.995)
    fig.subplots_adjust(top=0.83)
    fig.tight_layout()
    save_figure(fig, out / "itop_metric_comparison")
    plt.close(fig)


def plot_ood_diagnostics(root: Path, out: Path) -> None:
    setup_tpami_style()
    metrics = {m: _load(root, m, "metrics.json") for m in ("joint_independent_gaussian", "joint_graph_student_t")}
    names = [METHODS[m] for m in metrics]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.5, 6.5))
    top_nll = [metrics[m]["ood"]["side_to_top_nll"] for m in metrics]
    auroc = [metrics[m]["ood"]["side_top_uncertainty_auroc"] for m in metrics]
    axes[0].bar(x, top_nll, color=[COLORS["midnight_blue"], COLORS["champagne_gold"]])
    axes[0].set_xticks(x, names, rotation=12, ha="right")
    axes[0].set_ylabel("Top-test NLL")
    axes[0].set_title("Cross-view OOD loss", loc="left", fontsize=10, fontweight="bold")
    axes[1].bar(x, auroc, color=[COLORS["midnight_blue"], COLORS["champagne_gold"]])
    axes[1].axhline(0.5, color=COLORS["dark_gray"], linestyle="--", linewidth=1, label="Chance")
    axes[1].set_xticks(x, names, rotation=12, ha="right")
    axes[1].set_ylabel("Side/top uncertainty AUROC")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("OOD uncertainty separation", loc="left", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=8)
    label_panels(axes, x=-0.10, y=1.08, fontsize=10)
    fig.suptitle(_subtitle(), fontsize=8, y=0.995)
    fig.subplots_adjust(top=0.83)
    fig.tight_layout()
    save_figure(fig, out / "itop_ood_diagnostics")
    plt.close(fig)


def plot_risk_coverage(root: Path, out: Path) -> None:
    setup_tpami_style()
    fig, ax = plt.subplots(figsize=cm2inch(10.5, 7.0))
    for view, color in (("side", COLORS["midnight_blue"]), ("top", COLORS["champagne_gold"])):
        payload = torch.load(root / "seed_42" / "joint_graph_student_t" / f"predictions_{view}.pt", map_location="cpu", weights_only=True)
        uncertainty = payload["frame_uncertainty"].float()
        error = payload["joint_errors"].float().mean(dim=-1) * 100.0
        order = torch.argsort(uncertainty)
        sorted_error = error[order]
        fractions = torch.linspace(0.1, 1.0, 91)
        risks = torch.stack([sorted_error[: max(1, int(float(f) * len(sorted_error)))].mean() for f in fractions]).numpy()
        ax.plot(fractions.numpy() * 100, risks, color=color, label=f"{view} ({'IID' if view == 'side' else 'OOD'})")
    ax.set_xlabel("Retained coverage (%)")
    ax.set_ylabel("Mean joint error (cm)")
    ax.set_title("Graph Student-t risk-coverage\n" + _subtitle(), loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(10, 100)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save_figure(fig, out / "itop_risk_coverage")
    plt.close(fig)


def plot_structure(root: Path, out: Path) -> None:
    setup_tpami_style()
    metrics = _load(root, "joint_graph_student_t", "metrics.json")
    distances = sorted(metrics["side"]["residual_correlation_by_skeleton_distance"], key=int)
    corr = [metrics["side"]["residual_correlation_by_skeleton_distance"][d] for d in distances]
    vis = [metrics[v]["visible_mpjpe_cm"] for v in ("side", "top")]
    occ = [metrics[v]["occluded_mpjpe_cm"] for v in ("side", "top")]
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.5, 6.5))
    axes[0].plot([int(d) for d in distances], corr, "o-", color=COLORS["midnight_blue"])
    axes[0].set_xlabel("Skeleton graph distance")
    axes[0].set_ylabel("Residual correlation")
    axes[0].set_title("Residual dependence by graph distance", loc="left", fontsize=10, fontweight="bold")
    x = np.arange(2)
    axes[1].bar(x - 0.17, vis, 0.34, color=COLORS["midnight_blue"], label="Visible")
    axes[1].bar(x + 0.17, occ, 0.34, color=COLORS["champagne_gold"], label="Occluded")
    axes[1].set_xticks(x, ["Side IID", "Top OOD"])
    axes[1].set_ylabel("MPJPE (cm)")
    axes[1].set_title("Visible/occluded joint error", loc="left", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=8)
    label_panels(axes, x=-0.10, y=1.08, fontsize=10)
    fig.suptitle(_subtitle(), fontsize=8, y=0.995)
    fig.subplots_adjust(top=0.83)
    fig.tight_layout()
    save_figure(fig, out / "itop_structure_diagnostics")
    plt.close(fig)


def plot_pose_examples(root: Path, out: Path) -> None:
    setup_tpami_style()
    fig = plt.figure(figsize=cm2inch(16.5, 7.8))
    for panel, view in enumerate(("side", "top"), start=1):
        payload = torch.load(root / "seed_42" / "joint_graph_student_t" / f"predictions_{view}.pt", map_location="cpu", weights_only=True)
        uncertainty = payload["joint_uncertainty"].float()
        idx = int(torch.argsort(payload["frame_uncertainty"].float())[len(payload["frame_uncertainty"]) // 2])
        pred = payload["mean"][idx].reshape(15, 3).numpy() * 100.0
        target = payload["target"][idx].reshape(15, 3).numpy() * 100.0
        u = uncertainty[idx].numpy()
        u = (u - u.min()) / (u.max() - u.min() + 1e-9)
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        for a, b in ITOP_SKELETON_EDGES:
            ax.plot(*np.stack([target[a], target[b]]).T, color=COLORS["gray"], linewidth=1.1, alpha=0.75)
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], color=COLORS["dark_gray"], s=15, label="Target")
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=u, cmap="Blues", vmin=0, vmax=1, s=35 + 45 * u, edgecolor=COLORS["champagne_gold"], linewidth=0.7, label="Prediction")
        for j, name in enumerate(ITOP_JOINT_NAMES):
            if j in (0, 6, 13):
                ax.text(*pred[j], name, fontsize=6)
        all_xyz = np.concatenate([target, pred])
        lo, hi = all_xyz.min(0), all_xyz.max(0)
        center = (lo + hi) / 2
        radius = max((hi - lo).max() / 2, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("x (cm)", fontsize=8)
        ax.set_ylabel("y (cm)", fontsize=8)
        ax.set_zlabel("z (cm)", fontsize=8)
        ax.set_title(f"{view.capitalize()} example ({'IID' if view == 'side' else 'OOD'})\nmedian frame uncertainty", fontsize=10, loc="left")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Illustrative ITOP uncertainty visualization\nNode size/color denotes predicted marginal uncertainty; this is not a calibration claim", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out / "itop_pose_uncertainty_examples")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/itop_dev_1of16_20260730"))
    parser.add_argument("--output", type=Path, default=Path("figures/itop_dev_1of16_20260730"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_training_curves(args.results, args.output)
    plot_metric_comparison(args.results, args.output)
    plot_ood_diagnostics(args.results, args.output)
    plot_risk_coverage(args.results, args.output)
    plot_structure(args.results, args.output)
    plot_pose_examples(args.results, args.output)
    print(f"ITOP figures written to {args.output}")


if __name__ == "__main__":
    main()
