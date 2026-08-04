"""Generate the two compact ITOP figures used by the TPAMI manuscript."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.itop_dataset import ITOP_SKELETON_EDGES
from plotting import COLORS, cm2inch, label_panels, save_figure, setup_tpami_style

MODEL_LABELS = {
    "deterministic": "Det.",
    "frozen_independent_gaussian": "Indep-G",
    "frozen_graph_gaussian": "Graph-G",
    "frozen_graph_student_t": "Graph-t",
}


def _metrics(root: Path, model: str) -> dict:
    path = root / "seed_42" / model / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _risk_curve(uncertainty: np.ndarray, error: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    order = np.argsort(uncertainty)
    cumulative = np.cumsum(error[order])
    counts = np.maximum(1, np.rint(fractions * len(error)).astype(int))
    return cumulative[counts - 1] / counts


def _bootstrap_risk_band(
    uncertainty: np.ndarray,
    error: np.ndarray,
    fractions: np.ndarray,
    *,
    seed: int = 42,
    repeats: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(repeats):
        indices = rng.integers(0, len(error), size=len(error))
        curves.append(_risk_curve(uncertainty[indices], error[indices], fractions))
    samples = np.stack(curves)
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def _graph_distance_counts(num_nodes: int) -> dict[int, int]:
    adjacency = [[] for _ in range(num_nodes)]
    for source, target in ITOP_SKELETON_EDGES:
        adjacency[source].append(target)
        adjacency[target].append(source)
    counts: dict[int, int] = {}
    for source in range(num_nodes):
        distances = [-1] * num_nodes
        distances[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        for target in range(source + 1, num_nodes):
            counts[distances[target]] = counts.get(distances[target], 0) + 1
    return counts


def plot_overview(root: Path, output: Path) -> None:
    setup_tpami_style()
    models = tuple(MODEL_LABELS)
    records = {model: _metrics(root, model) for model in models}
    fig, axes = plt.subplots(1, 3, figsize=cm2inch(18.0, 7.0))
    y = np.arange(len(models))[::-1]
    side_color, top_color = COLORS["midnight_blue"], COLORS["champagne_gold"]

    for axis, metric, label, title in zip(
        axes[:2],
        ("mpjpe_cm", "nll"),
        ("MPJPE (cm)", "Proper NLL"),
        ("Point error", "Probabilistic fit"),
    ):
        for index, model in enumerate(models):
            side = records[model]["side"].get(metric)
            top = records[model]["top"].get(metric)
            if side is not None:
                axis.plot(side, y[index] + 0.11, "o", color=side_color, ms=6)
                axis.text(side, y[index] + 0.18, f"{side:.1f}", color=side_color, fontsize=8)
            if top is not None:
                axis.plot(top, y[index] - 0.11, "o", color=top_color, ms=6)
                axis.text(top, y[index] - 0.20, f"{top:.1f}", color=top_color, fontsize=8)
        axis.set_yticks(y, [MODEL_LABELS[model] for model in models])
        axis.set_xlabel(label)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x")

    auroc = [records[model]["ood"].get("side_top_uncertainty_auroc", np.nan) for model in models]
    axes[2].scatter(auroc, y, color=COLORS["navy_light"], s=36, zorder=3)
    for value, ypos in zip(auroc, y):
        axes[2].text(value + 0.025, ypos, f"{value:.3f}", va="center", fontsize=8)
    axes[2].axvline(0.5, color=COLORS["dark_gray"], linestyle="--", linewidth=1)
    axes[2].set_xlim(0, 1.08)
    axes[2].set_yticks(y, [MODEL_LABELS[model] for model in models])
    axes[2].set_xlabel("Side/top AUROC")
    axes[2].set_title("OOD ranking", loc="left", fontweight="bold")
    axes[2].grid(axis="x")

    for axis in axes:
        axis.tick_params(axis="y", length=0)
    axes[0].plot([], [], "o", color=side_color, label="Side IID")
    axes[0].plot([], [], "o", color=top_color, label="Top OOD")
    axes[0].legend(loc="lower right", fontsize=8)
    label_panels(axes, x=-0.12, y=1.06, fontsize=11)
    fig.suptitle("ITOP full side-train audit", y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output / "itop_final_overview", formats=("pdf",))
    plt.close(fig)


def plot_structure(root: Path, output: Path) -> None:
    setup_tpami_style()
    model_root = root / "seed_42" / "frozen_graph_student_t"
    metrics = _metrics(root, "frozen_graph_student_t")
    counts = _graph_distance_counts(15)
    distances = sorted(metrics["side"]["residual_correlation_by_skeleton_distance"], key=int)

    fig, axes = plt.subplots(1, 2, figsize=cm2inch(18.0, 7.0))
    for view, color, label in (
        ("side", COLORS["midnight_blue"], "Side IID"),
        ("top", COLORS["champagne_gold"], "Top OOD"),
    ):
        values = [metrics[view]["residual_correlation_by_skeleton_distance"][d] for d in distances]
        axes[0].plot([int(d) for d in distances], values, "o-", color=color, label=label)
    axes[0].set_xlabel("Skeleton graph distance")
    axes[0].set_ylabel("Residual correlation")
    axes[0].set_title("Structured residual dependence", loc="left", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].set_xticks([int(d) for d in distances])
    values = [
        metrics["side"]["residual_correlation_by_skeleton_distance"][distance]
        for distance in distances
    ]
    for distance, value in zip(distances, values):
        axes[0].annotate(f"n={counts[int(distance)]}", (int(distance), value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)

    fractions = np.linspace(0.1, 1.0, 10)
    for view, color, label in (
        ("side", COLORS["midnight_blue"], "Side IID"),
        ("top", COLORS["champagne_gold"], "Top OOD"),
    ):
        payload = torch.load(model_root / f"predictions_{view}.pt", map_location="cpu", weights_only=True)
        uncertainty = payload["frame_uncertainty"].numpy()
        error = payload["joint_errors"].float().mean(dim=-1).numpy() * 100.0
        curve = _risk_curve(uncertainty, error, fractions)
        lower, upper = _bootstrap_risk_band(uncertainty, error, fractions)
        axes[1].plot(fractions * 100, curve, "o-", color=color, label=label)
        axes[1].fill_between(fractions * 100, lower, upper, color=color, alpha=0.15)
    axes[1].set_xlabel("Retained coverage (%)")
    axes[1].set_ylabel("Mean joint error (cm)")
    axes[1].set_title("Selective risk with 95% bootstrap bands", loc="left", fontweight="bold")
    axes[1].legend(fontsize=8)
    label_panels(axes, x=-0.12, y=1.06, fontsize=11)
    fig.tight_layout()
    save_figure(fig, output / "itop_final_structure", formats=("pdf",))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_overview(args.results, args.output)
    plot_structure(args.results, args.output)
    print(f"ITOP final figures written to {args.output}")


if __name__ == "__main__":
    main()
