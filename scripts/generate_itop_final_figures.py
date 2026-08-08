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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.itop_dataset import ITOP_SKELETON_EDGES
from plotting import COLORS, cm2inch, label_panels, save_figure, setup_tpami_style

MODEL_LABELS = {
    "deterministic": "Det.",
    "frozen_full_student_t": "Full-t",
    "frozen_independent_gaussian": "Indep-G",
    "frozen_independent_student_t": "Indep-t",
    "frozen_low_rank_student_t": "LR-t",
    "frozen_graph_gaussian": "Graph-G",
    "frozen_graph_student_t": "Graph-t",
}

METHOD_COLORS = {
    "deterministic": COLORS["dark_gray"],
    "frozen_full_student_t": COLORS["violet"],
    "frozen_independent_gaussian": COLORS["red_strong"],
    "frozen_independent_student_t": COLORS["red_2"],
    "frozen_low_rank_student_t": COLORS["teal"],
    "frozen_graph_gaussian": COLORS["green_3"],
    "frozen_graph_student_t": COLORS["blue_main"],
}

ACTIVE_COORDS = {
    "frozen_full_student_t": 1035,
    "frozen_independent_gaussian": 90,
    "frozen_independent_student_t": 90,
    "frozen_low_rank_student_t": 181,
    "frozen_graph_gaussian": 174,
    "frozen_graph_student_t": 174,
}


def _seed_root(root: Path) -> Path:
    """Locate the canonical seed artifact under a result or figure-input root."""
    candidates = (root / "seed_42", root / "figures_input" / "seed_42")
    for candidate in candidates:
        if (candidate / "frozen_graph_student_t" / "metrics.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find seed_42 artifacts under "
        f"{root} or {root / 'figures_input'}"
    )


def _metrics(root: Path, model: str) -> dict:
    path = _seed_root(root) / model / "metrics.json"
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


def _bootstrap_distance_intervals(
    residual: np.ndarray,
    distances: list[str],
    *,
    seed: int = 42,
    repeats: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap graph-distance residual correlations over frames."""
    rng = np.random.default_rng(seed)
    num_frames, num_joints = residual.shape[:2]
    adjacency = [[] for _ in range(num_joints)]
    for source, target in ITOP_SKELETON_EDGES:
        adjacency[source].append(target)
        adjacency[target].append(source)
    pairs: dict[int, list[tuple[int, int]]] = {}
    for source in range(num_joints):
        distance = [-1] * num_joints
        distance[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        for target in range(source + 1, num_joints):
            if distance[target] > 0:
                pairs.setdefault(distance[target], []).append((source, target))

    bootstraps = []
    for _ in range(repeats):
        indices = rng.integers(0, num_frames, size=num_frames)
        sample = residual[indices]
        centered = sample - sample.mean(axis=0, keepdims=True)
        covariance = np.einsum("fja,fka->jk", centered, centered)
        scale = np.sqrt(np.maximum(np.diag(covariance), np.finfo(float).eps))
        correlation = covariance / (scale[:, None] * scale[None, :])
        bootstraps.append(
            [
                np.mean(
                    [
                        correlation[source, target]
                        for source, target in pairs[int(distance)]
                    ]
                )
                for distance in distances
            ]
        )
    values = np.asarray(bootstraps)
    return np.percentile(values, 2.5, axis=0), np.percentile(values, 97.5, axis=0)


def plot_overview(root: Path, output: Path) -> None:
    setup_tpami_style()
    seed_root = _seed_root(root)
    models = tuple(
        model for model in MODEL_LABELS if (seed_root / model).is_dir()
    )
    records = {model: _metrics(root, model) for model in models}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=cm2inch(18.2, 7.4),
        gridspec_kw={"width_ratios": (1.36, 0.92)},
    )
    probabilistic_models = tuple(
        model
        for model in models
        if records[model]["side"].get("nll") is not None
        and records[model]["top"].get("nll") is not None
    )
    axis = axes[0]
    for model in probabilistic_models:
        side_nll = records[model]["side"]["nll"]
        top_nll = records[model]["top"]["nll"]
        is_gaussian = "gaussian" in model
        axis.scatter(
            side_nll,
            top_nll,
            s=34 + 2.2 * np.sqrt(ACTIVE_COORDS[model]),
            marker="s" if is_gaussian else "o",
            color=METHOD_COLORS[model],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        axis.annotate(
            MODEL_LABELS[model],
            (side_nll, top_nll),
            xytext=(5, 4),
            textcoords="offset points",
            color=METHOD_COLORS[model],
            fontsize=7.5,
            fontweight="bold",
        )
    axis.set_yscale("log")
    axis.set_xlim(-77, -13)
    axis.set_ylim(1.8, 1500)
    axis.set_yticks((2, 10, 50, 200, 1000), ("2", "10", "50", "200", "1000"))
    axis.set_xlabel("Side IID proper NLL")
    axis.set_ylabel("Top OOD proper NLL (log scale)")
    axis.set_title("Proper-score trade-off", loc="left", fontweight="bold")
    axis.grid(which="major", alpha=0.65)
    family_handles = (
        Line2D(
            [], [], marker="o", color="none", markerfacecolor=COLORS["dark_gray"],
            markeredgecolor="white", markersize=5, label="Student-$t$",
        ),
        Line2D(
            [], [], marker="s", color="none", markerfacecolor=COLORS["dark_gray"],
            markeredgecolor="white", markersize=5, label="Gaussian",
        ),
    )
    axis.legend(
        handles=family_handles,
        title=r"Family (area $\propto$ coordinates)",
        loc="lower right",
        fontsize=6.5,
        title_fontsize=6.5,
        frameon=True,
        handletextpad=0.4,
        borderpad=0.4,
    )

    auroc_models = tuple(
        model
        for model in probabilistic_models
        if np.isfinite(records[model]["ood"].get("side_top_uncertainty_auroc", np.nan))
    )
    y = np.arange(len(auroc_models))[::-1]
    axis = axes[1]
    axis.axvline(
        0.5,
        color=COLORS["dark_gray"],
        linestyle="--",
        linewidth=1.0,
    )
    for model, value, ypos in zip(
        auroc_models,
        (records[model]["ood"]["side_top_uncertainty_auroc"] for model in auroc_models),
        y,
    ):
        color = METHOD_COLORS[model]
        axis.hlines(ypos, 0.5, value, color=color, linewidth=2.1, alpha=0.85, zorder=2)
        axis.scatter(
            value,
            ypos,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=43,
            zorder=3,
        )
        axis.annotate(
            f"{value:.3f}",
            (value, ypos),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.2,
            color=color,
        )
    axis.set_xlim(0, 1.06)
    axis.set_yticks(y, [MODEL_LABELS[model] for model in auroc_models])
    axis.set_xlabel("Side/top uncertainty AUROC")
    axis.set_title("Single-seed OOD ranking", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.65)
    axis.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.20, top=0.86, wspace=0.35)
    save_figure(fig, output / "itop_final_overview", formats=("pdf", "png"))
    plt.close(fig)


def plot_structure(root: Path, output: Path) -> None:
    setup_tpami_style()
    model_root = _seed_root(root) / "frozen_graph_student_t"
    metrics = _metrics(root, "frozen_graph_student_t")
    counts = _graph_distance_counts(15)
    distances = sorted(
        metrics["side"]["residual_correlation_by_skeleton_distance"], key=int
    )
    distance_values = [int(distance) for distance in distances]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=cm2inch(18.0, 7.4),
        gridspec_kw={"width_ratios": (1.05, 1.0)},
    )
    for view, color, label in (
        ("side", COLORS["blue_main"], "Side IID"),
        ("top", COLORS["red_strong"], "Top OOD"),
    ):
        values = [
            metrics[view]["residual_correlation_by_skeleton_distance"][d]
            for d in distances
        ]
        prediction = torch.load(
            model_root / f"predictions_{view}.pt",
            map_location="cpu",
            weights_only=True,
        )
        residual = (
            prediction["target"] - prediction["mean"]
        ).reshape(-1, 15, 3).numpy()
        lower, upper = _bootstrap_distance_intervals(residual, distances)
        axes[0].errorbar(
            distance_values,
            values,
            yerr=(np.asarray(values) - lower, upper - np.asarray(values)),
            fmt="o-",
            color=color,
            capsize=2.5,
            linewidth=1.8,
            markersize=5,
            label=label,
        )
    axes[0].set_xlabel("Skeleton graph distance")
    axes[0].set_ylabel("Residual correlation")
    axes[0].set_title("Structured residual dependence", loc="left", fontweight="bold")
    axes[0].set_ylim(-0.05, 0.95)
    axes[0].set_xticks(
        distance_values, [f"{d}\n(n={counts[d]})" for d in distance_values]
    )
    fractions = np.linspace(0.1, 1.0, 10)
    for view, color, label in (
        ("side", COLORS["blue_main"], "Side IID"),
        ("top", COLORS["red_strong"], "Top OOD"),
    ):
        payload = torch.load(model_root / f"predictions_{view}.pt", map_location="cpu", weights_only=True)
        uncertainty = payload["frame_uncertainty"].numpy()
        error = payload["joint_errors"].float().mean(dim=-1).numpy() * 100.0
        curve = _risk_curve(uncertainty, error, fractions)
        lower, upper = _bootstrap_risk_band(uncertainty, error, fractions)
        axes[1].plot(
            fractions * 100,
            curve,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=4.5,
            label=label,
        )
        axes[1].fill_between(
            fractions * 100,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    axes[1].set_xlabel("Retained coverage (%)")
    axes[1].set_ylabel("Mean joint error (cm)")
    axes[1].set_title("Selective risk with 95% bootstrap bands", loc="left", fontweight="bold")
    axes[1].set_ylim(
        min(0.0, float(np.nanmin(lower)) - 1.0),
        float(np.nanmax(upper)) + 1.0,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        handlelength=2.0,
        columnspacing=1.8,
    )
    label_panels(axes, x=-0.14, y=1.07, fontsize=10)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.20, top=0.82, wspace=0.31)
    save_figure(fig, output / "itop_final_structure", formats=("pdf", "png"))
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
