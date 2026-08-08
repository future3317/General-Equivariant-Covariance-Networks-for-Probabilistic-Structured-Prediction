"""Plot the controlled synthetic statistical-closure audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plotting import (
    COLORS,
    PALETTE,
    cm2inch,
    label_panels,
    save_figure,
    setup_tpami_style,
)

LABELS = {
    "full": "Full",
    "low_rank": "Low-rank",
    "isotypic_block": "Block",
    "graph_precision": "Graph",
}


def plot_cross_family_recovery(input_path: Path, output_path: Path) -> None:
    """Render the controlled teacher/learner matrix as a log-scaled heatmap."""
    data = json.loads(input_path.read_text(encoding="utf-8"))
    families = data["matrix_families"]
    rows = data["rows"]
    means = np.empty((len(families), len(families)), dtype=float)
    stds = np.empty_like(means)
    for teacher_index, teacher in enumerate(families):
        for learner_index, learner in enumerate(families):
            values = [
                float(row["covariance_relative_error"])
                for row in rows
                if row["teacher_family"] == teacher
                and row["learner_family"] == learner
            ]
            if len(values) != 3:
                raise ValueError(
                    f"expected three seeds for {teacher}->{learner}, got {len(values)}"
                )
            means[teacher_index, learner_index] = np.mean(values)
            stds[teacher_index, learner_index] = np.std(values, ddof=1)

    setup_tpami_style()
    fig, axis = plt.subplots(figsize=cm2inch(11.8, 8.8))
    lower = max(float(means.min()) * 0.75, np.finfo(float).tiny)
    upper = float(means.max()) * 1.08
    image = axis.imshow(
        means,
        cmap="magma",
        norm=LogNorm(vmin=lower, vmax=upper),
        aspect="equal",
    )
    labels = [LABELS[family] for family in families]
    axis.set_xticks(np.arange(len(families)), labels)
    axis.set_yticks(np.arange(len(families)), labels)
    axis.set_xlabel("Learner family")
    axis.set_ylabel("Teacher family")
    axis.set_title("Family match controls scatter recovery", loc="left", fontweight="bold")
    axis.grid(False)
    for row_index in range(len(families)):
        for column_index in range(len(families)):
            value = means[row_index, column_index]
            text_color = "white" if value < 0.08 else "#20242A"
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}\n$\\pm${stds[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.2,
                fontweight="bold" if row_index == column_index else "normal",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.052, pad=0.05)
    colorbar.set_label("Relative scatter error (log scale)")
    axis.tick_params(length=0)
    fig.subplots_adjust(left=0.20, right=0.88, bottom=0.16, top=0.88)
    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cross-family-input",
        type=Path,
        help="Optional cross-family recovery JSON produced by the same benchmark.",
    )
    parser.add_argument(
        "--cross-family-output",
        type=Path,
        help="Output basename for the optional cross-family heatmap.",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["rows"]
    families = [name for name in LABELS if any(row["family"] == name for row in rows)]
    grouped = [[row for row in rows if row["family"] == name] for name in families]

    setup_tpami_style()
    fig, axes = plt.subplots(1, 3, figsize=cm2inch(18, 6.4))
    x = np.arange(len(families))
    colors = [PALETTE[i] for i in range(len(families))]

    # (a) Recovery is a lower-is-better metric; show seed dispersion explicitly.
    means = [np.mean([row["covariance_relative_error"] for row in group]) for group in grouped]
    stds = [np.std([row["covariance_relative_error"] for row in group], ddof=1) for group in grouped]
    axes[0].errorbar(x, means, yerr=stds, fmt="none", ecolor=COLORS["dark_gray"], capsize=3, zorder=2)
    axes[0].scatter(x, means, c=colors, s=42, zorder=3)
    axes[0].set_title("Scatter recovery")
    axes[0].set_ylabel("Relative scatter error")
    axes[0].set_xticks(x, [LABELS[name] for name in families])
    axes[0].set_ylim(bottom=0)

    # (b) Coverage is compared with the declared Student-t reference levels.
    offsets = (-0.12, 0.12)
    for offset, key, nominal, marker in zip(offsets, ("coverage_90", "coverage_95"), (0.90, 0.95), ("o", "s")):
        values = [np.mean([row[key] for row in group]) for group in grouped]
        spread = [np.std([row[key] for row in group], ddof=1) for group in grouped]
        axes[1].errorbar(x + offset, values, yerr=spread, fmt=marker, color=COLORS["midnight_blue"] if key.endswith("90") else COLORS["champagne_gold"], capsize=3, label=f"Nominal {int(nominal * 100)}%")
        axes[1].axhline(nominal, color=COLORS["dark_gray"], linestyle="--", linewidth=0.9)
    axes[1].set_title("Coverage")
    axes[1].set_ylabel("Coverage")
    axes[1].set_xticks(x, [LABELS[name] for name in families])
    axes[1].set_ylim(0.84, 1.01)

    # (c) Both diagnostics are absolute invariance errors; use a log scale.
    eq = [max(row["equivariance_max_abs"] for row in group) for group in grouped]
    basis = [max(row["basis_change_nll_abs_error"] for row in group) for group in grouped]
    axes[2].plot(x, eq, marker="o", color=COLORS["midnight_blue"], label="O(3) output")
    axes[2].plot(x, basis, marker="s", color=COLORS["champagne_gold"], label="Orthogonal NLL")
    axes[2].set_yscale("log")
    axes[2].set_title("Invariance")
    axes[2].set_ylabel("Absolute error")
    axes[2].set_xticks(x, [LABELS[name] for name in families])

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=20)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    label_panels(axes, x=-0.12, y=1.04)
    fig.suptitle(
        "Controlled covariance recovery: 128 contexts, 32 repeats, 3 seeds, Student-$t$ $\\nu=5$",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.06, 0.06, 1, 0.96))
    save_figure(fig, args.output)
    plt.close(fig)
    if (args.cross_family_input is None) != (args.cross_family_output is None):
        parser.error(
            "--cross-family-input and --cross-family-output must be provided together"
        )
    if args.cross_family_input is not None:
        plot_cross_family_recovery(
            args.cross_family_input,
            args.cross_family_output,
        )


if __name__ == "__main__":
    main()
