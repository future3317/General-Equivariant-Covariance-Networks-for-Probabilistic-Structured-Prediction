"""Render the dielectric family-by-radial-law design-space audit.

This figure is an appendix visualization only.  It reads the already
aggregated factorial JSON and never retrains or recomputes a metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plotting import FAMILY_COLORS, LAW_COLORS, cm2inch, save_figure, setup_tpami_style

ACTIVE_COORDINATES = {
    "isotropic": 1,
    "block": 2,
    "low_rank": 13,
    "full": 21,
}
FAMILY_LABELS = {
    "isotropic": "Isotropic",
    "block": "Block",
    "low_rank": "Low-rank",
    "full": "Full",
}
LAW_LABELS = {"gaussian": "Gaussian", "student_t": r"Student-$t$"}


def _load_aggregate(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["aggregate"]
    expected = {
        (family, law)
        for family in ACTIVE_COORDINATES
        for law in LAW_LABELS
    }
    observed = {(row["family"], row["distribution"]) for row in rows}
    if observed != expected or len(rows) != 8:
        raise ValueError(f"expected exactly eight factorial arms, got {sorted(observed)}")
    return rows


def plot_design_space(input_path: Path, output_path: Path) -> None:
    rows = _load_aggregate(input_path)
    setup_tpami_style()
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(18.3, 7.0), sharex=True)
    specifications = (
        ("nll_mean", "nll_std", "Normalized test NLL"),
        ("energy_score_mean", "energy_score_std", "Energy score"),
    )
    x = np.asarray([ACTIVE_COORDINATES[row["family"]] for row in rows], dtype=float)
    for axis, (value_key, spread_key, ylabel) in zip(axes, specifications):
        for row, x_value in zip(rows, x):
            family = row["family"]
            law = row["distribution"]
            marker = "o" if law == "student_t" else "s"
            axis.errorbar(
                x_value,
                row[value_key],
                yerr=row[spread_key],
                fmt=marker,
                color=FAMILY_COLORS[family],
                ecolor=FAMILY_COLORS[family],
                elinewidth=0.8,
                capsize=2.2,
                markersize=6.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=f"{FAMILY_LABELS[family]} / {LAW_LABELS[law]}",
                zorder=3,
            )
        axis.set_xlabel("Active operator coordinates")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted(set(x)), ["1", "2", "13", "21"])
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.3, alpha=0.45)
        axis.set_axisbelow(True)
    axes[0].set_title("Family and radial-law choices", loc="left", fontweight="bold")
    axes[1].set_title("Proper-score view", loc="left", fontweight="bold")

    family_handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="none", markersize=6,
            markerfacecolor=FAMILY_COLORS[family], markeredgecolor="white",
            label=FAMILY_LABELS[family],
        )
        for family in ("isotropic", "block", "low_rank", "full")
    ]
    law_handles = [
        plt.Line2D(
            [], [], marker=marker, linestyle="none", markersize=6,
            color=LAW_COLORS[law], label=LAW_LABELS[law],
        )
        for law, marker in (("gaussian", "s"), ("student_t", "o"))
    ]
    axes[1].legend(
        handles=family_handles + law_handles,
        ncol=2,
        loc="upper left",
        frameon=False,
        fontsize=6.5,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    fig.text(
        0.5,
        0.01,
        "Points show three-seed means; error bars show sample standard deviation. "
        "Colors encode operator family and markers encode radial law.",
        ha="center",
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.84, wspace=0.28)
    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_design_space(args.input, args.output)


if __name__ == "__main__":
    main()
