"""Plot the downloaded dielectric spectral-window sensitivity audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plotting import COLORS, cm2inch, save_figure, setup_tpami_style

RUNS = (
    ("seed42_b64_spectral_window_v2", r"Window $[-4,4]$", COLORS["red_strong"]),
    ("final_student_t_h64_window2", r"Window $[-2,2]$", COLORS["blue_main"]),
    (
        "final_student_t_centered_b128",
        r"Centered: shape $[-2,2]$",
        COLORS["midnight_blue"],
    ),
)


def _load(root: Path, run: str) -> dict[str, float | str]:
    test = json.loads((root / run / "test_metrics.json").read_text(encoding="utf-8"))
    diagnostics = test["probabilistic_diagnostics"]
    calibration = diagnostics["calibration"]
    observed = np.asarray(calibration["observed_coverages"], dtype=float)
    levels = np.asarray(calibration["confidence_levels"], dtype=float)
    coverage90 = float(observed[np.argmin(np.abs(levels - 0.9))])
    spectrum = diagnostics["spectrum"]
    return {
        "nll": float(test.get("nll", test["loss"])),
        "coverage90": coverage90,
        "whitened_defect": float(diagnostics["whitened_residual_covariance_trace"]),
        "condition_max": float(spectrum["condition_number_max"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup_tpami_style()
    values = [_load(args.results, run) for run, _, _ in RUNS]
    labels = [r"$[-4,4]$", r"$[-2,2]$", "centered"]
    colors = [color for _, _, color in RUNS]
    x = np.arange(len(RUNS))

    fig, axes = plt.subplots(1, 4, figsize=cm2inch(18.0, 5.8))
    panels = (
        ("nll", "Test NLL", "lower is better"),
        ("coverage90", "Coverage", "90% nominal"),
        ("whitened_defect", "Whitened trace", r"Student-$t$ target $45$"),
        ("condition_max", "Cond. number", r"certified bound $e^4$"),
    )
    for ax, (key, ylabel, subtitle) in zip(axes, panels):
        plotted = [float(record[key]) for record in values]
        ax.scatter(x, plotted, s=52, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
        ax.plot(x, plotted, color=COLORS["gray"], linewidth=1.0, zorder=1)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
        ax.set_title(subtitle, loc="left", fontsize=8)
        ax.tick_params(axis="both", labelsize=7.5)
        if key == "coverage90":
            ax.axhline(0.90, color=COLORS["dark_gray"], linestyle="--", linewidth=1.0)
            ax.set_ylim(0, 1.0)
        if key == "condition_max":
            ax.axhline(np.exp(4), color=COLORS["dark_gray"], linestyle="--", linewidth=1.0)
            ax.set_yscale("log")
    for index, ax in enumerate(axes):
        ax.set_title(f"({chr(97 + index)}) {panels[index][2]}", loc="left", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.32, top=0.82, wspace=0.55)
    save_figure(fig, args.output / "dielectric_window_audit", formats=("pdf", "png"))
    plt.close(fig)

    payload = {
        "scope": "downloaded single-seed server runs; descriptive, not factorial",
        "runs": {run: record for (run, _, _), record in zip(RUNS, values)},
    }
    (args.output / "dielectric_window_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
