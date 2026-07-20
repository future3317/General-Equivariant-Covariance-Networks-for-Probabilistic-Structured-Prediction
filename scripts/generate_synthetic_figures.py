"""Generate TPAMI figures for the synthetic covariance-recovery benchmark.

This script loads metrics produced by ``experiments/synthetic_covariance_recovery.py``
(or runs the experiment if a configuration is missing) and creates publication-ready
figures using the unified ``plotting`` style.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plotting import (
    COLORS,
    PALETTE,
    cm2inch,
    get_color,
    label_panels,
    save_figure,
    setup_tpami_style,
)


# Configurations used in the main synthetic benchmark figure.
SYNTHETIC_CONFIGS = [
    {"output_irreps": "1o", "name": r"$V = \ell=1$ (vector)"},
    {"output_irreps": "0e + 2e", "name": r"$V = \ell=0 \oplus \ell=2$"},
    {"output_irreps": "0e + 1o + 2e", "name": r"$V = \ell=0 \oplus \ell=1 \oplus \ell=2$"},
]


def run_experiment(output_irreps: str, save_dir: Path) -> dict:
    """Run the synthetic experiment for one output representation."""
    cmd = [
        sys.executable,
        "-m",
        "experiments.synthetic_covariance_recovery",
        "--output_irreps",
        output_irreps,
        "--save_dir",
        str(save_dir),
        "--num_epochs",
        "200",
        "--num_train",
        "2000",
        "--num_test",
        "500",
    ]
    print(f"Running synthetic experiment for {output_irreps} ...")
    subprocess.run(cmd, check=True)
    with open(save_dir / "metrics.json") as f:
        return json.load(f)


def load_or_run(config: dict, results_root: Path) -> dict:
    """Load metrics for a config, running the experiment if missing."""
    safe_name = config["output_irreps"].replace(" ", "_").replace("+", "plus")
    save_dir = results_root / safe_name
    metrics_path = save_dir / "metrics.json"

    if not metrics_path.exists():
        run_experiment(config["output_irreps"], save_dir)

    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics["_name"] = config["name"]
    metrics["_output_irreps"] = config["output_irreps"]
    return metrics


def plot_metric_comparison(metrics_list: list[dict], save_path: str | Path) -> None:
    """Bar chart comparing final metrics across output representations."""
    setup_tpami_style()

    names = [m["_name"] for m in metrics_list]
    x = np.arange(len(names))
    width = 0.22

    rel_err = [m["cov_rel_error"] for m in metrics_list]
    log_err = [m["log_euclidean_error"] for m in metrics_list]
    eig_err = [m["eigenvalue_error"] for m in metrics_list]
    mu_err = [m["mu_mae"] for m in metrics_list]

    fig, ax = plt.subplots(figsize=cm2inch(16, 9))

    ax.bar(x - 1.5 * width, rel_err, width, label="Cov. rel. error", color=PALETTE[0])
    ax.bar(x - 0.5 * width, log_err, width, label="Log-Euclidean error", color=PALETTE[1])
    ax.bar(x + 0.5 * width, eig_err, width, label="Eigenvalue error", color=PALETTE[2])
    ax.bar(x + 1.5 * width, mu_err, width, label="Mean MAE", color=PALETTE[3])

    ax.set_ylabel("Error")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Synthetic Covariance Recovery Across Output Representations")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def plot_coverage_calibration(metrics_list: list[dict], save_path: str | Path) -> None:
    """Coverage comparison and whitened residual covariance across configs."""
    setup_tpami_style()

    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16, 6))
    ax_cov, ax_white = axes

    names = [m["_name"] for m in metrics_list]
    x = np.arange(len(names))
    width = 0.25

    # Left: empirical coverage at different levels.
    levels = ["coverage_50", "coverage_90", "coverage_95"]
    targets = [0.50, 0.90, 0.95]
    labels = ["50%", "90%", "95%"]
    for i, (key, target, label) in enumerate(zip(levels, targets, labels)):
        vals = [m[key] for m in metrics_list]
        ax_cov.bar(x + (i - 1) * width, vals, width, label=label, color=PALETTE[i])
    ax_cov.axhline(0.90, color=COLORS["dark_gray"], linestyle="--", linewidth=1.0)
    ax_cov.axhline(0.95, color=COLORS["dark_gray"], linestyle="--", linewidth=1.0)
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(names)
    ax_cov.set_title("Confidence Ellipsoid Coverage")
    ax_cov.legend(title="Target", loc="lower right")
    ax_cov.set_ylim(0.0, 1.05)

    # Right: whitened residual covariance trace.
    white_trace = [m["whitened_cov_trace"] for m in metrics_list]
    bars = ax_white.bar(x, white_trace, color=PALETTE[0])
    ax_white.axhline(1.0, color=COLORS["dark_gray"], linestyle="--", linewidth=1.0, label="Ideal")
    ax_white.set_ylabel(r"$\mathrm{Tr}\,\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top]$")
    ax_white.set_xticks(x)
    ax_white.set_xticklabels(names)
    ax_white.set_title("Whitened Residual Covariance Trace")
    ax_white.legend()
    for bar, val in zip(bars, white_trace):
        height = bar.get_height()
        ax_white.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    label_panels(axes, x=-0.16, y=1.04)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default="results/synthetic_covariance_recovery",
        help="Root directory containing per-configuration metrics.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures/synthetic",
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--run_missing",
        action="store_true",
        help="Run the synthetic experiment for missing configurations.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    metrics_list = []
    for config in SYNTHETIC_CONFIGS:
        safe_name = config["output_irreps"].replace(" ", "_").replace("+", "plus")
        metrics_path = results_root / safe_name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        elif args.run_missing:
            metrics = load_or_run(config, results_root)
        else:
            print(f"Skipping {config['output_irreps']}: metrics not found at {metrics_path}")
            continue
        metrics["_name"] = config["name"]
        metrics["_output_irreps"] = config["output_irreps"]
        metrics_list.append(metrics)

    if not metrics_list:
        print("No metrics found. Run with --run_missing to generate them.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_comparison(metrics_list, output_dir / "synthetic_metric_comparison")
    plot_coverage_calibration(metrics_list, output_dir / "synthetic_coverage_calibration")

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
