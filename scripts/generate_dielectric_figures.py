"""Generate TPAMI figures for the dielectric tensor benchmark.

Loads a trained checkpoint from ``scripts/train_dielectric.py`` and produces
publication-ready diagnostic figures using the unified ``plotting`` style.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from scipy.stats import chi2
from scipy.stats import f as f_dist
from scipy.stats import t as student_t_dist

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dielectric_dataset import get_dielectric_irreps_loaders
from data.representation_metrics import transformed_spectral_bounds
from data.tensor_conversions import irreps_to_km
from evaluation.calibration import calibration_error, qq_data
from evaluation.metrics import empirical_coverage, mahalanobis_distance_squared
from plotting import (
    COLORS,
    DENSITY_CMAP,
    DIVERGING_CMAP,
    cm2inch,
    label_panels,
    save_figure,
    setup_tpami_style,
)
from scripts.dielectric_runtime import (
    collect_dielectric_predictions,
    configure_inference_contract,
    inference_contract_from_args,
    inference_contract_hash,
    load_dielectric_checkpoint,
    load_dielectric_data_args,
    load_run_record,
)


def plot_training_curves(history: list[dict], save_path: Path) -> None:
    """Plot train/val loss and validation MAEs over epochs.

    Stage markers are read from the history when a staged run records them.
    A single-stage history is labeled as such rather than assigning
    unrecorded mean/scatter/joint boundaries.
    """
    setup_tpami_style()

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["loss"] for h in history]
    val_phys_mae = [h["phys_mae"] for h in history]
    val_log_mae = [h["log_mae"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.5, 6.8))
    ax_loss, ax_mae = axes

    ax_loss.plot(epochs, train_loss, label="Train loss", color=COLORS["midnight_blue"])
    ax_loss.plot(
        epochs,
        val_loss,
        label="Validation loss",
        color=COLORS["champagne_gold"],
        linestyle="--",
    )
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("Loss", fontsize=9)
    ax_loss.set_title("Training and Validation Loss", fontsize=10)
    ax_loss.legend(fontsize=7)

    physical_line = ax_mae.plot(
        epochs,
        val_phys_mae,
        label="Physical MAE",
        color=COLORS["midnight_blue"],
    )
    ax_mae.set_ylabel("Physical MAE", fontsize=9, color=COLORS["midnight_blue"])
    ax_mae.tick_params(axis="y", labelcolor=COLORS["midnight_blue"])
    ax_log = ax_mae.twinx()
    log_line = ax_log.plot(
        epochs,
        val_log_mae,
        label="Log-KM MAE",
        color=COLORS["champagne_gold"],
        linestyle="--",
    )
    ax_mae.set_xlabel("Epoch", fontsize=9)
    ax_mae.set_title("Validation MAE", fontsize=10)
    ax_log.set_ylabel("Log-KM MAE", fontsize=9, color=COLORS["champagne_gold"])
    ax_log.tick_params(axis="y", labelcolor=COLORS["champagne_gold"])
    ax_mae.legend(physical_line + log_line, ["Physical MAE", "Log-KM MAE"], fontsize=7)

    stage_names = [
        item.get("training_stage", item.get("stage"))
        for item in history
    ]
    if any(name is not None for name in stage_names):
        previous = stage_names[0]
        start_epoch = epochs[0]
        for epoch, name in zip(epochs[1:], stage_names[1:]):
            if name != previous:
                ax_loss.axvline(epoch - 0.5, color=COLORS["gray"], linestyle=":")
                ax_mae.axvline(epoch - 0.5, color=COLORS["gray"], linestyle=":")
                ax_mae.text(
                    start_epoch,
                    1.03,
                    str(previous),
                    transform=ax_mae.get_xaxis_transform(),
                    fontsize=7,
                    color=COLORS["dark_gray"],
                )
                start_epoch = epoch
                previous = name
        ax_mae.text(
            start_epoch,
            1.03,
            str(previous),
            transform=ax_mae.get_xaxis_transform(),
            fontsize=7,
            color=COLORS["dark_gray"],
        )
    else:
        ax_mae.text(
            0.02,
            0.97,
            "single recorded run",
            transform=ax_mae.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color=COLORS["dark_gray"],
        )

    for ax in (ax_loss, ax_mae, ax_log):
        ax.tick_params(labelsize=8)
    label_panels(axes, x=-0.10, y=1.02, fontsize=9)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def plot_parity(pred_km: np.ndarray, target_km: np.ndarray, save_path: Path) -> None:
    """Density-aware prediction--target plots in log-KM coordinates."""
    setup_tpami_style()

    d = pred_km.shape[-1]
    n_cols = 3
    n_rows = int(np.ceil(d / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=cm2inch(16.5, 10.0))
    axes = np.atleast_1d(axes).flatten()

    first_hexbin = None
    for i in range(d):
        ax = axes[i]
        values = np.concatenate((target_km[:, i], pred_km[:, i]))
        values = values[np.isfinite(values)]
        lo, hi = values.min(), values.max()
        pad = max(1e-6, 0.02 * (hi - lo))
        lo, hi = lo - pad, hi + pad
        hb = ax.hexbin(
            target_km[:, i],
            pred_km[:, i],
            gridsize=38,
            mincnt=1,
            bins="log",
            cmap=DENSITY_CMAP,
            linewidths=0,
        )
        if first_hexbin is None:
            first_hexbin = hb
        ax.plot(
            [lo, hi],
            [lo, hi],
            "--",
            color=COLORS["champagne_gold"],
            linewidth=1.25,
            label="Identity" if i == 0 else None,
        )
        residual = pred_km[:, i] - target_km[:, i]
        r2 = 1 - np.sum(residual**2) / (
            np.sum((target_km[:, i] - target_km[:, i].mean()) ** 2) + 1e-12
        )
        mae = np.mean(np.abs(residual))
        ax.text(
            0.04,
            0.96,
            f"MAE {mae:.3f}\n$R^2$ {r2:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            color=COLORS["dark_gray"],
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Target log-KM", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("Prediction log-KM", fontsize=8)
        ax.set_title(f"Component {i + 1}", fontsize=9)
        ax.tick_params(labelsize=7)

    if first_hexbin is not None:
        cbar_ax = fig.add_axes([0.945, 0.16, 0.012, 0.68])
        cbar = fig.colorbar(
            first_hexbin,
            cax=cbar_ax,
        )
        cbar.set_label("log$_{10}$(count)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    for j in range(d, len(axes)):
        axes[j].axis("off")

    label_panels(axes[:d], x=-0.08, y=1.01, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.08, top=0.94, wspace=0.34, hspace=0.36)
    save_figure(fig, save_path)
    plt.close(fig)


def plot_uncertainty_alignment(
    mu_irreps: torch.Tensor,
    y_irreps: torch.Tensor,
    scale_irreps: torch.Tensor,
    save_path: Path,
    *,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
) -> dict[str, object]:
    """Diagnose whether predicted uncertainty matches residual structure.

    The first two panels compare residual and predicted correlation structure;
    the third reports marginal coverage.  The covariance basis conversion uses
    the row-vector convention of :func:`irreps_to_km`.
    """
    setup_tpami_style()
    mu = irreps_to_km(mu_irreps).double()
    target = irreps_to_km(y_irreps).double()
    residual = mu - target
    basis = irreps_to_km(torch.eye(6, dtype=torch.float64))
    scale = torch.einsum("ab,nbc,cd->nad", basis.T, scale_irreps.double(), basis)
    residual_cov = torch.cov(residual.T)
    if distribution == "student_t":
        if student_t_dof <= 2:
            raise ValueError("student_t_dof must exceed 2 for covariance diagnostics")
        predicted_cov = scale.mean(dim=0) * (student_t_dof / (student_t_dof - 2.0))
    elif distribution == "gaussian":
        predicted_cov = scale.mean(dim=0)
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    residual_corr = residual_cov / torch.sqrt(
        torch.outer(torch.diag(residual_cov), torch.diag(residual_cov))
    )
    predicted_corr = predicted_cov / torch.sqrt(
        torch.outer(torch.diag(predicted_cov), torch.diag(predicted_cov))
    )

    normal = torch.distributions.Normal(0.0, 1.0)
    marginal_coverage: dict[str, list[float]] = {}
    for level in (0.5, 0.9):
        z = (normal.icdf(torch.tensor((1.0 + level) / 2.0))
             if distribution == "gaussian" else torch.tensor(
                 float(student_t_dist.ppf((1.0 + level) / 2.0, df=student_t_dof)),
                 dtype=scale.dtype,
             ))
        marginal_coverage[f"coverage_{int(level * 100):02d}"] = (
            (residual.abs() <= z * torch.sqrt(torch.diagonal(scale, dim1=-2, dim2=-1)))
            .double()
            .mean(dim=0)
            .tolist()
        )

    labels = [r"$c_{11}$", r"$c_{22}$", r"$c_{33}$", r"$c_{23}$", r"$c_{13}$", r"$c_{12}$"]
    distribution_label = r"Student-$t$" if distribution == "student_t" else "Gaussian"
    correlation_difference = predicted_corr - residual_corr
    fig = plt.figure(figsize=cm2inch(18.2, 11.6))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 0.05, 0.60),
        hspace=0.84,
        wspace=0.22,
    )
    heat_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    correlation_bar_axis = fig.add_subplot(grid[1, :2])
    difference_bar_axis = fig.add_subplot(grid[1, 2])
    coverage_axis = fig.add_subplot(grid[2, :])
    norm = Normalize(vmin=-1.0, vmax=1.0)
    correlation_image = None
    for ax, matrix, title in (
        (heat_axes[0], residual_corr.numpy(), "(a) Empirical residual"),
        (heat_axes[1], predicted_corr.numpy(), "(b) Mean predicted"),
    ):
        correlation_image = ax.imshow(matrix, cmap=DIVERGING_CMAP, norm=norm)
        ax.set_xticks(range(6), labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(6), labels, fontsize=7)
        ax.set_title(
            title,
            loc="left",
            fontsize=9,
            fontweight="bold",
            pad=4,
        )
        ax.axvline(2.5, color="white", linewidth=1.0)
        ax.axhline(2.5, color="white", linewidth=1.0)
        ax.grid(False)
    difference_limit = max(
        0.1,
        float(torch.max(torch.abs(correlation_difference)).item()),
    )
    difference_image = heat_axes[2].imshow(
        correlation_difference.numpy(),
        cmap=DIVERGING_CMAP,
        norm=Normalize(vmin=-difference_limit, vmax=difference_limit),
    )
    heat_axes[2].set_xticks(range(6), labels, rotation=45, ha="right", fontsize=7)
    heat_axes[2].set_yticks(range(6), labels, fontsize=7)
    heat_axes[2].set_title(
        "(c) Predicted - empirical",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=4,
    )
    heat_axes[2].axvline(2.5, color="white", linewidth=1.0)
    heat_axes[2].axhline(2.5, color="white", linewidth=1.0)
    heat_axes[2].grid(False)
    correlation_bar = fig.colorbar(
        correlation_image,
        cax=correlation_bar_axis,
        orientation="horizontal",
    )
    correlation_bar.set_label("Correlation", fontsize=8)
    correlation_bar.ax.tick_params(labelsize=7)
    difference_bar = fig.colorbar(
        difference_image,
        cax=difference_bar_axis,
        orientation="horizontal",
    )
    difference_bar.set_label("Correlation defect", fontsize=8)
    difference_bar.ax.tick_params(labelsize=7)

    y = np.arange(6)
    coverage_axis.axvline(
        0.5,
        color=COLORS["midnight_blue"],
        linestyle=":",
        linewidth=1.1,
    )
    coverage_axis.axvline(
        0.9,
        color=COLORS["champagne_gold"],
        linestyle=":",
        linewidth=1.1,
    )
    coverage_axis.scatter(
        marginal_coverage["coverage_50"],
        y - 0.10,
        marker="o",
        s=32,
        color=COLORS["midnight_blue"],
        label="50% marginal interval",
        zorder=3,
    )
    coverage_axis.scatter(
        marginal_coverage["coverage_90"],
        y + 0.10,
        marker="s",
        s=32,
        color=COLORS["champagne_gold"],
        label="90% marginal interval",
        zorder=3,
    )
    coverage_axis.set_yticks(y, labels)
    coverage_axis.set_xlim(0.25, 1.01)
    coverage_axis.set_xlabel("Empirical marginal coverage")
    coverage_axis.set_title(
        f"(d) Component calibration ({distribution_label})",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    coverage_axis.legend(
        fontsize=7,
        loc="center",
        bbox_to_anchor=(0.58, 0.18),
        ncol=2,
    )
    coverage_axis.grid(axis="x", alpha=0.25)
    coverage_axis.tick_params(axis="y", length=0)
    for ax in (*heat_axes, coverage_axis):
        ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.96)
    save_figure(fig, save_path)
    plt.close(fig)

    residual_std = torch.sqrt(torch.diag(residual_cov))
    predicted_std = torch.sqrt(torch.diag(predicted_cov))
    return {
        "residual_std": residual_std.tolist(),
        "predicted_std": predicted_std.tolist(),
        "predicted_to_residual_std_ratio": (predicted_std / (residual_std + 1e-12)).tolist(),
        "marginal_coverage": marginal_coverage,
        "residual_correlation": residual_corr.tolist(),
        "predicted_correlation": predicted_corr.tolist(),
        "predicted_minus_residual_correlation": correlation_difference.tolist(),
    }


def plot_calibration(
    mu: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    save_path: Path,
    *,
    distribution: str = "gaussian",
    student_t_dof: float = 5.0,
) -> None:
    """Coverage calibration and Q-Q plot for Mahalanobis distances."""
    setup_tpami_style()
    distribution_label = r"Student-$t$" if distribution == "student_t" else "Gaussian"

    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.5, 6.8))
    ax_cov, ax_qq = axes

    # Left: confidence level vs empirical coverage.
    levels = np.linspace(0.1, 0.95, 10)
    coverages = empirical_coverage(
        mu, y, scale, levels=levels.tolist(), reference=distribution,
        student_t_dof=student_t_dof,
    )
    observed = np.asarray(
        [coverages[f"coverage_{int(level * 100):02d}"] for level in levels], dtype=float
    )
    # Bootstrap the complete coverage curve at the sample level.  This keeps
    # the dependence between confidence levels visible and avoids presenting
    # ten unrelated binomial intervals as a calibration band.
    maha2 = mahalanobis_distance_squared(y - mu, scale).detach().cpu().numpy()
    thresholds = (
        chi2.ppf(levels, df=float(mu.shape[-1]))
        if distribution == "gaussian"
        else float(mu.shape[-1]) * f_dist.ppf(
            levels, dfn=float(mu.shape[-1]), dfd=float(student_t_dof)
        )
    )
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(maha2), size=(500, len(maha2)))
    bootstrap_coverages = (maha2[indices, None] < thresholds[None, None, :]).mean(axis=1)
    lower_ci, upper_ci = np.percentile(bootstrap_coverages, (2.5, 97.5), axis=0)

    ax_cov.plot(
        levels,
        levels,
        "--",
        color=COLORS["champagne_gold"],
        linewidth=1.2,
        label="Perfect calibration",
    )
    ax_cov.fill_between(
        levels,
        lower_ci,
        upper_ci,
        color=COLORS["champagne_light"],
        alpha=0.38,
        label="95% bootstrap band",
    )
    ax_cov.plot(
        levels,
        observed,
        "o-",
        color=COLORS["midnight_blue"],
        linewidth=2.0,
        markersize=5,
        label="Model",
    )
    ax_cov.set_xlabel("Confidence level", fontsize=9)
    ax_cov.set_ylabel("Empirical coverage", fontsize=9)
    ax_cov.set_title("Log-KM Ellipsoid Calibration", fontsize=10)
    ax_cov.legend(loc="lower right", fontsize=7)
    ax_cov.set_xlim(0.0, 1.0)
    ax_cov.set_ylim(0.0, 1.0)
    ax_cov.set_aspect("equal", adjustable="box")
    ax_cov.set_box_aspect(1)

    # Right: Q-Q plot.
    theoretical, empirical = qq_data(
        mu, y, scale, num_quantiles=100, reference=distribution,
        student_t_dof=student_t_dof,
    )
    theoretical = np.maximum(np.asarray(theoretical, dtype=float), 1e-8)
    empirical = np.maximum(np.asarray(empirical, dtype=float), 1e-8)
    ax_qq.plot(
        theoretical,
        empirical,
        "o",
        color=COLORS["midnight_blue"],
        markersize=4,
        alpha=0.7,
        label="Empirical",
    )
    min_val = min(theoretical.min(), empirical.min())
    max_val = max(theoretical.max(), empirical.max())
    ax_qq.plot(
        [min_val, max_val],
        [min_val, max_val],
        "--",
        color=COLORS["champagne_gold"],
        linewidth=1.2,
        label="Reference",
    )
    qq_label = (r"Theoretical $\chi^2$ quantile" if distribution == "gaussian"
                else rf"Theoretical $dF_{{d,\nu}}$ quantile ($\nu={student_t_dof:g}$)")
    ax_qq.set_xlabel(qq_label, fontsize=9)
    ax_qq.set_ylabel(r"Empirical Mahalanobis$^2$ quantile", fontsize=9)
    ax_qq.set_title(f"Q-Q Calibration ({distribution_label})", fontsize=10)
    ax_qq.legend(fontsize=7)
    ax_qq.set_xscale("log")
    ax_qq.set_yscale("log")

    for ax in axes:
        ax.tick_params(labelsize=8)
    label_panels(axes, x=-0.10, y=1.02, fontsize=9)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def plot_risk_coverage(
    mu: torch.Tensor, y: torch.Tensor, scale: torch.Tensor, save_path: Path
) -> dict[str, float]:
    """Compare two equivariant scalar uncertainty rankings in log-KM space."""
    setup_tpami_style()

    residual = torch.abs(mu - y)
    mae_per_sample = residual.mean(dim=-1)

    fractions = np.linspace(0.1, 1.0, 91)
    uncertainty_scores = {
        r"Trace$(S)$": torch.diagonal(scale, dim1=-2, dim2=-1).sum(dim=-1),
        r"$\lambda_{\max}(S)$": torch.linalg.eigvalsh(scale)[..., -1],
    }
    risks_by_score: dict[str, np.ndarray] = {}
    for label, uncertainty in uncertainty_scores.items():
        sorted_mae = mae_per_sample[torch.argsort(uncertainty)].numpy()
        risks_by_score[label] = np.asarray(
            [sorted_mae[: max(1, int(f * len(sorted_mae)))].mean() for f in fractions]
        )

    fig, ax = plt.subplots(figsize=cm2inch(10, 7))
    for label, risks in risks_by_score.items():
        ax.plot(
            fractions * 100,
            risks,
            "-",
            color=(COLORS["midnight_blue"] if label.startswith("Trace") else COLORS["champagne_gold"]),
            linewidth=2.3,
            label=label,
        )
    ax.axhline(
        mae_per_sample.mean().item(),
        color=COLORS["dark_gray"],
        linestyle="--",
        linewidth=1.2,
        label="Full-set MAE",
    )
    ax.set_xlabel("Coverage (%)", fontsize=9)
    ax.set_ylabel("Log-KM MAE", fontsize=9)
    ax.set_title("Uncertainty-Risk Ranking", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    ax.set_xlim(10, 100)

    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    return {
        f"{label}_risk_at_90_percent": float(risks[np.searchsorted(fractions, 0.9)])
        for label, risks in risks_by_score.items()
    }


def plot_spectral_diagnostics(
    scale: torch.Tensor,
    log_variance_bounds: tuple[float, float] | None,
    save_path: Path,
    condition_log_bound: float | None = None,
) -> dict[str, float]:
    """Plot covariance-spectrum utilization and condition-number distribution."""
    setup_tpami_style()
    log_eigenvalues = torch.log(torch.linalg.eigvalsh(scale)).numpy().ravel()
    condition_numbers = np.exp(
        np.ptp(log_eigenvalues.reshape(-1, scale.shape[-1]), axis=1)
    )

    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.5, 6.8))
    ax_spectrum, ax_condition = axes
    ax_spectrum.hist(
        log_eigenvalues,
        bins=36,
        density=True,
        color=COLORS["midnight_blue"],
        edgecolor="white",
        alpha=0.85,
    )
    if log_variance_bounds is not None:
        lower, upper = log_variance_bounds
        ax_spectrum.axvline(
            lower,
            color=COLORS["champagne_gold"],
            linestyle="--",
            label=f"Spectral window [{lower:.2f}, {upper:.2f}]",
        )
        ax_spectrum.axvline(upper, color=COLORS["champagne_gold"], linestyle="--")
    spectrum_min = float(log_eigenvalues.min())
    spectrum_max = float(log_eigenvalues.max())
    spectrum_pad = max(0.05, 0.06 * (spectrum_max - spectrum_min))
    ax_spectrum.set_xlim(spectrum_min - spectrum_pad, spectrum_max + spectrum_pad)
    ax_spectrum.set_xlabel(r"$\log$ scatter eigenvalue", fontsize=9)
    ax_spectrum.set_ylabel("Density", fontsize=9)
    ax_spectrum.set_title("Spectral-Window Utilization", fontsize=10)
    ax_spectrum.legend(loc="upper left", fontsize=7)

    sorted_condition = np.sort(condition_numbers)
    quantiles = np.linspace(0.0, 1.0, len(sorted_condition), endpoint=True)
    ax_condition.plot(
        sorted_condition,
        quantiles,
        color=COLORS["midnight_blue"],
        linewidth=2.3,
        label="Empirical CDF",
    )
    if condition_log_bound is None and log_variance_bounds is not None:
        condition_log_bound = log_variance_bounds[1] - log_variance_bounds[0]
    if condition_log_bound is not None:
        upper_condition = np.exp(condition_log_bound)
        ax_condition.axvline(
            upper_condition,
            color=COLORS["champagne_gold"],
            linestyle="--",
            linewidth=1.2,
            label=rf"Certified bound $e^{{{condition_log_bound:g}}}={upper_condition:.1f}$",
        )
    ax_condition.set_xscale("log")
    ax_condition.xaxis.set_major_locator(LogLocator(base=10))
    ax_condition.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_condition.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax_condition.xaxis.set_minor_formatter(NullFormatter())
    ax_condition.set_xlabel("Condition number", fontsize=9)
    ax_condition.set_ylabel("Empirical CDF", fontsize=9)
    ax_condition.set_title("Conditioning of Predicted Scatters", fontsize=10)
    ax_condition.legend(loc="lower right", fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=8)
    label_panels(axes, x=-0.10, y=1.02, fontsize=9)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    return {
        "log_eigenvalue_min": float(log_eigenvalues.min()),
        "log_eigenvalue_max": float(log_eigenvalues.max()),
        "condition_number_mean": float(condition_numbers.mean()),
        "condition_number_max": float(condition_numbers.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints_dielectric",
        help="Directory with trained model.",
    )
    parser.add_argument(
        "--output_dir", default="figures/dielectric", help="Where figures are saved."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (checkpoint_dir / "args.json").exists():
        raise FileNotFoundError(
            f"args.json not found in {checkpoint_dir}. Run train_dielectric.py first."
        )
    if not (checkpoint_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"best_model.pt not found in {checkpoint_dir}.")

    model, _, _ = load_dielectric_checkpoint(checkpoint_dir, args.device)
    train_args = load_dielectric_data_args(checkpoint_dir)
    record = load_run_record(checkpoint_dir)
    inference_contract = record.get("inference_contract") or inference_contract_from_args(
        train_args, args.device
    )
    configure_inference_contract(inference_contract)

    _, _, test_loader = get_dielectric_irreps_loaders(
        data_dir=train_args.data_dir,
        batch_size=train_args.batch_size,
        num_workers=getattr(train_args, "num_workers", 0),
        persistent_workers=getattr(train_args, "persistent_workers", False),
        pin_memory=getattr(train_args, "pin_memory", False),
        prefetch_factor=getattr(train_args, "prefetch_factor", None),
        lmax=train_args.lmax,
        storage=getattr(train_args, "dataset_storage", "files"),
        shard_cache_size=getattr(train_args, "shard_cache_size", 2),
    )

    preds = collect_dielectric_predictions(
        model, test_loader, args.device, inference_contract=inference_contract
    )
    pred_km = irreps_to_km(preds["mu_irreps"]).numpy()
    target_km = preds["y_km"].numpy()

    with open(checkpoint_dir / "history.json") as f:
        history = json.load(f)

    plot_training_curves(history, output_dir / "dielectric_training_curves")
    plot_parity(pred_km, target_km, output_dir / "dielectric_parity")
    uncertainty_alignment = plot_uncertainty_alignment(
        preds["mu_irreps"],
        preds["y_irreps"],
        preds["scale_irreps"],
        output_dir / "dielectric_uncertainty_alignment",
        distribution=getattr(train_args, "distribution", "gaussian"),
        student_t_dof=getattr(train_args, "student_t_dof", 5.0),
    )
    plot_calibration(
        preds["mu_irreps"],
        preds["y_irreps"],
        preds["scale_irreps"],
        output_dir / "dielectric_calibration",
        distribution=getattr(train_args, "distribution", "gaussian"),
        student_t_dof=getattr(train_args, "student_t_dof", 5.0),
    )
    risk_coverage = plot_risk_coverage(
        preds["mu_irreps"],
        preds["y_irreps"],
        preds["scale_irreps"],
        output_dir / "dielectric_risk_coverage",
    )
    condition_log_bound = None
    if train_args.covariance_parameterization == "spectral_window":
        bounds = (train_args.log_variance_min, train_args.log_variance_max)
        condition_log_bound = train_args.log_variance_max - train_args.log_variance_min
    elif train_args.covariance_parameterization == "centered_spectral_window":
        # The centered shape has zero mean after the map; its coordinate-wise
        # deviation can therefore span [shape_min-shape_max,
        # shape_max-shape_min] before the bounded common mean log-scale is added.
        shape_radius = train_args.shape_max - train_args.shape_min
        bounds = (
            train_args.volume_min - shape_radius,
            train_args.volume_max + shape_radius,
        )
        # The common mean log-scale cancels in a condition number. Only the
        # centered trace-free shape range controls the certified ratio.
        condition_log_bound = train_args.shape_max - train_args.shape_min
    else:
        bounds = None
    if bounds is not None and getattr(train_args, "representation_metric", "none") == "block_auto":
        metric = torch.tensor(
            [float(train_args.metric_scalar)] + [float(train_args.metric_l2)] * 5,
            dtype=torch.float64,
        )
        bounds = transformed_spectral_bounds(bounds, metric)
    spectrum = plot_spectral_diagnostics(
        preds["scale_irreps"], bounds, output_dir / "dielectric_spectrum",
        condition_log_bound=condition_log_bound,
    )

    # Print test calibration metrics.
    cal_err = calibration_error(
        preds["mu_irreps"], preds["y_irreps"], preds["scale_irreps"],
        reference=getattr(train_args, "distribution", "gaussian"),
        student_t_dof=getattr(train_args, "student_t_dof", 5.0),
    )
    coverage = empirical_coverage(
        preds["mu_irreps"], preds["y_irreps"], preds["scale_irreps"],
        reference=getattr(train_args, "distribution", "gaussian"),
        student_t_dof=getattr(train_args, "student_t_dof", 5.0),
    )
    mahalanobis2 = mahalanobis_distance_squared(
        preds["y_irreps"] - preds["mu_irreps"], preds["scale_irreps"]
    )
    with open(output_dir / "figure_metrics.json", "w") as f:
        json.dump(
            {
                "coordinate_space": "log_kelvin_mandel",
                "scale_materialization_dtype": "float64",
                "distribution": getattr(train_args, "distribution", "gaussian"),
                "student_t_dof": getattr(train_args, "student_t_dof", 5.0),
                "inference_contract": inference_contract,
                "inference_contract_hash": inference_contract_hash(inference_contract),
                "calibration": cal_err,
                "coverage": coverage,
                "mahalanobis2_mean": float(mahalanobis2.mean().item()),
                "risk_coverage": risk_coverage,
                "spectrum": spectrum,
                "uncertainty_alignment": uncertainty_alignment,
            },
            f,
            indent=2,
        )
    print(f"ECE: {cal_err['ece']:.4f}, ACE: {cal_err['ace']:.4f}")
    print(f"Coverage: {coverage}")
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
