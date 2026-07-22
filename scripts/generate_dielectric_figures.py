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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dielectric_dataset import get_dielectric_irreps_loaders
from data.tensor_conversions import irreps_to_km
from equivcompiler import FeatureSpec, FullCovariance, SpectralWindowCovariance, plan_readout
from evaluation.calibration import calibration_error, qq_data
from evaluation.metrics import empirical_coverage, mahalanobis_distance_squared
from models import EquivariantBackbone
from plotting import (
    COLORS,
    PALETTE,
    cm2inch,
    label_panels,
    save_figure,
    setup_tpami_style,
)
from scripts._common import tensor_product_kwargs


def load_model(checkpoint_dir: Path, device: str):
    """Reconstruct and load the dielectric model from a checkpoint directory."""
    with open(checkpoint_dir / "args.json") as f:
        args = argparse.Namespace(**json.load(f))

    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
        atom_features=args.atom_features,
        **tensor_product_kwargs(args),
    )
    parameterization = args.covariance_parameterization
    covariance = (
        FullCovariance()
        if parameterization == "matrix_exp"
        else SpectralWindowCovariance(
            args.log_variance_min,
            args.log_variance_max,
        )
    )
    model = plan_readout(
        FeatureSpec.from_backbone(backbone),
        output="0e + 2e",
        covariance=covariance,
        distribution="gaussian",
        output_scope="global",
    ).bind(backbone).to(device)

    state = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, args


@torch.inference_mode()
def collect_predictions(model, dataloader, device):
    """Collect test predictions in likelihood coordinates.

    The learned generator remains FP32.  For calibration and declared spectral
    bounds, we materialize the identical spectral map in FP64 to avoid a
    finite-precision eigendecomposition artifact at the smallest variance.
    """
    all_mu = []
    all_scale = []
    all_y_irreps = []
    all_y_km = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue
        result = model(batch)
        if model.spd_map is None:
            raise TypeError("dielectric figures require a probabilistic SPD map")
        all_mu.append(result["mu"].double().cpu())
        all_scale.append(model.spd_map(result["params"].double()).cpu())
        all_y_irreps.append(batch.y_irreps.double().cpu())
        all_y_km.append(batch.y_km.double().cpu())

    return {
        "mu_irreps": torch.cat(all_mu, dim=0),
        "scale_irreps": torch.cat(all_scale, dim=0),
        "y_irreps": torch.cat(all_y_irreps, dim=0),
        "y_km": torch.cat(all_y_km, dim=0),
    }


def plot_training_curves(history: list[dict], save_path: Path) -> None:
    """Plot train/val loss and validation MAEs over epochs."""
    setup_tpami_style()

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["loss"] for h in history]
    val_phys_mae = [h["phys_mae"] for h in history]
    val_log_mae = [h["log_mae"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.5, 6.8))
    ax_loss, ax_mae = axes

    ax_loss.plot(epochs, train_loss, label="Train loss", color=PALETTE[0])
    ax_loss.plot(epochs, val_loss, label="Val loss", color=PALETTE[1])
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("Loss", fontsize=9)
    ax_loss.set_title("Training and Validation Loss", fontsize=10)
    ax_loss.legend(fontsize=7)

    ax_mae.plot(epochs, val_phys_mae, label="Physical MAE", color=PALETTE[2])
    ax_mae.plot(epochs, val_log_mae, label="Log-KM MAE", color=PALETTE[3])
    ax_mae.set_xlabel("Epoch", fontsize=9)
    ax_mae.set_ylabel("MAE", fontsize=9)
    ax_mae.set_title("Validation MAE", fontsize=10)
    ax_mae.legend(fontsize=7)

    for ax in axes:
        ax.tick_params(labelsize=8)
    label_panels(axes, x=-0.10, y=1.02, fontsize=9)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def plot_parity(pred_km: np.ndarray, target_km: np.ndarray, save_path: Path) -> None:
    """Parity plot per Kelvin-Mandel component."""
    setup_tpami_style()

    d = pred_km.shape[-1]
    n_cols = 3
    n_rows = int(np.ceil(d / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=cm2inch(16.5, 10.0))
    axes = np.atleast_1d(axes).flatten()

    lims = [(target_km[:, i].min(), target_km[:, i].max()) for i in range(d)]

    for i in range(d):
        ax = axes[i]
        ax.scatter(target_km[:, i], pred_km[:, i], s=6, alpha=0.35, color=PALETTE[0])
        lo, hi = lims[i]
        ax.plot([lo, hi], [lo, hi], "--", color=COLORS["dark_gray"], linewidth=1.2)
        if i >= n_cols:
            ax.set_xlabel("Target", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("Prediction", fontsize=8)
        r2 = 1 - np.sum((target_km[:, i] - pred_km[:, i]) ** 2) / (
            np.sum((target_km[:, i] - target_km[:, i].mean()) ** 2) + 1e-12
        )
        ax.set_title(f"KM component {i + 1}: $R^2={r2:.3f}$", fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(d, len(axes)):
        axes[j].axis("off")

    label_panels(axes[:d], x=-0.08, y=1.01, fontsize=9)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def plot_calibration(
    mu: torch.Tensor, y: torch.Tensor, scale: torch.Tensor, save_path: Path
) -> None:
    """Coverage calibration and Q-Q plot for Mahalanobis distances."""
    setup_tpami_style()

    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16.5, 6.8))
    ax_cov, ax_qq = axes

    # Left: confidence level vs empirical coverage.
    levels = np.linspace(0.1, 0.95, 10)
    coverages = empirical_coverage(mu, y, scale, levels=levels.tolist())
    observed = [coverages[f"coverage_{int(level * 100):02d}"] for level in levels]

    ax_cov.plot(
        levels,
        levels,
        "--",
        color=COLORS["dark_gray"],
        linewidth=1.2,
        label="Perfect calibration",
    )
    ax_cov.plot(
        levels,
        observed,
        "o-",
        color=PALETTE[0],
        linewidth=2.0,
        markersize=5,
        label="Model",
    )
    ax_cov.fill_between(levels, levels, observed, alpha=0.15, color=PALETTE[0])
    ax_cov.set_xlabel("Confidence level", fontsize=9)
    ax_cov.set_ylabel("Empirical coverage", fontsize=9)
    ax_cov.set_title("Confidence Ellipsoid Calibration", fontsize=10)
    ax_cov.legend(loc="lower right", fontsize=7)
    ax_cov.set_xlim(0.0, 1.0)
    ax_cov.set_ylim(0.0, 1.0)

    # Right: Q-Q plot.
    theoretical, empirical = qq_data(mu, y, scale, num_quantiles=100)
    ax_qq.plot(
        theoretical,
        empirical,
        "o",
        color=PALETTE[0],
        markersize=4,
        alpha=0.7,
        label="Empirical",
    )
    max_val = max(theoretical.max(), empirical.max())
    ax_qq.plot(
        [0, max_val],
        [0, max_val],
        "--",
        color=COLORS["dark_gray"],
        linewidth=1.2,
        label="Reference",
    )
    ax_qq.set_xlabel(r"Theoretical $\chi^2$ quantile", fontsize=9)
    ax_qq.set_ylabel(r"Empirical Mahalanobis$^2$ quantile", fontsize=9)
    ax_qq.set_title("Q-Q Calibration", fontsize=10)
    ax_qq.legend(fontsize=7)

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
    for (label, risks), color in zip(risks_by_score.items(), PALETTE):
        ax.plot(fractions * 100, risks, "-", color=color, linewidth=2.3, label=label)
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
        color=PALETTE[0],
        edgecolor="white",
        alpha=0.85,
    )
    if log_variance_bounds is not None:
        lower, upper = log_variance_bounds
        ax_spectrum.axvline(lower, color=COLORS["tertiary"], linestyle="--", label="Window")
        ax_spectrum.axvline(upper, color=COLORS["tertiary"], linestyle="--")
        ax_spectrum.set_xlim(lower - 0.35, upper + 0.35)
    ax_spectrum.set_xlabel(r"$\log$ covariance eigenvalue", fontsize=9)
    ax_spectrum.set_ylabel("Density", fontsize=9)
    ax_spectrum.set_title("Spectral-Window Utilization", fontsize=10)
    ax_spectrum.legend(loc="upper center", fontsize=7)

    sorted_condition = np.sort(condition_numbers)
    quantiles = np.linspace(0.0, 1.0, len(sorted_condition), endpoint=True)
    ax_condition.plot(sorted_condition, quantiles, color=PALETTE[1], linewidth=2.3)
    if log_variance_bounds is not None:
        upper_condition = np.exp(log_variance_bounds[1] - log_variance_bounds[0])
        ax_condition.axvline(
            upper_condition,
            color=COLORS["tertiary"],
            linestyle="--",
            linewidth=1.2,
            label="Theoretical maximum",
        )
    ax_condition.set_xscale("log")
    ax_condition.set_xlabel("Condition number", fontsize=9)
    ax_condition.set_ylabel("Empirical CDF", fontsize=9)
    ax_condition.set_title("Conditioning of Predicted Covariances", fontsize=10)
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

    model, train_args = load_model(checkpoint_dir, args.device)

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

    preds = collect_predictions(model, test_loader, args.device)
    pred_km = irreps_to_km(preds["mu_irreps"]).numpy()
    target_km = preds["y_km"].numpy()

    with open(checkpoint_dir / "history.json") as f:
        history = json.load(f)

    plot_training_curves(history, output_dir / "dielectric_training_curves")
    plot_parity(pred_km, target_km, output_dir / "dielectric_parity")
    plot_calibration(
        preds["mu_irreps"],
        preds["y_irreps"],
        preds["scale_irreps"],
        output_dir / "dielectric_calibration",
    )
    risk_coverage = plot_risk_coverage(
        preds["mu_irreps"],
        preds["y_irreps"],
        preds["scale_irreps"],
        output_dir / "dielectric_risk_coverage",
    )
    bounds = (
        (train_args.log_variance_min, train_args.log_variance_max)
        if train_args.covariance_parameterization == "spectral_window"
        else None
    )
    spectrum = plot_spectral_diagnostics(
        preds["scale_irreps"], bounds, output_dir / "dielectric_spectrum"
    )

    # Print test calibration metrics.
    cal_err = calibration_error(
        preds["mu_irreps"], preds["y_irreps"], preds["scale_irreps"]
    )
    coverage = empirical_coverage(
        preds["mu_irreps"], preds["y_irreps"], preds["scale_irreps"]
    )
    mahalanobis2 = mahalanobis_distance_squared(
        preds["y_irreps"] - preds["mu_irreps"], preds["scale_irreps"]
    )
    with open(output_dir / "figure_metrics.json", "w") as f:
        json.dump(
            {
                "coordinate_space": "log_kelvin_mandel",
                "scale_materialization_dtype": "float64",
                "calibration": cal_err,
                "coverage": coverage,
                "mahalanobis2_mean": float(mahalanobis2.mean().item()),
                "risk_coverage": risk_coverage,
                "spectrum": spectrum,
            },
            f,
            indent=2,
        )
    print(f"ECE: {cal_err['ece']:.4f}, ACE: {cal_err['ace']:.4f}")
    print(f"Coverage: {coverage}")
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
