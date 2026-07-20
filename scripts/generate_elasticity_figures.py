"""Generate TPAMI figures for the elasticity tensor benchmark.

Loads a trained low-rank covariance checkpoint from ``scripts/train_elasticity.py``
and produces publication-ready diagnostic figures using the unified ``plotting`` style.
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

from data.elasticity_dataset import get_elasticity_irreps_loaders
from data.tensor_conversions import irreps_to_elasticity_21d
from distributions import GaussianNLL
from evaluation.calibration import calibration_error, qq_data
from evaluation.metrics import empirical_coverage
from models import (
    EquivariantBackbone,
    EquivariantMeanHead,
    O3EquivariantLowRankCovarianceHead,
    StructuredProbabilisticPredictor,
)
from plotting import (
    COLORS,
    PALETTE,
    cm2inch,
    label_panels,
    save_figure,
    setup_tpami_style,
)
from representations import O3IrrepsSpec, rank4_elasticity_irreps
from spd_maps import LowRankPlusIsotropicMap


def load_model(checkpoint_dir: Path, device: str):
    """Reconstruct and load the elasticity model from a checkpoint directory."""
    with open(checkpoint_dir / "args.json") as f:
        args = argparse.Namespace(**json.load(f))

    output_spec = O3IrrepsSpec(rank4_elasticity_irreps())
    backbone = EquivariantBackbone(
        hidden_dim=args.hidden_dim,
        lmax=args.lmax,
        num_layers=args.num_layers,
        atom_feature_dim=49,
        num_basis=args.num_basis,
    )
    mean_head = EquivariantMeanHead(backbone.irreps_out, output_spec.irreps, pool=True)
    cov_head = O3EquivariantLowRankCovarianceHead(
        backbone.irreps_out, output_spec, rank=args.rank, pool=True
    )

    model = StructuredProbabilisticPredictor(
        backbone=backbone,
        output_spec=output_spec,
        mean_head=mean_head,
        covariance_head=cov_head,
        spd_map=LowRankPlusIsotropicMap(dim=output_spec.dim, rank=args.rank),
        distribution=GaussianNLL(),
    ).to(device)

    state = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, args


@torch.inference_mode()
def collect_predictions(model, dataloader, device):
    """Collect mean, scale and target tensors over a dataloader."""
    all_mu_irreps = []
    all_scale = []
    all_y_irreps = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue
        result = model(batch, return_scale=True)
        all_mu_irreps.append(result["mu"].cpu())
        all_scale.append(result["scale"].cpu())
        all_y_irreps.append(batch.y_irreps.cpu())

    return {
        "mu_irreps": torch.cat(all_mu_irreps, dim=0),
        "scale": torch.cat(all_scale, dim=0),
        "y_irreps": torch.cat(all_y_irreps, dim=0),
    }


def plot_training_curves(history: list[dict], save_path: Path) -> None:
    """Plot train/val loss and validation MAE over epochs."""
    setup_tpami_style()

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["loss"] for h in history]
    val_mae = [h["mae"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16, 6))
    ax_loss, ax_mae = axes

    ax_loss.plot(epochs, train_loss, label="Train loss", color=PALETTE[0])
    ax_loss.plot(epochs, val_loss, label="Val loss", color=PALETTE[1])
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()

    ax_mae.plot(epochs, val_mae, color=PALETTE[2], linewidth=2.0)
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE (GPa)")
    ax_mae.set_title("Validation MAE")

    label_panels(axes, x=-0.18, y=1.04)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def plot_parity(pred_21d: np.ndarray, target_21d: np.ndarray, save_path: Path, n_show: int = 6) -> None:
    """Parity plot for selected independent elasticity components."""
    setup_tpami_style()

    # Show first n_show components (3 diagonal + 3 representative off-diagonal).
    components = list(range(n_show))
    n_cols = 3
    n_rows = int(np.ceil(len(components) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=cm2inch(14, 3.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, comp in enumerate(components):
        ax = axes[idx]
        ax.scatter(target_21d[:, comp], pred_21d[:, comp], s=10, alpha=0.4, color=PALETTE[0])
        lo, hi = target_21d[:, comp].min(), target_21d[:, comp].max()
        ax.plot([lo, hi], [lo, hi], "--", color=COLORS["dark_gray"], linewidth=1.2)
        ax.set_xlabel(f"Target $C_{{{comp + 1}}}$")
        ax.set_ylabel(f"Predicted $C_{{{comp + 1}}}$")
        r2 = 1 - np.sum((target_21d[:, comp] - pred_21d[:, comp]) ** 2) / (
            np.sum((target_21d[:, comp] - target_21d[:, comp].mean()) ** 2) + 1e-12
        )
        ax.set_title(f"$R^2$ = {r2:.3f}")

    for j in range(len(components), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Elasticity Tensor Parity Plot (selected components)", y=1.02)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def plot_calibration(mu: torch.Tensor, y: torch.Tensor, scale: torch.Tensor, save_path: Path) -> None:
    """Coverage calibration and Q-Q plot for Mahalanobis distances."""
    setup_tpami_style()

    fig, axes = plt.subplots(1, 2, figsize=cm2inch(16, 6))
    ax_cov, ax_qq = axes

    levels = np.linspace(0.1, 0.95, 10)
    coverages = empirical_coverage(mu, y, scale, levels=levels.tolist())
    observed = [coverages[f"coverage_{int(l * 100):02d}"] for l in levels]

    ax_cov.plot(levels, levels, "--", color=COLORS["dark_gray"], linewidth=1.2, label="Perfect calibration")
    ax_cov.plot(levels, observed, "o-", color=PALETTE[0], linewidth=2.0, markersize=5, label="Model")
    ax_cov.fill_between(levels, levels, observed, alpha=0.15, color=PALETTE[0])
    ax_cov.set_xlabel("Confidence level")
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.set_title("Confidence Ellipsoid Calibration")
    ax_cov.legend(loc="lower right")
    ax_cov.set_xlim(0.0, 1.0)
    ax_cov.set_ylim(0.0, 1.0)

    theoretical, empirical = qq_data(mu, y, scale, num_quantiles=100)
    ax_qq.plot(theoretical, empirical, "o", color=PALETTE[0], markersize=4, alpha=0.7, label="Empirical")
    max_val = max(theoretical.max(), empirical.max())
    ax_qq.plot([0, max_val], [0, max_val], "--", color=COLORS["dark_gray"], linewidth=1.2, label="Reference")
    ax_qq.set_xlabel(r"Theoretical $\chi^2$ quantile")
    ax_qq.set_ylabel(r"Empirical Mahalanobis$^2$ quantile")
    ax_qq.set_title("Q-Q Calibration")
    ax_qq.legend()

    label_panels(axes, x=-0.18, y=1.04)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def plot_risk_coverage(mu: torch.Tensor, y: torch.Tensor, scale: torch.Tensor, save_path: Path) -> None:
    """Risk-coverage curve: coverage vs MAE when retaining most confident fraction."""
    setup_tpami_style()

    residual = torch.abs(mu - y)
    mae_per_sample = residual.mean(dim=-1)
    uncertainty = torch.diagonal(scale, dim1=-2, dim2=-1).sum(dim=-1)

    sorted_idx = torch.argsort(uncertainty)
    sorted_mae = mae_per_sample[sorted_idx].numpy()

    fractions = np.linspace(0.1, 1.0, 50)
    risks = [sorted_mae[: int(f * len(sorted_mae))].mean() for f in fractions]

    fig, ax = plt.subplots(figsize=cm2inch(10, 7))
    ax.plot(fractions * 100, risks, "-", color=PALETTE[0], linewidth=2.5)
    ax.axhline(risks[-1], color=COLORS["dark_gray"], linestyle="--", linewidth=1.2, label="Full-set MAE")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("MAE (GPa)")
    ax.set_title("Risk-Coverage Curve")
    ax.legend()
    ax.set_xlim(10, 100)

    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints_elasticity", help="Directory with trained model.")
    parser.add_argument("--output_dir", default="figures/elasticity", help="Where figures are saved.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (checkpoint_dir / "args.json").exists():
        raise FileNotFoundError(f"args.json not found in {checkpoint_dir}. Run train_elasticity.py first.")
    if not (checkpoint_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"best_model.pt not found in {checkpoint_dir}.")

    model, train_args = load_model(checkpoint_dir, args.device)

    train_loader, _, test_loader = get_elasticity_irreps_loaders(
        data_dir=train_args.data_dir,
        batch_size=train_args.batch_size,
        num_workers=train_args.num_workers if hasattr(train_args, "num_workers") else 0,
    )

    # Obtain train stats for unnormalization.
    if isinstance(train_loader.dataset, torch.utils.data.Subset):
        train_dataset = train_loader.dataset.dataset
    else:
        train_dataset = train_loader.dataset
    mean_21d = train_dataset.mean_21d
    std_21d = train_dataset.std_21d

    preds = collect_predictions(model, test_loader, args.device)

    with open(checkpoint_dir / "history.json") as f:
        history = json.load(f)

    # Unnormalize predictions and targets to physical 21D for parity plot.
    mean_t = torch.tensor(mean_21d, dtype=torch.float32)
    std_t = torch.tensor(std_21d, dtype=torch.float32)
    pred_21d_norm = irreps_to_elasticity_21d(preds["mu_irreps"])
    pred_21d = pred_21d_norm * std_t + mean_t

    test_21d_norm = torch.stack([test_loader.dataset[i].y for i in range(len(test_loader.dataset))])
    test_21d = test_21d_norm * std_t + mean_t

    plot_training_curves(history, output_dir / "elasticity_training_curves")
    plot_parity(pred_21d.numpy(), test_21d.numpy(), output_dir / "elasticity_parity")
    plot_calibration(preds["mu_irreps"], preds["y_irreps"], preds["scale"], output_dir / "elasticity_calibration")
    plot_risk_coverage(preds["mu_irreps"], preds["y_irreps"], preds["scale"], output_dir / "elasticity_risk_coverage")

    cal_err = calibration_error(preds["mu_irreps"], preds["y_irreps"], preds["scale"])
    coverage = empirical_coverage(preds["mu_irreps"], preds["y_irreps"], preds["scale"])
    print(f"ECE: {cal_err['ece']:.4f}, ACE: {cal_err['ace']:.4f}")
    print(f"Coverage: {coverage}")
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
