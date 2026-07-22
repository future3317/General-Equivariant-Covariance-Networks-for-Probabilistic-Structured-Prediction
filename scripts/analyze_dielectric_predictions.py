"""Audit mean/uncertainty alignment for a trained dielectric checkpoint.

This is deliberately separate from the paper-figure generator: the audit
reports coordinate-wise signal strength, residual correlation, and predictive
scale in Kelvin--Mandel coordinates so a poor parity plot is not confused
with a calibration failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.dielectric_dataset import get_dielectric_irreps_loaders
from data.tensor_conversions import irreps_to_km
from scripts.generate_dielectric_figures import collect_predictions, load_model


def _covariance_to_km(scale_irreps: torch.Tensor) -> torch.Tensor:
    """Change a covariance from e3nn ``0e+2e`` coordinates to KM coordinates."""
    basis = irreps_to_km(torch.eye(6, dtype=scale_irreps.dtype, device=scale_irreps.device))
    return torch.einsum("ab,nbc,dc->nad", basis, scale_irreps, basis)


@torch.inference_mode()
def audit(checkpoint_dir: Path, device: str) -> dict:
    model, args = load_model(checkpoint_dir, device)
    _, _, loader = get_dielectric_irreps_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=getattr(args, "num_workers", 0),
        persistent_workers=getattr(args, "persistent_workers", False),
        pin_memory=getattr(args, "pin_memory", False),
        prefetch_factor=getattr(args, "prefetch_factor", None),
        lmax=args.lmax,
        storage=getattr(args, "dataset_storage", "files"),
        shard_cache_size=getattr(args, "shard_cache_size", 2),
    )
    predictions = collect_predictions(model, loader, device)
    mu = irreps_to_km(predictions["mu_irreps"]).double()
    target = predictions["y_km"].double()
    scale = _covariance_to_km(predictions["scale_irreps"].double())
    residual = mu - target

    basis = irreps_to_km(torch.eye(6, dtype=torch.float64))
    residual_cov = torch.cov(residual.T)
    target_cov = torch.cov(target.T)
    mean_scale = scale.mean(dim=0)
    mean_corr = mean_scale / torch.sqrt(
        torch.outer(torch.diag(mean_scale), torch.diag(mean_scale))
    )
    residual_corr = residual_cov / torch.sqrt(
        torch.outer(torch.diag(residual_cov), torch.diag(residual_cov))
    )
    solved = torch.linalg.solve(scale, residual.unsqueeze(-1)).squeeze(-1)
    maha2 = (residual * solved).sum(dim=-1)

    components = []
    for index in range(6):
        truth = target[:, index]
        error = residual[:, index]
        prediction = mu[:, index]
        components.append(
            {
                "component": index + 1,
                "target_mean": float(truth.mean()),
                "target_std": float(truth.std()),
                "prediction_std": float(prediction.std()),
                "bias": float(error.mean()),
                "mae": float(error.abs().mean()),
                "rmse": float(torch.sqrt((error.square()).mean())),
                "r2": float(1 - error.square().sum() / (truth.var(unbiased=False) * len(truth) + 1e-12)),
                "pearson": float(torch.corrcoef(torch.stack((truth, prediction)))[0, 1]),
                "mean_predictive_std": float(torch.sqrt(scale[:, index, index]).mean()),
                "median_predictive_std": float(torch.sqrt(scale[:, index, index]).median()),
            }
        )

    return {
        "coordinate_space": "log_kelvin_mandel",
        "num_samples": int(target.shape[0]),
        "basis_orthogonality_error": float(torch.linalg.norm(basis @ basis.T - torch.eye(6))),
        "components": components,
        "target_covariance": target_cov.tolist(),
        "residual_covariance": residual_cov.tolist(),
        "mean_predictive_scale": mean_scale.tolist(),
        "mean_predictive_correlation": mean_corr.tolist(),
        "residual_correlation": residual_corr.tolist(),
        "mahalanobis2": {
            "mean": float(maha2.mean()),
            "median": float(maha2.median()),
            "q90": float(torch.quantile(maha2, 0.90)),
            "q95": float(torch.quantile(maha2, 0.95)),
            "q99": float(torch.quantile(maha2, 0.99)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = audit(Path(args.checkpoint_dir), args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
