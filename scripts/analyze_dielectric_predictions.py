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
from evaluation import (
    irrep_resolved_whitening_defect,
    student_t_radial_pit,
    whitened_angular_defect,
    symmetric_whitened_residuals,
)
from evaluation.calibration import marginal_interval_quantile
from scripts.dielectric_runtime import (
    collect_dielectric_predictions,
    configure_inference_contract,
    inference_contract_from_args,
    inference_contract_hash,
    load_dielectric_checkpoint,
    load_dielectric_data_args,
    load_run_record,
)


def _covariance_to_km(scale_irreps: torch.Tensor) -> torch.Tensor:
    """Change a row-vector covariance from e3nn coordinates to KM coordinates.

    ``irreps_to_km`` applies the linear map on the last dimension by right
    multiplication, so a covariance transforms as ``L.T @ S @ L``.
    """
    basis = irreps_to_km(torch.eye(6, dtype=scale_irreps.dtype, device=scale_irreps.device))
    return torch.einsum("ab,nbc,cd->nad", basis.T, scale_irreps, basis)


def _covariance_projection_rank(model) -> int | None:
    """Measure local coefficient-map rank before SPD nonlinearities."""
    projection = getattr(getattr(model, "joint_head", None), "covariance_projection", None)
    if projection is None or not hasattr(projection, "irreps_in"):
        return None
    feature = torch.randn(
        1,
        projection.irreps_in.dim,
        device=next(projection.parameters()).device,
        dtype=next(projection.parameters()).dtype,
        requires_grad=True,
    )
    with torch.enable_grad():
        jacobian = torch.autograd.functional.jacobian(
            lambda value: projection(value).squeeze(0), feature
        )
    matrix = jacobian.reshape(projection.irreps_out.dim, projection.irreps_in.dim)
    return int(torch.linalg.matrix_rank(matrix.detach().double()).item())


def audit(checkpoint_dir: Path, device: str) -> dict:
    model, _, _ = load_dielectric_checkpoint(checkpoint_dir, device)
    args = load_dielectric_data_args(checkpoint_dir)
    record = load_run_record(checkpoint_dir)
    contract = record.get("inference_contract") or inference_contract_from_args(args, device)
    configure_inference_contract(contract)
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
    predictions = collect_dielectric_predictions(model, loader, device, inference_contract=contract)
    mu_irreps = predictions["mu_irreps"].double()
    target_irreps = predictions["y_irreps"].double()
    mu = irreps_to_km(mu_irreps)
    target = predictions["y_km"].double()
    scale_irreps = predictions["scale_irreps"].double()
    basis = irreps_to_km(torch.eye(6, dtype=torch.float64))
    scale = _covariance_to_km(scale_irreps)
    residual = mu - target
    distribution = getattr(args, "distribution", "gaussian")
    student_t_dof = float(getattr(args, "student_t_dof", 5.0))

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
    residual_irreps = mu_irreps - target_irreps
    maha2_irreps = (
        residual_irreps
        * torch.linalg.solve(scale_irreps, residual_irreps.unsqueeze(-1)).squeeze(-1)
    ).sum(dim=-1)
    angular_defect = whitened_angular_defect(
        mu,
        target,
        scale,
        nu=student_t_dof if distribution == "student_t" else None,
    )
    symmetric_z = symmetric_whitened_residuals(mu, target, scale)
    radial_pit = (
        student_t_radial_pit(mu, target, scale, student_t_dof)
        if distribution == "student_t"
        else None
    )
    operator_basis = getattr(getattr(model, "joint_head", None), "operator_basis", None)
    if operator_basis is None:
        raise RuntimeError("checkpoint model does not expose a compiled covariance operator basis")
    irrep_defect = irrep_resolved_whitening_defect(
        mu_irreps,
        target_irreps,
        scale_irreps,
        operator_basis,
        nu=student_t_dof if distribution == "student_t" else 1e6,
    )
    local_covariance_rank = _covariance_projection_rank(model)

    components = []
    marginal_levels = (0.5, 0.8, 0.9, 0.95)
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
                "marginal_coverage": {
                    f"coverage_{int(level * 100):02d}": float(
                        (
                            error.abs()
                            <= marginal_interval_quantile(
                                level,
                                reference=distribution,
                                student_t_dof=student_t_dof,
                            )
                            * torch.sqrt(scale[:, index, index])
                        ).double().mean()
                    )
                    for level in marginal_levels
                },
            }
        )

    return {
        "coordinate_space": "log_kelvin_mandel",
        "inference_contract": contract,
        "inference_contract_hash": inference_contract_hash(contract),
        "num_samples": int(target.shape[0]),
        "basis_orthogonality_error": float(torch.linalg.norm(basis @ basis.T - torch.eye(6))),
        "coordinate_change_check": {
            "target_max_error": float(torch.max(torch.abs(target - irreps_to_km(target_irreps)))),
            "mahalanobis2_irreps_mean": float(maha2_irreps.mean()),
            "mahalanobis2_km_mean": float(maha2.mean()),
            "scale_symmetry_error_irreps": float(torch.max(torch.abs(scale_irreps - scale_irreps.transpose(-1, -2)))),
        },
        "irrep_statistics": {
            "target_std": target_irreps.std(dim=0).tolist(),
            "prediction_std": mu_irreps.std(dim=0).tolist(),
            "residual_std": residual_irreps.std(dim=0).tolist(),
        },
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
        "symmetry_aware_calibration": {
            "whitened_angular_defect": float(angular_defect),
            "whitened_second_moment_trace": float(
                symmetric_z.square().sum(dim=-1).mean()
            ),
            "radial_pit_mean": None if radial_pit is None else float(radial_pit.mean()),
            "radial_pit_std": None if radial_pit is None else float(radial_pit.std()),
            "radial_pit_q10": None if radial_pit is None else float(torch.quantile(radial_pit, 0.10)),
            "radial_pit_q90": None if radial_pit is None else float(torch.quantile(radial_pit, 0.90)),
            "irrep_resolved_defect": irrep_defect,
        },
        "covariance_expressivity": {
            "operator_basis_dimension": int(operator_basis.operator_dim),
            "covariance_projection_local_rank": local_covariance_rank,
            "target_full_symmetric_operator_dimension": int(target.shape[-1] * (target.shape[-1] + 1) // 2),
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
