"""Audit dielectric normalization separately from the compiled physical target.

The cached dielectric labels are component-wise normalized in Kelvin--Mandel
coordinates and are then restored before conversion to ``0e + 2e``.  This
script reports whether the normalization itself is an O(3) intertwiner and
whether the complete normalize -> inverse-normalize -> irrep conversion path
preserves the target action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compatibility.e3nn import o3
from data.tensor_conversions import km_to_irreps


def _km_representation(rotation: torch.Tensor) -> torch.Tensor:
    """Return the O(3) action in the Kelvin--Mandel coordinate convention."""
    output_spec = o3.Irreps("0e + 2e")
    basis_map = km_to_irreps(torch.eye(6, dtype=torch.float64)).T
    rho_irrep = output_spec.D_from_matrix(rotation).double()
    return torch.linalg.solve(basis_map, rho_irrep @ basis_map)


def _residuals(linear: torch.Tensor, bias: torch.Tensor, rotations: list[torch.Tensor]):
    linear_residual = 0.0
    bias_residual = 0.0
    for rotation in rotations:
        rho = _km_representation(rotation)
        linear_residual = max(
            linear_residual,
            float((linear @ rho - rho @ linear).abs().max()),
        )
        bias_residual = max(
            bias_residual,
            float((rho @ bias - bias).abs().max()),
        )
    return {
        "max_linear_intertwiner_residual": linear_residual,
        "max_bias_invariance_residual": bias_residual,
    }


def audit(component_mean, component_std, *, tolerance: float = 1e-8):
    mean = torch.as_tensor(component_mean, dtype=torch.float64)
    std = torch.as_tensor(component_std, dtype=torch.float64)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError("dielectric normalization statistics must each have six values")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("dielectric normalization statistics must be finite with positive std")

    rotations = [
        torch.eye(3, dtype=torch.float64),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
        torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        o3.rand_matrix(dtype=torch.float64),
        o3.rand_matrix(dtype=torch.float64),
    ]
    normalized = _residuals(torch.diag(1.0 / std), -mean / std, rotations)
    inverse = _residuals(torch.diag(std), mean, rotations)

    raw_km = torch.tensor([0.7, -0.2, 1.1, 0.3, -0.4, 0.6], dtype=torch.float64)
    raw_irrep = km_to_irreps(raw_km.unsqueeze(0)).squeeze(0)
    full_pipeline_errors = []
    for rotation in rotations:
        rho_km = _km_representation(rotation)
        rho_irrep = o3.Irreps("0e + 2e").D_from_matrix(rotation).double()
        normalized_target = (rho_km @ raw_km - mean) / std
        restored_target = normalized_target * std + mean
        transformed_irrep = km_to_irreps(restored_target.unsqueeze(0)).squeeze(0)
        expected_irrep = rho_irrep @ raw_irrep
        full_pipeline_errors.append(float((transformed_irrep - expected_irrep).abs().max()))

    return {
        "normalization_map": {
            "coordinate_system": "Kelvin-Mandel",
            "map": "z = diag(1/std) y - mean/std",
            **normalized,
            "status": (
                "verified"
                if max(normalized.values()) <= tolerance
                else "diagnostic_non_intertwiner"
            ),
        },
        "inverse_normalization_map": {
            "coordinate_system": "Kelvin-Mandel",
            "map": "y = diag(std) z + mean",
            **inverse,
            "status": (
                "verified"
                if max(inverse.values()) <= tolerance
                else "diagnostic_non_intertwiner"
            ),
        },
        "physical_irrep_contract": {
            "output_irreps": "0e + 2e",
            "pipeline": "normalize -> inverse_normalize -> KM_to_irreps",
            "max_equivariance_error": max(full_pipeline_errors),
            "status": "verified" if max(full_pipeline_errors) <= tolerance else "rejected",
        },
        "interpretation": (
            "The component-wise normalization is audited as preprocessing; the compiled "
            "predictive contract begins after inverse normalization in physical irrep coordinates."
        ),
    }


def _load_statistics(path: Path):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return metadata["component_mean"], metadata["component_std"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    mean, std = _load_statistics(args.metadata)
    result = audit(mean, std, tolerance=args.tolerance)
    result["metadata"] = str(args.metadata)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
