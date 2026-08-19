"""Equivariant isospectral orientation calibration."""

from __future__ import annotations

import torch

from compatibility.e3nn import o3
from representations import O3AdaptiveLifting, O3SkewOperatorBasis


class EquivariantIsospectralOrientationCalibrator(torch.nn.Module):
    """Predict an equivariant skew generator and conjugate an SPD scale.

    The coefficient head is target-directed: its output is compiled from
    ``Lambda^2(output_irreps)`` by the same O(3) lifting planner used by the
    probabilistic readout.  Zero initialization makes the module an exact
    identity calibrator at the start of a staged calibration phase.
    """

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        *,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_irreps = o3.Irreps(output_irreps)
        self.operator_basis = O3SkewOperatorBasis(self.output_irreps)
        self.coefficient_head = O3AdaptiveLifting(
            self.hidden_irreps,
            self.operator_basis.operator_irreps,
        )
        if zero_init:
            for parameter in self.coefficient_head.parameters():
                torch.nn.init.zeros_(parameter)

    @property
    def generator_irreps(self) -> o3.Irreps:
        return self.operator_basis.operator_irreps

    def generator(self, pooled_features: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficient_head(pooled_features)
        return self.operator_basis.assemble(coefficients)

    def forward(
        self,
        pooled_features: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if scale.shape[-2:] != (self.operator_basis.output_dim,) * 2:
            raise ValueError("scale dimensions do not match output_irreps")
        generator = self.generator(pooled_features)
        orthogonal = torch.linalg.matrix_exp(generator)
        calibrated = orthogonal @ scale @ orthogonal.transpose(-1, -2)
        return 0.5 * (calibrated + calibrated.transpose(-1, -2))
