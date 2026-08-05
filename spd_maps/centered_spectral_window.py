"""SPD map with independent mean log-scale and centered log-shape control."""

from __future__ import annotations

import torch

from spd_maps.base import SPDMap, symmetrize
from spd_maps.spectral_window import SpectralWindowMap


class CenteredSpectralWindowMap(SPDMap):
    """Map a symmetric generator to SPD while separating scale from shape.

    The trace component controls the common log-scale and the trace-free
    component is passed through a bounded spectral window.  Consequently the
    condition-number bound depends only on ``shape_min/max`` and is independent
    of the predicted overall uncertainty level.

    ``volume_min`` and ``volume_max`` are retained as compatibility names for
    the serialized/CLI schema. They bound the common mean log-eigenvalue
    before exponentiation; they are not bounds on log determinant.
    """

    def __init__(
        self,
        shape_min: float = -2.0,
        shape_max: float = 2.0,
        volume_min: float = -8.0,
        volume_max: float = 8.0,
    ) -> None:
        super().__init__()
        if not shape_min < shape_max or not volume_min < volume_max:
            raise ValueError("spectral bounds must be strictly increasing")
        self.shape_min = float(shape_min)
        self.shape_max = float(shape_max)
        self.volume_min = float(volume_min)
        self.volume_max = float(volume_max)
        self.shape_map = SpectralWindowMap(shape_min, shape_max)

    @property
    def max_condition_number(self) -> float:
        return float(torch.exp(torch.tensor(self.shape_max - self.shape_min)))

    def _log_spectrum(self, A: torch.Tensor) -> torch.Tensor:
        A = symmetrize(A)
        d = A.shape[-1]
        volume = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1) / d
        volume = self.volume_min + (self.volume_max - self.volume_min) * torch.sigmoid(volume)
        # Subtract the generator trace, not the bounded volume.
        raw_volume = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1) / d
        centered = A - raw_volume[..., None, None] * torch.eye(
            d, dtype=A.dtype, device=A.device
        )
        shape = self.shape_map(centered)
        zero_residual = torch.zeros(
            *centered.shape[:-1], dtype=centered.dtype, device=centered.device
        )
        shape_logdet = self.shape_map.statistics(centered, zero_residual)[0]
        shape = shape * torch.exp(-shape_logdet[..., None, None] / d)
        return torch.exp(volume)[..., None, None] * shape

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        return self._log_spectrum(A)

    def statistics(self, A: torch.Tensor, residual: torch.Tensor):
        scale = self.forward(A)
        chol = torch.linalg.cholesky(scale)
        solved = torch.cholesky_solve(residual.unsqueeze(-1), chol).squeeze(-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        return logdet, (residual * solved).sum(-1)

    def logdet(self, A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.slogdet(self.forward(A))[1]

    def precision_action(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.statistics(A, residual)[1]
