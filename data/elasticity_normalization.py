"""Representation-compatible normalization for elasticity targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from compatibility.e3nn import o3
from data.representation_metrics import fit_multiplicity_whitening
from data.tensor_conversions import elasticity_21d_to_irreps, irreps_to_elasticity_21d
from representations import rank4_elasticity_irreps


@dataclass(frozen=True)
class ElasticityNormalizationStats:
    """Statistics fitted on one training split."""

    mode: str
    mean_21d: np.ndarray | None = None
    std_21d: np.ndarray | None = None
    mean_irreps: np.ndarray | None = None
    scale_irreps: np.ndarray | None = None
    whitening_irreps: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mean_21d": None if self.mean_21d is None else self.mean_21d.tolist(),
            "std_21d": None if self.std_21d is None else self.std_21d.tolist(),
            "mean_irreps": None
            if self.mean_irreps is None
            else self.mean_irreps.tolist(),
            "scale_irreps": None
            if self.scale_irreps is None
            else self.scale_irreps.tolist(),
            "whitening_irreps": None
            if self.whitening_irreps is None
            else self.whitening_irreps.tolist(),
        }


def _irrep_blocks(irreps: o3.Irreps):
    offset = 0
    for multiplicity, irrep in irreps:
        width = int(multiplicity * irrep.dim)
        yield offset, offset + width, irrep
        offset += width


class ElasticityTargetNormalizer:
    """Map physical 21D targets to normalized rank-4 irrep coordinates.

    ``legacy_voigt`` reproduces the historical component-wise Voigt path.  It
    is retained for numerical reproducibility, but its per-coordinate affine
    map is not representation-compatible in general and therefore carries no
    equivariance guarantee for the normalized targets.
    ``representation_compatible`` converts to irreps first, centers only
    invariant ``0e`` coordinates, and applies one positive scale per isotypic
    block.  ``representation_compatible_multiplicity`` replaces that scalar
    with a positive whitening operator on each multiplicity space.  Both
    operations commute with every orthogonal representation matrix of the
    rank-4 elasticity output.
    """

    _IRREPS = o3.Irreps(rank4_elasticity_irreps())

    def __init__(self, stats: ElasticityNormalizationStats, *, eps: float = 1e-8):
        if stats.mode not in {
            "legacy_voigt",
            "representation_compatible",
            "representation_compatible_multiplicity",
        }:
            raise ValueError(f"unknown elasticity normalization mode: {stats.mode}")
        self.stats = stats
        self.eps = float(eps)
        if stats.mode == "legacy_voigt":
            if stats.mean_21d is None or stats.std_21d is None:
                raise ValueError("legacy_voigt requires mean_21d and std_21d")
        elif stats.mode == "representation_compatible" and (
            stats.mean_irreps is None or stats.scale_irreps is None
        ):
            raise ValueError(
                "representation_compatible requires mean_irreps and scale_irreps"
            )
        elif stats.mode == "representation_compatible_multiplicity" and (
            stats.mean_irreps is None or stats.whitening_irreps is None
        ):
            raise ValueError(
                "representation_compatible_multiplicity requires mean_irreps "
                "and whitening_irreps"
            )

    @classmethod
    def fit(
        cls,
        elasticity_21d: np.ndarray,
        mode: str = "legacy_voigt",
        *,
        eps: float = 1e-8,
    ) -> ElasticityTargetNormalizer:
        values = np.asarray(elasticity_21d, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 21:
            raise ValueError("elasticity_21d must have shape (N, 21)")
        if values.shape[0] == 0:
            raise ValueError("cannot fit normalization on an empty dataset")

        if mode == "legacy_voigt":
            stats = ElasticityNormalizationStats(
                mode=mode,
                mean_21d=values.mean(axis=0),
                std_21d=values.std(axis=0) + eps,
            )
            return cls(stats, eps=eps)
        if mode not in {
            "representation_compatible",
            "representation_compatible_multiplicity",
        }:
            raise ValueError(f"unknown elasticity normalization mode: {mode}")

        irrep_values = elasticity_21d_to_irreps(torch.from_numpy(values).float())
        irrep_values = irrep_values.double().numpy()
        mean = np.zeros(21, dtype=np.float64)
        scale = np.zeros(21, dtype=np.float64)
        for start, stop, irrep in _irrep_blocks(cls._IRREPS):
            block = irrep_values[:, start:stop]
            if irrep.l == 0:
                # Every 0e copy is invariant; only these channels may be shifted.
                mean[start:stop] = block.mean(axis=0)
            centered = block - mean[start:stop]
            block_rms = float(np.sqrt(np.mean(centered * centered)))
            scale[start:stop] = max(block_rms, eps)
        if mode == "representation_compatible":
            stats = ElasticityNormalizationStats(
                mode=mode, mean_irreps=mean, scale_irreps=scale
            )
        else:
            centered = irrep_values - mean
            whitening, _ = fit_multiplicity_whitening(
                torch.from_numpy(centered), cls._IRREPS, eps=eps
            )
            stats = ElasticityNormalizationStats(
                mode=mode,
                mean_irreps=mean,
                whitening_irreps=whitening.double().numpy(),
            )
        return cls(stats, eps=eps)

    @classmethod
    def from_stats(
        cls, stats: tuple[np.ndarray, np.ndarray] | dict | ElasticityNormalizationStats
    ) -> ElasticityTargetNormalizer:
        if isinstance(stats, tuple):
            stats = ElasticityNormalizationStats(
                mode="legacy_voigt", mean_21d=np.asarray(stats[0]), std_21d=np.asarray(stats[1])
            )
        if isinstance(stats, dict):
            stats = ElasticityNormalizationStats(
                mode=stats["mode"],
                mean_21d=_array_or_none(stats.get("mean_21d")),
                std_21d=_array_or_none(stats.get("std_21d")),
                mean_irreps=_array_or_none(stats.get("mean_irreps")),
                scale_irreps=_array_or_none(stats.get("scale_irreps")),
                whitening_irreps=_array_or_none(stats.get("whitening_irreps")),
            )
        return cls(stats)

    @property
    def mean_21d(self) -> np.ndarray | None:
        return self.stats.mean_21d

    @property
    def std_21d(self) -> np.ndarray | None:
        return self.stats.std_21d

    def transform(self, values_21d: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(values_21d)
        if values.shape[-1] != 21:
            raise ValueError("values_21d must end in dimension 21")
        if self.stats.mode == "legacy_voigt":
            mean = _tensor(self.stats.mean_21d, values)
            std = _tensor(self.stats.std_21d, values)
            return elasticity_21d_to_irreps((values - mean) / std)
        irreps = elasticity_21d_to_irreps(values)
        mean = _tensor(self.stats.mean_irreps, values)
        if self.stats.mode == "representation_compatible_multiplicity":
            whitening = _tensor(self.stats.whitening_irreps, values)
            return (irreps - mean) @ whitening.transpose(-1, -2)
        scale = _tensor(self.stats.scale_irreps, values)
        return (irreps - mean) / scale

    def inverse(self, normalized_irreps: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(normalized_irreps)
        if values.shape[-1] != 21:
            raise ValueError("normalized_irreps must end in dimension 21")
        if self.stats.mode == "legacy_voigt":
            normalized_21d = irreps_to_elasticity_21d(values)
            return normalized_21d * _tensor(self.stats.std_21d, values) + _tensor(
                self.stats.mean_21d, values
            )
        mean = _tensor(self.stats.mean_irreps, values)
        if self.stats.mode == "representation_compatible_multiplicity":
            whitening = _tensor(self.stats.whitening_irreps, values)
            centered = torch.linalg.solve(
                whitening, values.unsqueeze(-1)
            ).squeeze(-1)
            return irreps_to_elasticity_21d(centered + mean)
        scale = _tensor(self.stats.scale_irreps, values)
        return irreps_to_elasticity_21d(values * scale + mean)


def _array_or_none(value) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=np.float64)


def _tensor(value: np.ndarray | None, reference: torch.Tensor) -> torch.Tensor:
    if value is None:
        raise ValueError("normalization statistics are incomplete")
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
