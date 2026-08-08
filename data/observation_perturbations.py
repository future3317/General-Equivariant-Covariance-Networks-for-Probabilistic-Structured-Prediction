"""Input-only perturbations for depth-observation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthPerturbationScale:
    """Perturbation strengths estimated without labels or held-out errors."""

    missing_fraction: float
    depth_noise_std: float

    def __post_init__(self) -> None:
        if not 0.0 < self.missing_fraction < 1.0:
            raise ValueError("missing_fraction must lie in (0, 1)")
        if self.depth_noise_std <= 0.0:
            raise ValueError("depth_noise_std must be positive")


def perturb_depth_observation(
    depth: np.ndarray,
    *,
    kind: str,
    scale: DepthPerturbationScale,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one physical input perturbation while preserving invalid pixels."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be a matrix, got shape {depth.shape}")
    if kind not in {"missing_block", "point_dropout", "depth_noise"}:
        raise ValueError(f"unsupported depth perturbation: {kind}")
    result = depth.copy()
    valid = np.isfinite(result) & (result > 0.0)
    if not valid.any():
        raise ValueError("depth observation has no valid pixels")

    if kind == "point_dropout":
        removed = valid & (rng.random(result.shape) < scale.missing_fraction)
        result[removed] = 0.0
    elif kind == "missing_block":
        height, width = result.shape
        side_fraction = np.sqrt(scale.missing_fraction)
        block_height = max(1, min(height, round(height * side_fraction)))
        block_width = max(1, min(width, round(width * side_fraction)))
        top = int(rng.integers(0, height - block_height + 1))
        left = int(rng.integers(0, width - block_width + 1))
        result[top : top + block_height, left : left + block_width] = 0.0
    else:
        noise = rng.normal(0.0, scale.depth_noise_std, size=result.shape)
        result[valid] = np.maximum(
            result[valid] + noise[valid].astype(np.float32),
            np.finfo(np.float32).eps,
        )

    if not (np.isfinite(result) & (result > 0.0)).any():
        raise ValueError("perturbation removed the complete observation")
    return result
