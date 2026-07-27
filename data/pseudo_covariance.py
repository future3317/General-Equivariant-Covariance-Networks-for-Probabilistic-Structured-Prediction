"""Train-only, O(3)-safe residual-covariance pseudo-labels.

This module deliberately distinguishes a residual covariance from a
Student-t scale.  Invariant kNN does *not* align neighbour output frames, so
it must never be used to supervise a directional matrix on a query.  The
only executable dielectric mode is therefore the isotropic projection of the
local residual covariance.  A directional mode requires an independently
verified transport certificate and is rejected here until one exists.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch


PSEUDO_CACHE_VERSION = 1


def invariant_structure_embedding(data, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """A deterministic O(3)/translation/permutation invariant graph descriptor.

    It uses only centered pairwise distances and pooled scalar atom features;
    it is intentionally independent of labels and fold-specific model bases.
    """
    pos = data.pos.detach().to(dtype=dtype, device="cpu")
    centered = pos - pos.mean(dim=0, keepdim=True)
    radius = torch.linalg.vector_norm(centered, dim=-1)
    pair = torch.pdist(pos)
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=dtype)
    def summary(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros(7, dtype=dtype)
        return torch.cat((
            values.mean().reshape(1), values.std(unbiased=False).reshape(1),
            torch.quantile(values, quantiles),
        ))
    parts = [summary(radius), summary(pair)]
    if hasattr(data, "node_features"):
        features = data.node_features.detach().to(dtype=dtype, device="cpu")
        # Mean/std retains scalar chemical content while remaining invariant
        # under atom ordering; restricting to a fixed prefix bounds cache size.
        features = features[:, : min(features.shape[-1], 16)]
        parts.extend((features.mean(0), features.std(0, unbiased=False)))
    elif hasattr(data, "z"):
        z = data.z.detach().to(dtype=dtype, device="cpu")
        parts.append(torch.stack((z.mean(), z.std(unbiased=False), z.min(), z.max())))
    return torch.cat(parts).contiguous()


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_oof_residual_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("split") != "train":
        raise ValueError("pseudo-labels require an OOF cache built exclusively from the train split")
    if int(payload.get("folds", 0)) != 5:
        raise ValueError("dielectric pseudo-labels require exactly five OOF folds")
    residuals = payload.get("residuals")
    assignments = payload.get("fold_assignments")
    if not isinstance(residuals, torch.Tensor) or not isinstance(assignments, torch.Tensor):
        raise ValueError("invalid OOF cache: residuals and fold_assignments tensors are required")
    if residuals.ndim != 2 or assignments.shape != (residuals.shape[0],):
        raise ValueError("invalid OOF residual dimensions")
    if not torch.isfinite(residuals).all() or not torch.isfinite(assignments).all():
        raise ValueError("OOF cache contains non-finite values")


def build_isotropic_pseudo_covariance(
    residuals: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    k: int,
    tau: float,
    shrinkage: float,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Build the O(3)-safe isotropic projection of local residual covariance.

    ``residuals`` are in an orthonormal irrep coordinate system.  Their local
    covariance trace is invariant even though its directional eigenvectors are
    not comparable between independently oriented structures.
    """
    if k < 1 or k >= residuals.shape[0]:
        raise ValueError("k must lie in [1, number_of_train_samples - 1]")
    if tau <= 0 or epsilon <= 0 or not 0 <= shrinkage <= 1:
        raise ValueError("require tau, epsilon > 0 and shrinkage in [0, 1]")
    residuals = residuals.detach().to(dtype=torch.float64, device="cpu")
    embeddings = embeddings.detach().to(dtype=torch.float64, device="cpu")
    if embeddings.ndim != 2 or embeddings.shape[0] != residuals.shape[0]:
        raise ValueError("embedding rows must match residual rows")
    distance2 = torch.cdist(embeddings, embeddings).square()
    distance2.fill_diagonal_(float("inf"))
    selected_distance2, neighbours = torch.topk(distance2, k=k, dim=1, largest=False)
    unnormalized = torch.exp(-selected_distance2 / tau)
    weights = unnormalized / unnormalized.sum(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    local = residuals[neighbours]
    mean = torch.einsum("nk,nkd->nd", weights, local)
    delta = local - mean[:, None, :]
    covariance = torch.einsum("nk,nki,nkj->nij", weights, delta, delta)
    dimension = residuals.shape[-1]
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    isotropic_variance = trace / dimension
    # Ledoit-style isotropic shrinkage leaves an isotropic projection unchanged
    # numerically, but recording it is important: the full local estimator was
    # shrunk before its safe invariant projection is used.
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * isotropic_variance[:, None, None] * torch.eye(dimension, dtype=torch.float64)[None]
    variance = shrunk.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(epsilon)
    covariance_target = variance[:, None, None] * torch.eye(dimension, dtype=torch.float64)[None]
    effective = 1.0 / weights.square().sum(dim=1)
    return {
        "embeddings": embeddings,
        "neighbours": neighbours,
        "weights": weights,
        "effective_neighbours": effective,
        "raw_residual_covariance": covariance,
        "isotropic_variance": variance,
        "covariance": covariance_target,
        "sqrt_covariance": variance.sqrt()[:, None, None] * torch.eye(dimension, dtype=torch.float64)[None],
    }


def validate_pseudo_cache(payload: Mapping[str, Any], *, expected_size: int | None = None) -> None:
    if payload.get("version") != PSEUDO_CACHE_VERSION or payload.get("split") != "train":
        raise ValueError("pseudo-label cache must be the current train-only format")
    if payload.get("mode") != "isotropic_only":
        raise ValueError("directional pseudo-labels require a verified transport certificate and are unavailable")
    if payload.get("coordinate_semantics") != "residual_covariance":
        raise ValueError("pseudo-label cache does not describe residual covariance")
    if payload.get("transport_certificate") is not None:
        raise ValueError("isotropic-only cache must not claim a directional transport certificate")
    for key in ("covariance", "sqrt_covariance", "isotropic_variance", "neighbours", "weights"):
        if not isinstance(payload.get(key), torch.Tensor):
            raise ValueError(f"pseudo-label cache is missing tensor {key!r}")
    covariance = payload["covariance"]
    if expected_size is not None and covariance.shape[0] != expected_size:
        raise ValueError("pseudo-label cache does not cover exactly the train split")
    if covariance.ndim != 3 or covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError("pseudo-label covariance has invalid shape")
    if not torch.isfinite(covariance).all():
        raise ValueError("pseudo-label covariance contains non-finite values")
