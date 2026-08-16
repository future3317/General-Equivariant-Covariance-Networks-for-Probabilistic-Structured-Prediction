"""Equivariant target metrics for finite-dimensional output representations."""

from __future__ import annotations

import torch

from compatibility.e3nn import o3


def fit_multiplicity_whitening(
    values: torch.Tensor,
    output_irreps: o3.Irreps,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit the maximal positive whitening metric in the representation commutant.

    For each repeated ``(l, parity)`` type, the returned block is the
    Kronecker product ``B_l`` with the irrep identity, where
    ``B_l=(C_l+eps I)^(-1/2)``.  It therefore
    mixes multiplicity copies but never mixes irrep coordinates, preserving
    the orthogonal representation action.
    """
    values = torch.as_tensor(values, dtype=torch.float64).detach()
    irreps = o3.Irreps(output_irreps)
    if values.ndim != 2 or values.shape[-1] != irreps.dim:
        raise ValueError(f"values must have shape (N, {irreps.dim})")
    if values.shape[0] == 0:
        raise ValueError("cannot fit a metric from an empty sample set")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    groups: dict[tuple[int, int], list[int]] = {}
    offset = 0
    for multiplicity, irrep in irreps:
        indices = list(range(offset, offset + int(multiplicity * irrep.dim)))
        groups.setdefault((irrep.l, irrep.p), []).extend(indices)
        offset += int(multiplicity * irrep.dim)

    whitening = values.new_zeros((irreps.dim, irreps.dim))
    stats: dict[str, float] = {}
    for (angular_momentum, parity), indices in groups.items():
        dimension = 2 * angular_momentum + 1
        multiplicity = len(indices) // dimension
        block = values[:, indices].reshape(-1, multiplicity, dimension)
        covariance = torch.einsum("nmd,nkd->mk", block, block) / (
            block.shape[0] * dimension
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(
            covariance + eps * torch.eye(multiplicity, dtype=values.dtype)
        )
        block_whitening = (
            eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
        ) @ eigenvectors.transpose(-1, -2)
        key = f"l{angular_momentum}{'e' if parity == 1 else 'o'}"
        stats[f"{key}_multiplicity"] = float(multiplicity)
        stats[f"{key}_condition_before"] = float(
            (eigenvalues[-1] / eigenvalues[0]).item()
        )
        full_block = torch.kron(
            block_whitening, torch.eye(dimension, dtype=values.dtype)
        )
        for row, source_row in enumerate(indices):
            for col, source_col in enumerate(indices):
                whitening[source_row, source_col] = full_block[row, col]

    stats.update(
        {
            "metric_min": float(whitening.diagonal().min()),
            "metric_max": float(whitening.diagonal().max()),
        }
    )
    return whitening.float(), stats


def infer_representation_block_metric(
    values: torch.Tensor,
    output_irreps: o3.Irreps,
    *,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Infer an equivariant diagonal metric for any finite O(3) output.

    Every isotypic block (fixed ``(l, parity)``) receives one RMS scale,
    repeated over its multiplicity and irrep components.  This is the most
    general diagonal metric that commutes with the representation; no dataset
    name or Cartesian tensor rank is involved.
    """
    values = torch.as_tensor(values).detach().float()
    irreps = o3.Irreps(output_irreps)
    if values.ndim != 2 or values.shape[-1] != irreps.dim:
        raise ValueError(f"values must have shape (N, {irreps.dim})")
    if values.shape[0] == 0:
        raise ValueError("cannot infer a metric from an empty sample set")
    metric = torch.empty(irreps.dim, dtype=torch.float32)
    stats: dict[str, float] = {}
    offset = 0
    for mul, ir in irreps:
        width = int(mul * ir.dim)
        rms = torch.sqrt(values[:, offset : offset + width].square().mean())
        scale = float(torch.clamp(rms, min=eps))
        key = f"l{ir.l}{ir.p:+d}"
        stats[f"{key}_rms"] = scale
        metric[offset : offset + width] = 1.0 / scale
        offset += width
    stats.update({"metric_min": float(metric.min()), "metric_max": float(metric.max())})
    return metric, stats


def transformed_spectral_bounds(
    log_variance_bounds: tuple[float, float], metric: torch.Tensor
) -> tuple[float, float]:
    """Bounds for physical scatter after ``S = D^{-1} S_tilde D^{-1}``.

    A spectral map constrains the scaled-coordinate scatter ``S_tilde``.  The
    corresponding physical eigenvalues lie in conservative bounds obtained
    from the smallest/largest diagonal metric entries.
    """
    lower, upper = log_variance_bounds
    metric = torch.as_tensor(metric, dtype=torch.float64)
    if metric.ndim != 1 or metric.numel() == 0 or bool((metric <= 0).any()):
        raise ValueError("metric must be a positive non-empty vector")
    return (
        float(lower - 2.0 * torch.log(metric.max()).item()),
        float(upper - 2.0 * torch.log(metric.min()).item()),
    )


def infer_rank2_block_metric(
    dataset, *, eps: float = 1e-3, max_samples: int | None = 256
) -> tuple[torch.Tensor, dict[str, float]]:
    """Infer a ``0e + 2e`` metric from training targets only.

    The scalar channel is scaled by its standard deviation.  The five
    ``2e`` channels share one scale based on their average squared norm, which
    is the only diagonal scaling that commutes with every ``O(3)`` action.
    """
    scalar_values: list[torch.Tensor] = []
    l2_energy: list[torch.Tensor] = []
    count = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    for index in range(count):
        target = dataset[index].y_irreps.reshape(-1, 6).detach().float()
        scalar_values.append(target[:, 0])
        l2_energy.append(target[:, 1:].square().mean(dim=-1))
    if not scalar_values:
        raise ValueError("cannot infer a representation metric from an empty dataset")
    scalar = torch.cat(scalar_values)
    l2_rms = torch.cat(l2_energy).mean().sqrt()
    scalar_std = scalar.std(unbiased=False)
    scalar_scale = float(torch.clamp(scalar_std, min=eps))
    l2_scale = float(torch.clamp(l2_rms, min=eps))
    metric = torch.tensor([1.0 / scalar_scale] + [1.0 / l2_scale] * 5, dtype=torch.float32)
    return metric, {
        "scalar_std": scalar_scale,
        "l2_rms_per_component": l2_scale,
        "metric_scalar": float(metric[0]),
        "metric_l2": float(metric[1]),
    }
