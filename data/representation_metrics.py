"""Equivariant target metrics for finite-dimensional output representations."""

from __future__ import annotations

import torch


def infer_rank2_block_metric(dataset, *, eps: float = 1e-3) -> tuple[torch.Tensor, dict[str, float]]:
    """Infer a ``0e + 2e`` metric from training targets only.

    The scalar channel is scaled by its standard deviation.  The five
    ``2e`` channels share one scale based on their average squared norm, which
    is the only diagonal scaling that commutes with every ``O(3)`` action.
    """
    scalar_values: list[torch.Tensor] = []
    l2_energy: list[torch.Tensor] = []
    for index in range(len(dataset)):
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
    metric = torch.tensor(
        [1.0 / scalar_scale] + [1.0 / l2_scale] * 5,
        dtype=torch.float32,
    )
    return metric, {
        "scalar_std": scalar_scale,
        "l2_rms_per_component": l2_scale,
        "metric_scalar": float(metric[0]),
        "metric_l2": float(metric[1]),
    }
