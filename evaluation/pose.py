"""Metrics for probabilistic articulated 3D pose prediction."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2, f, rankdata


def as_joint_positions(pose: torch.Tensor, num_joints: int = 15) -> torch.Tensor:
    if pose.shape[-2:] == (num_joints, 3):
        return pose
    if pose.shape[-1] != num_joints * 3:
        raise ValueError(f"pose must end in {num_joints * 3} or ({num_joints}, 3)")
    return pose.reshape(*pose.shape[:-1], num_joints, 3)


def joint_errors(pose: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-joint Euclidean errors in the input coordinate unit."""
    pose_joints = as_joint_positions(pose)
    target_joints = as_joint_positions(target)
    return torch.linalg.vector_norm(pose_joints - target_joints, dim=-1)


def mpjpe(pose: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return joint_errors(pose, target).mean()


def pck(
    pose: torch.Tensor,
    target: torch.Tensor,
    threshold_meters: float,
) -> torch.Tensor:
    return (joint_errors(pose, target) <= threshold_meters).to(pose.dtype).mean()


def risk_coverage_auc(risk: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    """Area under selective risk as increasingly uncertain items are retained."""
    if risk.ndim != 1 or error.ndim != 1 or risk.shape != error.shape:
        raise ValueError("risk and error must be equal-length vectors")
    order = torch.argsort(risk)
    ordered_error = error[order]
    coverage_count = torch.arange(
        1, len(order) + 1, device=error.device, dtype=error.dtype
    )
    selective_risk = torch.cumsum(ordered_error, dim=0) / coverage_count
    return selective_risk.mean()


def marginal_joint_covariances(
    covariance: torch.Tensor, num_joints: int = 15
) -> torch.Tensor:
    """Extract the ``3 x 3`` marginal covariance of every joint."""
    expected = num_joints * 3
    if covariance.shape[-2:] != (expected, expected):
        raise ValueError(f"covariance must end in ({expected}, {expected})")
    blocks = [
        covariance[..., 3 * joint : 3 * joint + 3, 3 * joint : 3 * joint + 3]
        for joint in range(num_joints)
    ]
    return torch.stack(blocks, dim=-3)


def joint_mahalanobis_squared(
    pose: torch.Tensor,
    target: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    residual = as_joint_positions(target) - as_joint_positions(pose)
    marginal = marginal_joint_covariances(covariance, residual.shape[-2])
    solved = torch.linalg.solve(marginal, residual.unsqueeze(-1)).squeeze(-1)
    return torch.sum(residual * solved, dim=-1)


def calibration_absolute_error(
    mahalanobis2: torch.Tensor,
    degrees_of_freedom: int,
    levels: tuple[float, ...] = tuple(index / 10 for index in range(1, 10)),
    *,
    student_t_dof: float | None = None,
) -> float:
    """Mean absolute coverage error for Gaussian or Student-t residuals."""
    values = mahalanobis2.detach().cpu().numpy().reshape(-1)
    if student_t_dof is None:
        thresholds = [chi2.ppf(level, df=degrees_of_freedom) for level in levels]
    else:
        if student_t_dof <= 0:
            raise ValueError("student_t_dof must be positive")
        thresholds = [
            degrees_of_freedom * f.ppf(level, dfn=degrees_of_freedom, dfd=student_t_dof)
            for level in levels
        ]
    observed = np.array([np.mean(values <= threshold) for threshold in thresholds])
    return float(np.mean(np.abs(observed - np.asarray(levels))))


def per_joint_marginal_coverage(
    mahalanobis2: torch.Tensor,
    *,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    student_t_dof: float | None = None,
) -> dict[str, list[float] | float]:
    """Coverage of each 3D joint marginal and its aggregate MACE."""
    if mahalanobis2.ndim != 2:
        raise ValueError("joint mahalanobis values must have shape (frames, joints)")
    values = mahalanobis2.detach().cpu().numpy()
    if student_t_dof is None:
        thresholds = [chi2.ppf(level, df=3) for level in levels]
    else:
        if student_t_dof <= 0:
            raise ValueError("student_t_dof must be positive")
        thresholds = [3 * f.ppf(level, dfn=3, dfd=student_t_dof) for level in levels]
    observed = np.stack(
        [(values <= threshold).mean(axis=0) for threshold in thresholds], axis=0
    )
    errors = np.abs(observed - np.asarray(levels)[:, None])
    return {
        "levels": list(levels),
        "coverage_by_level_and_joint": observed.tolist(),
        "mace_by_joint": errors.mean(axis=0).tolist(),
        "mace": float(errors.mean()),
    }


def joint_residual_correlation(
    pose: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Rotation-invariant Pearson-like correlation between vector residuals."""
    num_joints = (
        pose.shape[-2]
        if pose.ndim >= 3 and pose.shape[-1] == 3
        else pose.shape[-1] // 3
    )
    residual = as_joint_positions(target, num_joints) - as_joint_positions(
        pose, num_joints
    )
    centered = residual - residual.mean(dim=0, keepdim=True)
    covariance = torch.einsum("fja,fka->jk", centered, centered)
    energy = torch.diagonal(covariance).clamp_min(torch.finfo(covariance.dtype).eps)
    return covariance / torch.sqrt(energy[:, None] * energy[None, :])


def residual_correlation_by_graph_distance(
    correlation: torch.Tensor,
    edges: tuple[tuple[int, int], ...],
) -> dict[str, float]:
    """Average residual correlation grouped by shortest skeleton distance."""
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be a square joint matrix")
    num_nodes = correlation.shape[0]
    adjacency = [[] for _ in range(num_nodes)]
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    grouped: dict[int, list[torch.Tensor]] = {}
    for source in range(num_nodes):
        distance = [-1] * num_nodes
        distance[source] = 0
        queue = [source]
        for node in queue:
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        for target in range(source + 1, num_nodes):
            if distance[target] > 0:
                grouped.setdefault(distance[target], []).append(
                    correlation[source, target]
                )
    return {
        str(distance): float(torch.stack(values).mean().item())
        for distance, values in sorted(grouped.items())
    }


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Tie-aware AUROC for scalar uncertainty scores and binary OOD labels."""
    scores_np = scores.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().astype(bool).reshape(-1)
    positives = int(labels_np.sum())
    negatives = len(labels_np) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative examples")
    ranks = rankdata(scores_np, method="average")
    statistic = ranks[labels_np].sum() - positives * (positives + 1) / 2
    return float(statistic / (positives * negatives))


def visible_occluded_mpjpe(
    pose: torch.Tensor,
    target: torch.Tensor,
    visible: torch.Tensor,
) -> dict[str, float]:
    errors = joint_errors(pose, target)
    visible = visible.bool()
    result: dict[str, float] = {}
    if visible.any():
        result["visible_mpjpe_m"] = float(errors[visible].mean().item())
    if (~visible).any():
        result["occluded_mpjpe_m"] = float(errors[~visible].mean().item())
    return result


def occlusion_uncertainty_ratio(
    covariance: torch.Tensor,
    visible: torch.Tensor,
) -> float:
    """Ratio of marginal variance on occluded versus visible joints."""
    marginal = marginal_joint_covariances(covariance, visible.shape[-1])
    uncertainty = torch.diagonal(marginal, dim1=-2, dim2=-1).sum(-1)
    visible = visible.bool()
    if not visible.any() or not (~visible).any():
        return float("nan")
    return float((uncertainty[~visible].mean() / uncertainty[visible].mean()).item())


def bone_length_error(
    samples: torch.Tensor,
    target: torch.Tensor,
    edges: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Mean absolute bone-length error for sampled coherent poses."""
    sample_joints = as_joint_positions(samples)
    target_joints = as_joint_positions(target)
    sample_lengths = torch.stack(
        [
            torch.linalg.vector_norm(
                sample_joints[..., target_node, :] - sample_joints[..., source, :],
                dim=-1,
            )
            for source, target_node in edges
        ],
        dim=-1,
    )
    target_lengths = torch.stack(
        [
            torch.linalg.vector_norm(
                target_joints[..., target_node, :] - target_joints[..., source, :],
                dim=-1,
            )
            for source, target_node in edges
        ],
        dim=-1,
    )
    while target_lengths.ndim < sample_lengths.ndim:
        target_lengths = target_lengths.unsqueeze(-2)
    return torch.abs(sample_lengths - target_lengths).mean()
