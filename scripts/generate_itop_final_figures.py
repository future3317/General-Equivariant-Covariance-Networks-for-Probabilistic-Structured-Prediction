"""Generate the two compact ITOP figures used by the TPAMI manuscript."""

# ruff: noqa: I001 -- torch must initialize before matplotlib on Windows.

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

# Import order avoids competing OpenMP runtimes when the certified map is evaluated.
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.itop_dataset import ITOP_OUTPUT_GRAPH, ITOP_SKELETON_EDGES
from equivcompiler import FeatureSpec, GraphPrecision, plan_readout
from plotting import COLORS, cm2inch, save_figure, setup_tpami_style

MODEL_LABELS = {
    "deterministic": "Det.",
    "frozen_full_student_t": "Full-t",
    "frozen_independent_gaussian": "Indep-G",
    "frozen_independent_student_t": "Indep-t",
    "frozen_low_rank_student_t": "LR-t",
    "frozen_graph_gaussian": "Graph-G",
    "frozen_graph_student_t": "Graph-t",
}

METHOD_COLORS = {
    "deterministic": COLORS["dark_gray"],
    "frozen_full_student_t": COLORS["violet"],
    "frozen_independent_gaussian": COLORS["red_strong"],
    "frozen_independent_student_t": COLORS["red_strong"],
    "frozen_low_rank_student_t": COLORS["teal"],
    "frozen_graph_gaussian": COLORS["blue_main"],
    "frozen_graph_student_t": COLORS["blue_main"],
}

ACTIVE_COORDS = {
    "frozen_full_student_t": 1035,
    "frozen_independent_gaussian": 90,
    "frozen_independent_student_t": 90,
    "frozen_low_rank_student_t": 181,
    "frozen_graph_gaussian": 174,
    "frozen_graph_student_t": 174,
}

ROBUSTNESS_MODELS = {
    "full_student_t": "frozen_full_student_t",
    "low_rank_student_t": "frozen_low_rank_student_t",
    "graph_student_t": "frozen_graph_student_t",
}


def _seed_root(root: Path) -> Path:
    """Locate the canonical seed artifact under a result or figure-input root."""
    candidates = (root / "seed_42", root / "figures_input" / "seed_42")
    for candidate in candidates:
        if (candidate / "frozen_graph_student_t" / "metrics.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find seed_42 artifacts under "
        f"{root} or {root / 'figures_input'}"
    )


def _metrics(root: Path, model: str) -> dict:
    path = _seed_root(root) / model / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _risk_curve(uncertainty: np.ndarray, error: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    order = np.argsort(uncertainty)
    cumulative = np.cumsum(error[order])
    counts = np.maximum(1, np.rint(fractions * len(error)).astype(int))
    return cumulative[counts - 1] / counts


def _bootstrap_risk_band(
    uncertainty: np.ndarray,
    error: np.ndarray,
    fractions: np.ndarray,
    *,
    seed: int = 42,
    repeats: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(repeats):
        indices = rng.integers(0, len(error), size=len(error))
        curves.append(_risk_curve(uncertainty[indices], error[indices], fractions))
    samples = np.stack(curves)
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def _graph_distance_counts(num_nodes: int) -> dict[int, int]:
    adjacency = [[] for _ in range(num_nodes)]
    for source, target in ITOP_SKELETON_EDGES:
        adjacency[source].append(target)
        adjacency[target].append(source)
    counts: dict[int, int] = {}
    for source in range(num_nodes):
        distances = [-1] * num_nodes
        distances[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        for target in range(source + 1, num_nodes):
            counts[distances[target]] = counts.get(distances[target], 0) + 1
    return counts


def _bootstrap_distance_intervals(
    residual: np.ndarray,
    distances: list[str],
    *,
    seed: int = 42,
    repeats: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap graph-distance residual correlations over frames."""
    rng = np.random.default_rng(seed)
    num_frames, num_joints = residual.shape[:2]
    adjacency = [[] for _ in range(num_joints)]
    for source, target in ITOP_SKELETON_EDGES:
        adjacency[source].append(target)
        adjacency[target].append(source)
    pairs: dict[int, list[tuple[int, int]]] = {}
    for source in range(num_joints):
        distance = [-1] * num_joints
        distance[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        for target in range(source + 1, num_joints):
            if distance[target] > 0:
                pairs.setdefault(distance[target], []).append((source, target))

    bootstraps = []
    for _ in range(repeats):
        indices = rng.integers(0, num_frames, size=num_frames)
        sample = residual[indices]
        centered = sample - sample.mean(axis=0, keepdims=True)
        energy = np.sum(centered * centered, axis=(0, 2))
        bootstraps.append(
            [
                np.mean(
                    [
                        np.sum(centered[:, source] * centered[:, target])
                        / np.sqrt(
                            max(
                                energy[source] * energy[target],
                                np.finfo(float).eps,
                            )
                        )
                        for source, target in pairs[int(distance)]
                    ]
                )
                for distance in distances
            ]
        )
    values = np.asarray(bootstraps)
    return np.percentile(values, 2.5, axis=0), np.percentile(values, 97.5, axis=0)


def _representative_side_frame(prediction: dict) -> int:
    """Select an IID exemplar without using its error or target geometry."""
    visible = prediction["visible_joints"].sum(dim=1).numpy()
    candidates = np.flatnonzero(visible == visible.max())
    uncertainty = prediction["frame_uncertainty"][candidates].numpy()
    median = float(np.median(uncertainty))
    return int(candidates[np.argmin(np.abs(uncertainty - median))])


def _graph_spd_map(root: Path):
    """Rebuild the exact certified parameter-layout transform saved with the run."""
    seed_root = _seed_root(root)
    candidates = (
        root / "frozen_graph_student_t" / "compilation.json",
        seed_root / "frozen_graph_student_t" / "compilation.json",
    )
    report_path = next((path for path in candidates if path.is_file()), None)
    if report_path is None:
        raise FileNotFoundError(
            "A graph compilation report is required to interpret saved coordinates"
        )
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    plan = plan_readout(
        FeatureSpec.from_irreps(saved["seed"]["irreps"], scope="node"),
        output=ITOP_OUTPUT_GRAPH.output_irreps,
        covariance=GraphPrecision(ITOP_OUTPUT_GRAPH),
        distribution="student_t",
        student_t_dof=float(saved["objective"]["degrees_of_freedom"]),
        output_scope="global",
    )
    rebuilt = plan.report.as_dict()
    saved_family = saved["family"]
    rebuilt_family = rebuilt["family"]
    if (
        saved_family["operator_program_hash"]
        != rebuilt_family["operator_program_hash"]
        or saved_family["parameter_count"] != rebuilt_family["parameter_count"]
        or saved_family["optimization"]["parameter_layout_transform"]
        != rebuilt_family["optimization"]["parameter_layout_transform"]
    ):
        raise ValueError("saved graph coordinates do not match the current certified layout")
    return plan.compilation.build_spd_map()


def _scatter_ellipse(
    center: np.ndarray,
    scatter: torch.Tensor,
    *,
    color: str,
) -> Ellipse:
    """Return the 50% contour for a 2-D Student-t marginal with fixed nu=5."""
    eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
    # q / 2 ~ F(2, 5), hence sqrt(q_0.5) = 1.263938113146554.
    radial = 1.263938113146554
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(1e-12)
    direction = eigenvectors[:, order[0]]
    angle = np.degrees(np.arctan2(float(direction[1]), float(direction[0])))
    return Ellipse(
        center,
        width=2.0 * radial * float(torch.sqrt(eigenvalues[0])),
        height=2.0 * radial * float(torch.sqrt(eigenvalues[1])),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=0.8,
        alpha=0.16,
        zorder=2,
    )


def _plot_pose_distribution(
    axis,
    prediction: dict,
    index: int,
    *,
    mapping,
    color: str,
    title: str,
) -> None:
    mean = prediction["mean"][index].reshape(15, 3).numpy()
    target = prediction["target"][index].reshape(15, 3).numpy()
    visible = prediction["visible_joints"][index].numpy()
    params = prediction["params"][index].unsqueeze(0).float()
    scatter = mapping(params)[0].detach().double()
    precision = mapping.precision(params)[0].detach().double()
    spans = np.ptp(target, axis=0)
    vertical = int(np.argmax(spans))
    horizontal = int(np.argmax(np.where(np.arange(3) == vertical, -np.inf, spans)))
    projection = (horizontal, vertical)
    mean_2d = mean[:, projection]
    target_2d = target[:, projection]
    edge_strength = torch.stack(
        [
            torch.linalg.matrix_norm(
                precision[
                    3 * source : 3 * source + 3,
                    3 * target_node : 3 * target_node + 3,
                ],
                ord="fro",
            )
            for source, target_node in ITOP_SKELETON_EDGES
        ]
    ).numpy()
    lower, upper = np.percentile(edge_strength, (10, 90))
    scale = np.clip((edge_strength - lower) / max(upper - lower, 1e-12), 0, 1)
    for (source, target_node), strength in zip(ITOP_SKELETON_EDGES, scale):
        axis.plot(
            *mean_2d[[source, target_node]].T,
            color=color,
            linewidth=0.8 + 2.2 * strength,
            alpha=0.88,
            zorder=3,
        )
        axis.plot(
            *target_2d[[source, target_node]].T,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
            zorder=1,
        )
    axis.scatter(
        *mean_2d[visible].T,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        s=20,
        zorder=5,
    )
    axis.scatter(
        *mean_2d[~visible].T,
        facecolor="white",
        edgecolor=color,
        linewidth=0.9,
        s=20,
        zorder=5,
    )
    for joint in (0, 6, 7, 8, 13, 14):
        coordinate_indices = [3 * joint + coordinate for coordinate in projection]
        block = scatter[coordinate_indices][:, coordinate_indices]
        axis.add_patch(
            _scatter_ellipse(mean_2d[joint], block, color=color)
        )
    points = np.concatenate((mean_2d, target_2d), axis=0)
    center = points.mean(axis=0)
    radius = max(float(np.ptp(points, axis=0).max()) * 0.58, 0.35)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, loc="left", fontweight="bold", pad=0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(
        0.03,
        0.02,
        f"visible {int(visible.sum())}/15 · camera-plane projection",
        transform=axis.transAxes,
        fontsize=7,
        color=COLORS["dark_gray"],
    )


def _plot_observation_shift(axis, side: dict, top: dict) -> None:
    """Show aggregate visibility and error shifts without a misleading pose overlay."""
    samples = (
        (
            "Side IID",
            COLORS["blue_main"],
            side["visible_joints"].sum(dim=1).numpy(),
            side["joint_errors"].float().mean(dim=-1).numpy() * 100.0,
        ),
        (
            "Top OOD",
            COLORS["red_strong"],
            top["visible_joints"].sum(dim=1).numpy(),
            top["joint_errors"].float().mean(dim=-1).numpy() * 100.0,
        ),
    )
    visibility_axis, error_axis = axis
    rng = np.random.default_rng(42)
    for panel, value_index, ylabel, ylim in (
        (visibility_axis, 2, "Visible joints (of 15)", (0.0, 15.6)),
        (error_axis, 3, "Frame mean joint error (cm)", None),
    ):
        values = [sample[value_index] for sample in samples]
        boxes = panel.boxplot(
            values,
            positions=(0, 1),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": COLORS["dark_gray"], "linewidth": 1.5},
            whiskerprops={"color": COLORS["dark_gray"], "linewidth": 1.0},
            capprops={"color": COLORS["dark_gray"], "linewidth": 1.0},
        )
        for box, (_, color, *_rest) in zip(boxes["boxes"], samples):
            box.set(facecolor=color, alpha=0.26, edgecolor=color, linewidth=1.25)
        for xpos, (_, color, *rest) in enumerate(samples):
            values_for_model = rest[value_index - 2]
            count = min(420, len(values_for_model))
            indices = rng.choice(len(values_for_model), size=count, replace=False)
            jitter = rng.uniform(-0.16, 0.16, size=count)
            panel.scatter(
                np.full(count, xpos) + jitter,
                values_for_model[indices],
                s=5.5,
                color=color,
                alpha=0.12,
                linewidths=0,
                zorder=1,
            )
            median = float(np.median(values_for_model))
            panel.annotate(
                f"median {median:.0f}" if value_index == 2 else f"median {median:.1f}",
                (xpos, median),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=color,
                fontweight="bold",
            )
        panel.set_xlim(-0.55, 1.55)
        panel.set_xticks((0, 1), ("Side", "Top"))
        panel.set_ylabel(ylabel)
        if ylim is not None:
            panel.set_ylim(ylim)
        panel.grid(axis="y", alpha=0.18)
    error_upper = max(float(np.max(sample[3])) for sample in samples)
    error_axis.set_ylim(0.0, 10.0 * np.ceil((error_upper + 1.0) / 10.0))
    visibility_axis.set_title("(b) Observation shift", loc="left", fontweight="bold")


def _marker_area(model: str) -> float:
    return 34 + 2.2 * np.sqrt(ACTIVE_COORDS[model])


def _load_robustness(specifications: list[str]) -> dict[str, dict]:
    records = {}
    for specification in specifications:
        try:
            model, path_text = specification.split("=", 1)
        except ValueError as error:
            raise ValueError(
                f"invalid --robustness specification: {specification}"
            ) from error
        if model not in ROBUSTNESS_MODELS:
            raise ValueError(f"unsupported robustness model: {model}")
        payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
        payload_model = payload.get("model")
        if payload_model is not None and payload_model != model:
            raise ValueError(
                f"robustness file identifies {payload_model}, expected {model}"
            )
        records[ROBUSTNESS_MODELS[model]] = payload["aggregate"]
    return records


def plot_overview(
    root: Path,
    output: Path,
    robustness: dict[str, dict],
) -> None:
    setup_tpami_style()
    seed_root = _seed_root(root)
    models = tuple(
        model for model in MODEL_LABELS if (seed_root / model).is_dir()
    )
    records = {model: _metrics(root, model) for model in models}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=cm2inch(18.2, 7.4),
        gridspec_kw={"width_ratios": (1.36, 0.92)},
    )
    probabilistic_models = tuple(
        model
        for model in models
        if records[model]["side"].get("nll") is not None
        and records[model]["top"].get("nll") is not None
    )
    axis = axes[0]
    for model in probabilistic_models:
        robust = robustness.get(model)
        side_nll = (
            robust["side_nll"]["mean"]
            if robust is not None
            else records[model]["side"]["nll"]
        )
        top_nll = (
            robust["top_nll"]["mean"]
            if robust is not None
            else records[model]["top"]["nll"]
        )
        is_gaussian = "gaussian" in model
        if robust is not None:
            axis.errorbar(
                side_nll,
                top_nll,
                xerr=robust["side_nll"]["std"],
                yerr=robust["top_nll"]["std"],
                fmt="none",
                ecolor=METHOD_COLORS[model],
                elinewidth=1.3,
                capsize=2.5,
                alpha=0.9,
                zorder=2,
            )
        axis.scatter(
            side_nll,
            top_nll,
            s=_marker_area(model),
            marker="s" if is_gaussian else "o",
            color=METHOD_COLORS[model],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        axis.annotate(
            MODEL_LABELS[model],
            (side_nll, top_nll),
            xytext=(5, 4),
            textcoords="offset points",
            color=METHOD_COLORS[model],
            fontsize=7.5,
            fontweight="bold",
        )
    axis.set_yscale("log")
    axis.set_xlim(-88, -13)
    axis.set_ylim(1.8, 1500)
    axis.set_yticks((2, 10, 50, 200, 1000), ("2", "10", "50", "200", "1000"))
    axis.set_xlabel("Side IID proper NLL")
    axis.set_ylabel("Top OOD proper NLL (log scale)")
    axis.set_title("Proper-score trade-off", loc="left", fontweight="bold")
    axis.grid(which="major", alpha=0.65)
    distribution_handles = (
        Line2D(
            [], [], marker="o", color="none", markerfacecolor=COLORS["dark_gray"],
            markeredgecolor="white", markersize=5, label="Student-$t$",
        ),
        Line2D(
            [], [], marker="s", color="none", markerfacecolor=COLORS["dark_gray"],
            markeredgecolor="white", markersize=5, label="Gaussian",
        ),
    )
    axis.legend(
        handles=distribution_handles,
        title="Distribution",
        loc="upper left",
        fontsize=6.5,
        title_fontsize=6.5,
        frameon=True,
        handletextpad=0.4,
        borderpad=0.4,
    )
    distribution_legend = axis.get_legend()
    coordinate_handles = tuple(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="none",
            markerfacecolor=COLORS["neutral"],
            markeredgecolor="white",
            markersize=np.sqrt(34 + 2.2 * np.sqrt(coordinates)),
            label=f"{coordinates:,}",
        )
        for coordinates in (90, 174, 1035)
    )
    axis.add_artist(distribution_legend)
    axis.legend(
        handles=coordinate_handles,
        title="Active coordinates",
        loc="lower right",
        fontsize=6.2,
        title_fontsize=6.2,
        frameon=True,
        handletextpad=0.4,
        borderpad=0.4,
    )

    auroc_models = tuple(
        model
        for model in probabilistic_models
        if np.isfinite(records[model]["ood"].get("side_top_uncertainty_auroc", np.nan))
    )
    y = np.arange(len(auroc_models))[::-1]
    axis = axes[1]
    axis.axvline(
        0.5,
        color=COLORS["dark_gray"],
        linestyle="--",
        linewidth=1.0,
    )
    for model, value, ypos in zip(
        auroc_models,
        (
            robustness[model]["side_top_uncertainty_auroc"]["mean"]
            if model in robustness
            else records[model]["ood"]["side_top_uncertainty_auroc"]
            for model in auroc_models
        ),
        y,
    ):
        color = METHOD_COLORS[model]
        robust = robustness.get(model)
        if robust is not None:
            seed_values = robust["side_top_uncertainty_auroc"]["values"]
            axis.scatter(
                seed_values,
                np.full(len(seed_values), ypos),
                color=color,
                alpha=0.28,
                edgecolor="none",
                s=22,
                zorder=2,
            )
            axis.errorbar(
                value,
                ypos,
                xerr=robust["side_top_uncertainty_auroc"]["std"],
                fmt="none",
                ecolor=color,
                elinewidth=1.4,
                capsize=2.5,
                zorder=2,
            )
        axis.hlines(ypos, 0.5, value, color=color, linewidth=2.1, alpha=0.85, zorder=2)
        axis.scatter(
            value,
            ypos,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=43,
            zorder=3,
        )
        axis.annotate(
            f"{value:.3f}",
            (value, ypos),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.2,
            color=color,
        )
    axis.set_xlim(0, 1.06)
    axis.set_yticks(y, [MODEL_LABELS[model] for model in auroc_models])
    axis.set_xlabel("Side/top uncertainty AUROC")
    axis.set_title("Head-seed OOD sensitivity", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.65)
    axis.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.20, top=0.86, wspace=0.35)
    save_figure(fig, output / "itop_final_overview", formats=("pdf", "png"))
    plt.close(fig)


def plot_structure(root: Path, output: Path) -> None:
    setup_tpami_style()
    model_root = _seed_root(root) / "frozen_graph_student_t"
    metrics = _metrics(root, "frozen_graph_student_t")
    counts = _graph_distance_counts(15)
    distances = sorted(
        metrics["side"]["residual_correlation_by_skeleton_distance"], key=int
    )
    distance_values = [int(distance) for distance in distances]

    side_prediction = torch.load(
        model_root / "predictions_side.pt", map_location="cpu", weights_only=True
    )
    top_prediction = torch.load(
        model_root / "predictions_top.pt", map_location="cpu", weights_only=True
    )
    side_index = _representative_side_frame(side_prediction)
    mapping = _graph_spd_map(root)
    fig = plt.figure(figsize=cm2inch(18.0, 12.3))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.02, 0.88),
        hspace=0.20,
        wspace=0.25,
    )
    pose_axis = fig.add_subplot(grid[0, 0])
    observation_grid = grid[0, 1].subgridspec(1, 2, wspace=0.42)
    observation_axes = (
        fig.add_subplot(observation_grid[0, 0]),
        fig.add_subplot(observation_grid[0, 1]),
    )
    diagnostic_axes = (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))
    _plot_pose_distribution(
        pose_axis,
        side_prediction,
        side_index,
        mapping=mapping,
        color=COLORS["blue_main"],
        title="(a) Side IID predictive pose",
    )
    _plot_observation_shift(observation_axes, side_prediction, top_prediction)
    for view, color, label in (
        ("side", COLORS["blue_main"], "Side IID"),
        ("top", COLORS["red_strong"], "Top OOD"),
    ):
        values = [
            metrics[view]["residual_correlation_by_skeleton_distance"][d]
            for d in distances
        ]
        prediction = torch.load(
            model_root / f"predictions_{view}.pt",
            map_location="cpu",
            weights_only=True,
        )
        residual = (
            prediction["target"] - prediction["mean"]
        ).reshape(-1, 15, 3).numpy()
        lower, upper = _bootstrap_distance_intervals(residual, distances)
        diagnostic_axes[0].errorbar(
            distance_values,
            values,
            yerr=(np.asarray(values) - lower, upper - np.asarray(values)),
            fmt="o-",
            color=color,
            capsize=2.5,
            linewidth=1.8,
            markersize=5,
            label=label,
        )
    diagnostic_axes[0].set_xlabel("Skeleton graph distance")
    diagnostic_axes[0].set_ylabel("Residual correlation")
    diagnostic_axes[0].set_title("(c) Structured residual dependence", loc="left", fontweight="bold")
    diagnostic_axes[0].set_ylim(-0.05, 0.95)
    diagnostic_axes[0].set_xticks(
        distance_values, [f"{d}\n(n={counts[d]})" for d in distance_values]
    )
    fractions = np.linspace(0.1, 1.0, 10)
    for view, color, label in (
        ("side", COLORS["blue_main"], "Side IID"),
        ("top", COLORS["red_strong"], "Top OOD"),
    ):
        payload = torch.load(model_root / f"predictions_{view}.pt", map_location="cpu", weights_only=True)
        uncertainty = payload["frame_uncertainty"].numpy()
        error = payload["joint_errors"].float().mean(dim=-1).numpy() * 100.0
        curve = _risk_curve(uncertainty, error, fractions)
        lower, upper = _bootstrap_risk_band(uncertainty, error, fractions)
        diagnostic_axes[1].plot(
            fractions * 100,
            curve,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=4.5,
            label=label,
        )
        diagnostic_axes[1].fill_between(
            fractions * 100,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    diagnostic_axes[1].set_xlabel("Retained coverage (%)")
    diagnostic_axes[1].set_ylabel("Mean joint error (cm)")
    diagnostic_axes[1].set_title("(d) Selective risk (95% bootstrap)", loc="left", fontweight="bold")
    diagnostic_axes[1].set_ylim(
        min(0.0, float(np.nanmin(lower)) - 1.0),
        float(np.nanmax(upper)) + 1.0,
    )
    diagnostic_axes[1].legend(loc="lower left", fontsize=6.8, handlelength=2.0)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.92)
    save_figure(fig, output / "itop_final_structure", formats=("pdf", "png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--robustness",
        action="append",
        default=[],
        metavar="MODEL=JSON",
        help=(
            "optional summarize_itop_robustness output; MODEL is one of "
            "full_student_t, low_rank_student_t, graph_student_t"
        ),
    )
    args = parser.parse_args()
    robustness = _load_robustness(args.robustness)
    plot_overview(args.results, args.output, robustness)
    plot_structure(args.results, args.output)
    print(f"ITOP final figures written to {args.output}")


if __name__ == "__main__":
    main()
