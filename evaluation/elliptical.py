"""Falsification diagnostics for single-component elliptical laws."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from scipy import stats

from evaluation.metrics import symmetric_whitened_residuals


def _uniformity_summary(values: np.ndarray) -> dict[str, Any]:
    """Summarize per-column PIT uniformity without pooling dependent columns."""
    if values.ndim != 2:
        raise ValueError("PIT values must have shape (samples, directions)")
    tests = [
        stats.kstest(values[:, index], "uniform") for index in range(values.shape[1])
    ]
    statistics = np.asarray([test.statistic for test in tests], dtype=np.float64)
    pvalues = np.asarray([test.pvalue for test in tests], dtype=np.float64)
    alpha = 0.05 / values.shape[1]
    histogram = np.histogram(values, bins=np.linspace(0.0, 1.0, 11))[0]
    expected = values.size / 10.0
    return {
        "median_ks": float(np.median(statistics)),
        "max_ks": float(statistics.max()),
        "min_pvalue": float(pvalues.min()),
        "bonferroni_alpha": float(alpha),
        "bonferroni_rejections": int((pvalues < alpha).sum()),
        "pooled_decile_l1_error": float(
            np.abs(histogram - expected).sum() / values.size
        ),
    }


def _standardized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(values, axis=0)
    ranks -= ranks.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(ranks, axis=0, keepdims=True)
    return ranks / np.maximum(norms, np.finfo(np.float64).tiny)


def _radius_direction_dependence(
    radial_pit: np.ndarray,
    angular_projections: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Test radial dependence on signed and axial direction coordinates."""
    radial_rank = _standardized_ranks(radial_pit[:, None])[:, 0]
    signed_rank = _standardized_ranks(angular_projections)
    axial_rank = _standardized_ranks(np.square(angular_projections))
    direction_ranks = np.concatenate((signed_rank, axial_rank), axis=1)
    correlations = radial_rank @ direction_ranks
    observed = float(np.abs(correlations).max())
    generator = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = generator.permutation(radial_rank)
        null_maximum = float(np.abs(permuted @ direction_ranks).max())
        exceedances += null_maximum >= observed
    return {
        "max_abs_spearman": observed,
        "median_abs_spearman": float(np.median(np.abs(correlations))),
        "signed_max_abs_spearman": float(
            np.abs(correlations[: angular_projections.shape[1]]).max()
        ),
        "axial_max_abs_spearman": float(
            np.abs(correlations[angular_projections.shape[1] :]).max()
        ),
        "max_statistic_permutation_pvalue": float(
            (exceedances + 1) / (permutations + 1)
        ),
        "permutations": int(permutations),
    }


def elliptical_falsification_from_whitened(
    whitened: torch.Tensor,
    *,
    reference: str,
    student_t_dof: float | None = None,
    num_directions: int = 64,
    permutations: int = 199,
    seed: int = 0,
) -> dict[str, Any]:
    """Audit radial, projection, angular, and radius-direction implications.

    ``whitened`` contains principal-symmetric-whitened residuals. Under a
    correctly specified single elliptical component, its direction is uniform
    on the sphere and independent of its radius. Gaussian projections are
    standard normal; Student-t projections are univariate Student-t with the
    declared degrees of freedom.
    """
    if whitened.ndim != 2 or whitened.shape[0] < 20 or whitened.shape[1] < 2:
        raise ValueError("whitened residuals must have shape (N,d), N>=20, d>=2")
    if reference not in {"gaussian", "student_t"}:
        raise ValueError("reference must be gaussian or student_t")
    if reference == "student_t" and (student_t_dof is None or student_t_dof <= 0):
        raise ValueError("Student-t diagnostics require positive degrees of freedom")
    if num_directions < 4 or permutations < 19:
        raise ValueError("use at least four directions and 19 permutations")
    if not bool(torch.isfinite(whitened).all()):
        raise ValueError("whitened residuals must be finite")

    z = whitened.detach().double().cpu()
    radius2 = z.square().sum(dim=-1)
    nonzero = radius2 > torch.finfo(z.dtype).tiny
    if int(nonzero.sum()) < 20:
        raise ValueError("too few nonzero residuals for directional diagnostics")
    z = z[nonzero]
    radius2 = radius2[nonzero]
    sample_count, dimension = z.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(
        num_directions, dimension, generator=generator, dtype=z.dtype
    )
    directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    projections = (z @ directions.T).numpy()

    if reference == "gaussian":
        radial_pit = stats.chi2.cdf(radius2.numpy(), df=dimension)
        projection_pit = stats.norm.cdf(projections)
        expected_second_moment = 1.0
    else:
        nu = float(student_t_dof)
        radial_pit = stats.f.cdf(radius2.numpy() / dimension, dimension, nu)
        projection_pit = stats.t.cdf(projections, df=nu)
        expected_second_moment = nu / (nu - 2.0) if nu > 2.0 else None

    radial_test = stats.kstest(radial_pit, "uniform")
    direction = z / radius2.sqrt().unsqueeze(-1)
    angular_projections = (direction @ directions.T).numpy()
    beta_shape = 0.5 * (dimension - 1)
    spherical_axis_pit = stats.beta.cdf(
        np.clip(0.5 * (angular_projections + 1.0), 0.0, 1.0),
        beta_shape,
        beta_shape,
    )
    angular_moment = direction.T @ direction / sample_count
    angular_identity = torch.eye(dimension, dtype=direction.dtype) / dimension

    record: dict[str, Any] = {
        "sample_count": int(sample_count),
        "dimension": int(dimension),
        "reference": reference,
        "student_t_dof": None if reference == "gaussian" else float(student_t_dof),
        "seed": int(seed),
        "radial_pit": {
            "mean": float(np.mean(radial_pit)),
            "std": float(np.std(radial_pit, ddof=1)),
            "q10": float(np.quantile(radial_pit, 0.10)),
            "q50": float(np.quantile(radial_pit, 0.50)),
            "q90": float(np.quantile(radial_pit, 0.90)),
            "ks": float(radial_test.statistic),
            "pvalue": float(radial_test.pvalue),
        },
        "projection_pit": {
            "directions": int(num_directions),
            **_uniformity_summary(projection_pit),
        },
        "direction_sphericality": {
            "mean_resultant_norm": float(
                torch.linalg.vector_norm(direction.mean(dim=0)).item()
            ),
            "second_moment_defect": float(
                torch.linalg.matrix_norm(angular_moment - angular_identity).item()
                * dimension
            ),
            "axis_pit": _uniformity_summary(spherical_axis_pit),
        },
        "radius_direction_dependence": _radius_direction_dependence(
            radial_pit,
            angular_projections,
            permutations=permutations,
            seed=seed + 1,
        ),
    }
    if expected_second_moment is not None:
        second_moment = z.T @ z / sample_count
        record["whitened_second_moment_defect"] = float(
            torch.linalg.matrix_norm(
                second_moment / expected_second_moment
                - torch.eye(dimension, dtype=z.dtype)
            ).item()
        )
    return record


def elliptical_falsification(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    **kwargs: Any,
) -> dict[str, Any]:
    """Whiten predictions with the existing invariant primitive and audit them."""
    whitened = symmetric_whitened_residuals(pred, target, scale)
    return elliptical_falsification_from_whitened(whitened, **kwargs)


def stratified_elliptical_falsification(
    whitened: torch.Tensor,
    strata: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Run the same declared audit on non-overlapping or overlapping strata."""
    records: dict[str, dict[str, Any]] = {}
    for name, mask in strata.items():
        if (
            mask.dtype != torch.bool
            or mask.ndim != 1
            or mask.shape[0] != whitened.shape[0]
        ):
            raise ValueError(f"invalid stratum mask: {name}")
        count = int(mask.sum())
        if count < 20:
            records[name] = {"sample_count": count, "status": "insufficient_samples"}
            continue
        records[name] = elliptical_falsification_from_whitened(whitened[mask], **kwargs)
    return records


def falsification_decision(
    record: Mapping[str, Any], *, alpha: float = 0.01
) -> dict[str, Any]:
    """Translate calibrated tests into distinct radial and elliptical decisions."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    radial_rejected = record["radial_pit"]["pvalue"] < alpha
    projection_rejected = record["projection_pit"]["bonferroni_rejections"] > 0
    spherical_rejected = (
        record["direction_sphericality"]["axis_pit"]["bonferroni_rejections"] > 0
    )
    independence_rejected = (
        record["radius_direction_dependence"]["max_statistic_permutation_pvalue"]
        < alpha
    )
    return {
        "alpha": float(alpha),
        "fixed_radial_law_rejected": bool(radial_rejected),
        "projection_law_rejected": bool(projection_rejected),
        "spherical_direction_rejected": bool(spherical_rejected),
        "radius_direction_independence_rejected": bool(independence_rejected),
        "single_component_elliptical_structure_rejected": bool(
            spherical_rejected or independence_rejected
        ),
        "note": (
            "Projection rejection alone can reflect radial or directional mismatch; "
            "elliptical structure is rejected only by direction sphericality or "
            "radius-direction dependence."
        ),
    }
