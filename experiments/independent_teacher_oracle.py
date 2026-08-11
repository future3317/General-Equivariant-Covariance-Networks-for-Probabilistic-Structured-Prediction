"""Independent NumPy/SciPy data-generating oracle for scatter recovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg, stats

SUPPORTED_FAMILIES = (
    "full",
    "low_rank",
    "isotypic_block",
    "graph_precision",
)


@dataclass(frozen=True)
class OracleProtocol:
    """Sampling counts and Student-t law for an oracle dataset."""

    train_contexts: int = 128
    train_replicates: int = 32
    validation_contexts: int = 64
    validation_replicates: int = 64
    test_contexts: int = 128
    test_replicates: int = 128
    calibration_draws: int = 65_536
    calibration_trials: int = 2_048
    nu: float = 5.0

    def __post_init__(self) -> None:
        counts = (
            self.train_contexts,
            self.train_replicates,
            self.validation_contexts,
            self.validation_replicates,
            self.test_contexts,
            self.test_replicates,
            self.calibration_draws,
            self.calibration_trials,
        )
        if any(value < 1 for value in counts):
            raise ValueError("oracle protocol counts must be positive")
        if self.nu <= 2.0:
            raise ValueError("nu must be greater than 2")


@dataclass(frozen=True)
class OracleDataset:
    """In-memory output of the independent data-generating process."""

    family: str
    seed: int
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def rank2_cartesian_basis() -> np.ndarray:
    """Return the explicit orthonormal ``0e + 2e`` Cartesian basis."""
    basis = np.zeros((6, 3, 3), dtype=np.float64)
    basis[0] = np.eye(3, dtype=np.float64) / math.sqrt(3.0)
    basis[1, 0, 2] = basis[1, 2, 0] = 1.0 / math.sqrt(2.0)
    basis[2, 0, 1] = basis[2, 1, 0] = 1.0 / math.sqrt(2.0)
    basis[3] = np.diag([-1.0, 2.0, -1.0]) / math.sqrt(6.0)
    basis[4, 1, 2] = basis[4, 2, 1] = 1.0 / math.sqrt(2.0)
    basis[5] = np.diag([-1.0, 0.0, 1.0]) / math.sqrt(2.0)
    return basis


def _matrix_exponential(matrices: np.ndarray) -> np.ndarray:
    return np.stack([linalg.expm(matrix) for matrix in matrices])


def _rank2_parts(contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if contexts.ndim != 2 or contexts.shape[1] != 6:
        raise ValueError("rank-2 oracle contexts must have shape (n, 6)")
    scalar = contexts[:, 0]
    coefficients = contexts[:, 1:]
    stf_basis = rank2_cartesian_basis()[1:]
    tensor = np.einsum("na,aij->nij", coefficients, stf_basis)
    return scalar, coefficients, tensor


def _stf_coordinates(tensor: np.ndarray) -> np.ndarray:
    return np.einsum("nij,aij->na", tensor, rank2_cartesian_basis()[1:])


def _full_scatter(contexts: np.ndarray) -> np.ndarray:
    scalar, coefficients, tensor = _rank2_parts(contexts)
    norm = np.linalg.norm(coefficients, axis=-1)
    normalized = coefficients / (1.0 + norm[:, None])
    stf_basis = rank2_cartesian_basis()[1:]
    products = (
        np.einsum("nij,ajk->naik", tensor, stf_basis)
        + np.einsum("aij,njk->naik", stf_basis, tensor)
    )
    stf_action = np.einsum("bij,naij->nab", stf_basis, products)
    stf_action = 0.5 * (stf_action + stf_action.transpose(0, 2, 1))

    log_scatter = np.zeros((contexts.shape[0], 6, 6), dtype=np.float64)
    log_scatter[:, 0, 0] = -0.35 + 0.12 * np.tanh(scalar)
    coupling = 0.11 * normalized
    log_scatter[:, 0, 1:] = coupling
    log_scatter[:, 1:, 0] = coupling
    block_scale = -0.08 + 0.08 * np.tanh(0.4 * scalar + 0.2 * norm**2)
    log_scatter[:, 1:, 1:] = (
        block_scale[:, None, None] * np.eye(5)[None]
        + 0.055 * stf_action
        + 0.045 * np.einsum("ni,nj->nij", normalized, normalized)
    )
    return _matrix_exponential(log_scatter)


def _low_rank_scatter(contexts: np.ndarray) -> np.ndarray:
    scalar, coefficients, tensor = _rank2_parts(contexts)
    norm2 = np.einsum("ni,ni->n", coefficients, coefficients)
    normalized = coefficients / (1.0 + np.sqrt(norm2)[:, None])
    tensor_square = tensor @ tensor
    tensor_square -= (
        np.trace(tensor_square, axis1=-2, axis2=-1)[:, None, None]
        * np.eye(3)[None]
        / 3.0
    )
    square_coordinates = _stf_coordinates(tensor_square)
    square_coordinates /= 1.0 + np.linalg.norm(square_coordinates, axis=-1)[:, None]

    first = np.concatenate(
        [0.18 * np.tanh(scalar)[:, None], 0.14 * normalized], axis=-1
    )
    second = np.concatenate(
        [
            (0.13 * np.tanh(norm2 - 1.0))[:, None],
            0.12 * square_coordinates,
        ],
        axis=-1,
    )
    factor = np.stack([first, second], axis=-1)
    sigma2 = np.exp(-0.28 + 0.10 * np.tanh(scalar) + 0.025 * norm2)
    return sigma2[:, None, None] * np.eye(6)[None] + factor @ factor.transpose(0, 2, 1)


def _isotypic_block_scatter(contexts: np.ndarray) -> np.ndarray:
    scalar, coefficients, _ = _rank2_parts(contexts)
    norm2 = np.einsum("ni,ni->n", coefficients, coefficients)
    scalar_scale = np.exp(-0.30 + 0.18 * np.tanh(scalar))
    stf_scale = np.exp(0.05 + 0.12 * np.tanh(0.3 * norm2 - 0.5))
    scatter = np.zeros((contexts.shape[0], 6, 6), dtype=np.float64)
    scatter[:, 0, 0] = scalar_scale
    scatter[:, 1:, 1:] = stf_scale[:, None, None] * np.eye(5)[None]
    return scatter


def _local_log_precision(vectors: np.ndarray, offset: float, shape: float) -> np.ndarray:
    norm2 = np.einsum("ni,ni->n", vectors, vectors)
    normalized_outer = np.einsum("ni,nj->nij", vectors, vectors) / (
        1.0 + norm2[:, None, None]
    )
    scalar = offset + 0.08 * np.tanh(norm2 - 1.0)
    return scalar[:, None, None] * np.eye(3)[None] + shape * normalized_outer


def _graph_precision_scatter(contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if contexts.ndim != 2 or contexts.shape[1] != 9:
        raise ValueError("graph oracle contexts must have shape (n, 9)")
    nodes = contexts.reshape(-1, 3, 3)
    unary = np.stack(
        [
            _matrix_exponential(_local_log_precision(nodes[:, node], 0.20, 0.10))
            for node in range(3)
        ],
        axis=1,
    )
    edges = ((0, 1), (1, 2))
    relational = np.stack(
        [
            _matrix_exponential(
                _local_log_precision(nodes[:, target] - nodes[:, source], -0.15, 0.07)
            )
            for source, target in edges
        ],
        axis=1,
    )
    precision = np.zeros((contexts.shape[0], 9, 9), dtype=np.float64)
    for node in range(3):
        node_slice = slice(3 * node, 3 * (node + 1))
        precision[:, node_slice, node_slice] += unary[:, node]
    for edge_index, (source, target) in enumerate(edges):
        source_slice = slice(3 * source, 3 * (source + 1))
        target_slice = slice(3 * target, 3 * (target + 1))
        block = relational[:, edge_index]
        precision[:, source_slice, source_slice] += block
        precision[:, target_slice, target_slice] += block
        precision[:, source_slice, target_slice] -= block
        precision[:, target_slice, source_slice] -= block
    scatter = np.stack(
        [linalg.cho_solve(linalg.cho_factor(matrix, lower=True), np.eye(9)) for matrix in precision]
    )
    return scatter, precision


def construct_distribution(
    family: str, contexts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Construct zero mean and the declared independent operator family."""
    contexts = np.asarray(contexts, dtype=np.float64)
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported oracle family: {family}")
    if family == "graph_precision":
        scatter, precision = _graph_precision_scatter(contexts)
    else:
        constructors = {
            "full": _full_scatter,
            "low_rank": _low_rank_scatter,
            "isotypic_block": _isotypic_block_scatter,
        }
        scatter = constructors[family](contexts)
        precision = None
    mean = np.zeros((contexts.shape[0], scatter.shape[-1]), dtype=np.float64)
    return mean, scatter, precision


def _sample_student_t(
    mean: np.ndarray,
    scatter: np.ndarray,
    replicates: int,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    cholesky = np.stack([linalg.cholesky(matrix, lower=True) for matrix in scatter])
    normal = rng.standard_normal((mean.shape[0], replicates, mean.shape[1]))
    chi_square = rng.chisquare(nu, size=(mean.shape[0], replicates))
    residual = np.einsum("nij,nrj->nri", cholesky, normal)
    residual *= np.sqrt(nu / chi_square)[:, :, None]
    return mean[:, None, :] + residual


def _coverage(
    observations: np.ndarray,
    mean: np.ndarray,
    scatter: np.ndarray,
    nu: float,
    level: float,
) -> float:
    residual = observations - mean[:, None, :]
    solved = np.stack(
        [
            linalg.cho_solve(linalg.cho_factor(matrix, lower=True), values.T).T
            for matrix, values in zip(scatter, residual)
        ]
    )
    q = np.einsum("nri,nri->nr", residual, solved)
    dimension = mean.shape[-1]
    threshold = dimension * stats.f.ppf(level, dimension, nu)
    return float(np.mean(q <= threshold))


def _equivariance_self_check(family: str, contexts: np.ndarray) -> float:
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if family == "graph_precision":
        representation = np.kron(np.eye(3), rotation)
    else:
        basis = rank2_cartesian_basis()
        rotated_basis = np.einsum("ij,ajk,lk->ail", rotation, basis, rotation)
        representation = np.einsum("aij,bij->ab", rotated_basis, basis)
    mean, scatter, precision = construct_distribution(family, contexts)
    rotated_mean, rotated_scatter, rotated_precision = construct_distribution(
        family, contexts @ representation.T
    )
    errors = [
        np.linalg.norm(rotated_mean - mean @ representation.T)
        / max(np.linalg.norm(mean), 1.0),
        np.linalg.norm(
            rotated_scatter - representation @ scatter @ representation.T
        )
        / np.linalg.norm(scatter),
    ]
    if precision is not None and rotated_precision is not None:
        errors.append(
            np.linalg.norm(
                rotated_precision - representation @ precision @ representation.T
            )
            / np.linalg.norm(precision)
        )
    return float(max(errors))


def build_oracle_dataset(
    family: str, seed: int, protocol: OracleProtocol
) -> OracleDataset:
    """Generate deterministic repeated Student-t observations for one family."""
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported oracle family: {family}")
    streams = np.random.SeedSequence(seed).spawn(4)
    generators = [np.random.default_rng(stream) for stream in streams]
    input_dim = 9 if family == "graph_precision" else 6
    split_specs = (
        ("train", protocol.train_contexts, protocol.train_replicates),
        ("validation", protocol.validation_contexts, protocol.validation_replicates),
        ("test", protocol.test_contexts, protocol.test_replicates),
    )
    arrays: dict[str, np.ndarray] = {}
    for split_index, (split, contexts_count, replicates) in enumerate(split_specs):
        rng = generators[split_index]
        contexts = 0.65 * rng.standard_normal((contexts_count, input_dim))
        mean, scatter, precision = construct_distribution(family, contexts)
        arrays[f"{split}_context_ids"] = (
            split_index * 1_000_000 + np.arange(contexts_count, dtype=np.int64)
        )
        arrays[f"{split}_inputs"] = contexts
        arrays[f"{split}_mean"] = mean
        arrays[f"{split}_scatter"] = scatter
        arrays[f"{split}_observations"] = _sample_student_t(
            mean, scatter, replicates, protocol.nu, rng
        )
        if precision is not None:
            arrays[f"{split}_precision"] = precision

    calibration_rng = generators[3]
    test_mean = arrays["test_mean"]
    test_scatter = arrays["test_scatter"]
    calibration_replicates = math.ceil(protocol.calibration_draws / len(test_mean))
    calibration = _sample_student_t(
        test_mean,
        test_scatter,
        calibration_replicates,
        protocol.nu,
        calibration_rng,
    ).reshape(-1, test_mean.shape[-1])[: protocol.calibration_draws]
    calibration_context = np.arange(protocol.calibration_draws) % len(test_mean)
    calibration = calibration.reshape(protocol.calibration_draws, 1, -1)
    calibration_mean = test_mean[calibration_context]
    calibration_scatter = test_scatter[calibration_context]
    coverage90 = _coverage(
        calibration, calibration_mean, calibration_scatter, protocol.nu, 0.90
    )
    coverage95 = _coverage(
        calibration, calibration_mean, calibration_scatter, protocol.nu, 0.95
    )
    metadata: dict[str, Any] = {
        "oracle_version": "independent_numpy_scipy_v1",
        "family": family,
        "seed": int(seed),
        "nu": float(protocol.nu),
        "teacher_coverage": {"coverage_90": coverage90, "coverage_95": coverage95},
        "self_checks": {
            "minimum_scatter_eigenvalue": float(
                np.linalg.eigvalsh(arrays["test_scatter"]).min()
            ),
            "equivariance_relative_max": _equivariance_self_check(
                family, arrays["test_inputs"][: min(4, protocol.test_contexts)]
            ),
        },
    }
    return OracleDataset(family=family, seed=int(seed), arrays=arrays, metadata=metadata)
