"""Process-isolated checks for the NumPy/SciPy teacher oracle."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.independent_teacher_oracle import (
    OracleDataset,
    OracleProtocol,
    build_oracle_dataset,
    construct_distribution,
    rank2_cartesian_basis,
)

SMOKE_PROTOCOL = OracleProtocol(
    train_contexts=4,
    train_replicates=3,
    validation_contexts=3,
    validation_replicates=3,
    test_contexts=4,
    test_replicates=5,
    calibration_draws=256,
    calibration_trials=32,
)


def _rank2_representation(rotation: np.ndarray) -> np.ndarray:
    basis = rank2_cartesian_basis()
    rotated_basis = np.einsum("ij,ajk,lk->ail", rotation, basis, rotation)
    return np.einsum("aij,bij->ab", rotated_basis, basis)


class IndependentTeacherOracleChecks(unittest.TestCase):
    def test_protocol_requires_finite_student_t_covariance(self):
        with self.assertRaisesRegex(ValueError, "nu must be greater than 2"):
            OracleProtocol(nu=2.0)

    def test_rank2_basis_is_the_explicit_orthonormal_contract(self):
        basis = rank2_cartesian_basis()
        expected = np.zeros((6, 3, 3), dtype=np.float64)
        expected[0] = np.eye(3) / math.sqrt(3.0)
        expected[1, 0, 2] = expected[1, 2, 0] = 1.0 / math.sqrt(2.0)
        expected[2, 0, 1] = expected[2, 1, 0] = 1.0 / math.sqrt(2.0)
        expected[3] = np.diag([-1.0, 2.0, -1.0]) / math.sqrt(6.0)
        expected[4, 1, 2] = expected[4, 2, 1] = 1.0 / math.sqrt(2.0)
        expected[5] = np.diag([-1.0, 0.0, 1.0]) / math.sqrt(2.0)
        np.testing.assert_allclose(basis, expected, atol=1e-15)
        np.testing.assert_allclose(
            np.einsum("aij,bij->ab", basis, basis),
            np.eye(6),
            atol=1e-15,
        )

    def test_datasets_are_finite_spd_and_reproducible(self):
        cases = (
            ("full", 6, 6),
            ("low_rank", 6, 6),
            ("isotypic_block", 6, 6),
            ("graph_precision", 9, 9),
        )
        for family, input_dim, output_dim in cases:
            with self.subTest(family=family):
                first = build_oracle_dataset(family, 3, SMOKE_PROTOCOL)
                second = build_oracle_dataset(family, 3, SMOKE_PROTOCOL)
                self.assertIsInstance(first, OracleDataset)
                self.assertEqual(first.arrays["test_inputs"].shape, (4, input_dim))
                self.assertEqual(
                    first.arrays["test_scatter"].shape,
                    (4, output_dim, output_dim),
                )
                self.assertEqual(
                    first.arrays["test_observations"].shape,
                    (4, 5, output_dim),
                )
                self.assertTrue(np.isfinite(first.arrays["test_observations"]).all())
                self.assertGreater(
                    float(np.linalg.eigvalsh(first.arrays["test_scatter"]).min()),
                    1e-10,
                )
                for key in first.arrays:
                    np.testing.assert_array_equal(
                        first.arrays[key],
                        second.arrays[key],
                    )

    def test_families_have_the_declared_matrix_structure(self):
        full = build_oracle_dataset("full", 5, SMOKE_PROTOCOL).arrays["test_scatter"]
        low_rank = build_oracle_dataset("low_rank", 5, SMOKE_PROTOCOL).arrays[
            "test_scatter"
        ]
        block = build_oracle_dataset("isotypic_block", 5, SMOKE_PROTOCOL).arrays[
            "test_scatter"
        ]
        graph = build_oracle_dataset("graph_precision", 5, SMOKE_PROTOCOL).arrays[
            "test_precision"
        ]
        self.assertGreater(float(np.abs(full[:, 0, 1:]).max()), 1e-4)
        eigenvalues = np.linalg.eigvalsh(low_rank)
        np.testing.assert_allclose(
            eigenvalues[:, :4],
            np.repeat(eigenvalues[:, :1], 4, axis=1),
            atol=1e-10,
        )
        np.testing.assert_allclose(block[:, 0, 1:], 0.0, atol=1e-14)
        np.testing.assert_allclose(
            block[:, 1:, 1:],
            np.eye(5)[None] * block[:, 1, 1, None, None],
            atol=1e-14,
        )
        np.testing.assert_allclose(graph[:, :3, 6:9], 0.0, atol=1e-14)

    def test_rank2_distributions_are_equivariant(self):
        contexts = np.array(
            [[0.25, 0.2, -0.1, 0.3, 0.15, -0.2]],
            dtype=np.float64,
        )
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rho = _rank2_representation(rotation)
        for family in ("full", "low_rank", "isotypic_block"):
            with self.subTest(family=family):
                mean, scatter, _ = construct_distribution(family, contexts)
                rotated_mean, rotated_scatter, _ = construct_distribution(
                    family,
                    contexts @ rho.T,
                )
                np.testing.assert_allclose(
                    rotated_mean,
                    mean @ rho.T,
                    atol=5e-12,
                )
                np.testing.assert_allclose(
                    rotated_scatter,
                    rho @ scatter @ rho.T,
                    atol=5e-12,
                )

    def test_graph_distribution_is_equivariant(self):
        contexts = np.arange(9, dtype=np.float64).reshape(1, 9) / 10.0
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rho = np.kron(np.eye(3), rotation)
        mean, scatter, precision = construct_distribution(
            "graph_precision",
            contexts,
        )
        rotated_mean, rotated_scatter, rotated_precision = construct_distribution(
            "graph_precision",
            contexts @ rho.T,
        )
        np.testing.assert_allclose(rotated_mean, mean @ rho.T, atol=5e-12)
        np.testing.assert_allclose(
            rotated_scatter,
            rho @ scatter @ rho.T,
            atol=5e-12,
        )
        np.testing.assert_allclose(
            rotated_precision,
            rho @ precision @ rho.T,
            atol=5e-12,
        )

    def test_teacher_radial_coverage_matches_student_t_law(self):
        protocol = OracleProtocol(
            train_contexts=2,
            train_replicates=2,
            validation_contexts=2,
            validation_replicates=2,
            test_contexts=8,
            test_replicates=8,
            calibration_draws=8_192,
            calibration_trials=32,
        )
        for family in ("full", "low_rank", "isotypic_block", "graph_precision"):
            with self.subTest(family=family):
                coverage = build_oracle_dataset(family, 13, protocol).metadata[
                    "teacher_coverage"
                ]
                self.assertLess(abs(coverage["coverage_90"] - 0.90), 0.025)
                self.assertLess(abs(coverage["coverage_95"] - 0.95), 0.025)


if __name__ == "__main__":
    unittest.main()
