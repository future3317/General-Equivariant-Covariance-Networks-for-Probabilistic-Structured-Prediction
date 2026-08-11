"""Pytest bridge for the process-isolated NumPy/SciPy oracle checks."""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path


def test_oracle_source_has_no_repository_or_torch_imports():
    source = Path("experiments/independent_teacher_oracle.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = {
        "torch",
        "e3nn",
        "equivcompiler",
        "spd_maps",
        "distributions",
        "evaluation",
    }
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert roots.isdisjoint(forbidden)


def test_independent_oracle_contract_suite():
    result = subprocess.run(
        [sys.executable, "tests/independent_teacher_oracle_checks.py", "-v"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rank2_basis_matches_public_runtime_coordinates():
    oracle_process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; from experiments.independent_teacher_oracle import "
                "rank2_cartesian_basis; "
                "print(json.dumps(rank2_cartesian_basis().tolist()))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    repository_process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; torch.set_default_dtype(torch.float64); "
                "from compatibility.e3nn import CartesianTensor; "
                "c=CartesianTensor('ij=ji'); "
                "eye=torch.eye(6,dtype=torch.float64); "
                "print(json.dumps([c.to_cartesian(eye[i]).tolist() "
                "for i in range(6)]))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    oracle_basis = json.loads(oracle_process.stdout)
    repository_basis = json.loads(repository_process.stdout)
    for oracle_matrix, repository_matrix in zip(oracle_basis, repository_basis):
        for oracle_row, repository_row in zip(oracle_matrix, repository_matrix):
            for oracle_value, repository_value in zip(oracle_row, repository_row):
                assert math.isclose(
                    oracle_value,
                    repository_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
