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


def test_oracle_cli_writes_artifact_without_loading_torch(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.independent_teacher_oracle",
            "--family",
            "full",
            "--seed",
            "2",
            "--output-dir",
            str(tmp_path),
            "--source-commit",
            "test",
            "--train-contexts",
            "2",
            "--train-replicates",
            "2",
            "--validation-contexts",
            "2",
            "--validation-replicates",
            "2",
            "--test-contexts",
            "2",
            "--test-replicates",
            "2",
            "--calibration-draws",
            "64",
            "--calibration-trials",
            "16",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(result.stdout)
    assert Path(artifact["npz_path"]).is_file()
    assert Path(artifact["manifest_path"]).is_file()


def test_production_student_t_nll_matches_scipy_reference():
    reference_code = (
        "import json,math,numpy as np; from scipy.special import gammaln; "
        "r=np.array([[0.25,-0.4]],dtype=np.float64); "
        "s=np.array([[[1.2,0.15],[0.15,0.8]]],dtype=np.float64); "
        "nu=5.; d=2; q=float(r[0]@np.linalg.solve(s[0],r[0])); "
        "v=-gammaln((nu+d)/2)+gammaln(nu/2)+0.5*d*math.log(nu*math.pi)"
        "+0.5*np.linalg.slogdet(s[0])[1]+0.5*(nu+d)*math.log1p(q/nu); "
        "print(json.dumps(float(v)))"
    )
    production_code = (
        "import json,torch; from distributions import StudentTNLL; "
        "from spd_maps import MatrixExponentialMap; torch.set_default_dtype(torch.float64); "
        "mu=torch.zeros(1,2); y=torch.tensor([[0.25,-0.4]]); "
        "s=torch.tensor([[[1.2,0.15],[0.15,0.8]]]); "
        "e,u=torch.linalg.eigh(s); a=u@torch.diag_embed(torch.log(e))@u.transpose(-1,-2); "
        "v,_=StudentTNLL(5.)(mu,a,y,MatrixExponentialMap()); "
        "print(json.dumps(float(v)))"
    )
    reference = subprocess.run(
        [sys.executable, "-c", reference_code],
        check=True,
        capture_output=True,
        text=True,
    )
    production = subprocess.run(
        [sys.executable, "-c", production_code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert math.isclose(
        json.loads(reference.stdout),
        json.loads(production.stdout),
        rel_tol=0.0,
        abs_tol=1e-10,
    )
