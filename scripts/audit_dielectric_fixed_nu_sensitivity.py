"""Post-hoc fixed-nu sensitivity for saved dielectric predictions.

The scan evaluates the production Student-t log-probability on saved
per-sample sufficient statistics. It never fits nu on the test set and does
not retrain or alter a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from distributions.student_t import student_t_log_prob_from_statistics
from evaluation.metrics import mahalanobis_distance_squared


def validate_nu_grid(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Validate the finite-covariance Student-t sensitivity grid."""

    grid = tuple(float(value) for value in values)
    if not grid or any(value <= 2.0 for value in grid):
        raise ValueError("nu grid must contain positive finite-covariance values nu > 2")
    if any(not torch.isfinite(torch.tensor(value)) for value in grid):
        raise ValueError("nu grid must be finite")
    return grid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def student_t_nll_from_prediction(prediction: dict[str, Any], nu: float) -> float:
    """Evaluate exact mean NLL from saved frame-level sufficient statistics."""

    validate_nu_grid([nu])
    if "scale" in prediction:
        mean = torch.as_tensor(prediction["mean"], dtype=torch.float64)
        target = torch.as_tensor(prediction["target"], dtype=torch.float64)
        scale = torch.as_tensor(prediction["scale"], dtype=torch.float64)
        sign, logdet = torch.linalg.slogdet(scale)
        if bool((sign <= 0).any()):
            raise ValueError("saved dielectric scale is not SPD")
        mahalanobis2 = mahalanobis_distance_squared(target - mean, scale)
        dimension = int(target.shape[-1])
    else:
        logdet = torch.as_tensor(prediction["frame_uncertainty"], dtype=torch.float64)
        mahalanobis2 = torch.as_tensor(prediction["frame_mahalanobis2"], dtype=torch.float64)
        dimension = int(torch.as_tensor(prediction["target"]).shape[-1])
    log_prob = student_t_log_prob_from_statistics(logdet, mahalanobis2, dimension, nu)
    if not bool(torch.isfinite(log_prob).all()):
        raise ValueError("saved prediction produced non-finite Student-t NLL")
    return float((-log_prob).mean().item())


def audit_prediction(path: Path, grid: tuple[float, ...]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"prediction artifact is not a dictionary: {path}")
    values = {str(nu): student_t_nll_from_prediction(payload, nu) for nu in grid}
    return {
        "prediction_path": str(path),
        "prediction_sha256": _sha256(path),
        "sample_count": int(torch.as_tensor(payload["target"]).shape[0]),
        "dimension": int(torch.as_tensor(payload["target"]).shape[-1]),
        "selection": "none; post-hoc sensitivity only",
        "nll_semantics": "production_student_t_log_prob_from_statistics",
        "nll_by_nu": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nu", type=float, nargs="+", default=[3.0, 5.0, 10.0, 30.0])
    args = parser.parse_args()
    grid = validate_nu_grid(args.nu)
    result = {
        "schema_version": 1,
        "kind": "dielectric_fixed_nu_posthoc_sensitivity",
        "nu_grid": list(grid),
        "records": [audit_prediction(Path(path), grid) for path in args.prediction],
        "evidence_status": {
            "existing_evidence": ["saved Full-t prediction sufficient statistics"],
            "new_evidence": ["post-hoc fixed-nu NLL sensitivity"],
            "supported_inference": ["the selected fixed-nu=5 density can be compared with nearby fixed laws"],
            "unresolved": ["no validation-selected or end-to-end learned global nu is claimed"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
