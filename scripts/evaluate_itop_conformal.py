"""Post-hoc ITOP conformal diagnostic from saved graph precision predictions.

The official ITOP test files are split here only for a diagnostic: half of the
side-view test predictions calibrate a region, the other half measures IID
coverage, and the complete top-view test measures cross-view transfer.  This
does not change the official benchmark metrics or provide an OOD guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from compatibility.e3nn import o3
from data.itop_dataset import ITOP_OUTPUT_GRAPH
from evaluation.conformal import evaluate_region, fit_split_conformal
from representations.irrep_layout import RepeatedIrrepLayout
from representations.symmetric_square import O3SymmetricOperatorBasis
from spd_maps.graph_precision import GraphStructuredPrecisionMap


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_prediction(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"mean", "target", "params", "frame_index", "view_id"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"prediction artifact is missing keys: {sorted(required)}")
    return payload


@torch.inference_mode()
def _materialize(path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    payload = _load_prediction(path)
    params = payload["params"].float()
    local_irreps = o3.Irreps([(1, ITOP_OUTPUT_GRAPH.node_irrep)])
    basis_module = O3SymmetricOperatorBasis(local_irreps)
    local_count = basis_module.basis.shape[0]
    expected_flat = ITOP_OUTPUT_GRAPH.num_potentials * local_count
    if params.ndim != 2 or params.shape[-1] != expected_flat:
        raise ValueError(
            f"expected {expected_flat} graph-local coefficients, got {params.shape}"
        )
    layout = RepeatedIrrepLayout(
        basis_module.operator_irreps, ITOP_OUTPUT_GRAPH.num_potentials
    )
    coefficients = layout.pack(params)
    basis = basis_module.basis.to(dtype=params.dtype, device=params.device)
    raw_blocks = torch.einsum("...nq,qij->...nij", coefficients, basis)
    shape = GraphStructuredPrecisionMap(ITOP_OUTPUT_GRAPH).forward(raw_blocks)
    shape = 0.5 * (shape + shape.transpose(-1, -2))
    return payload, shape


def _split_indices(
    frame_index: torch.Tensor,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if frame_index.ndim != 1 or frame_index.numel() < 2:
        raise ValueError("frame_index must contain at least two examples")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0,1)")
    if frame_index.unique().numel() != frame_index.numel():
        raise ValueError("frame_index must be unique before post-hoc splitting")
    count = int(frame_index.numel() * calibration_fraction)
    if count < 1 or count >= frame_index.numel():
        raise ValueError("calibration split must leave at least one evaluation example")
    permutation = torch.randperm(
        frame_index.numel(), generator=torch.Generator().manual_seed(seed)
    )
    return permutation[:count], permutation[count:]


def _validate_view_pair(
    side: dict[str, torch.Tensor], top: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Check that the two saved predictions describe the same test frames."""
    for name, payload, expected_view in (
        ("side", side, 0),
        ("top", top, 1),
    ):
        frame_index = payload["frame_index"]
        view_id = payload["view_id"]
        if frame_index.ndim != 1 or view_id.ndim != 1:
            raise ValueError(f"{name} frame_index/view_id must be one-dimensional")
        if frame_index.numel() != view_id.numel():
            raise ValueError(f"{name} frame_index/view_id lengths disagree")
        if frame_index.unique().numel() != frame_index.numel():
            raise ValueError(f"{name} frame_index must be unique")
        observed_views = view_id.unique().tolist()
        if observed_views != [expected_view]:
            raise ValueError(
                f"{name} predictions must have view_id={expected_view}, "
                f"got {observed_views}"
            )
    if not torch.equal(side["frame_index"], top["frame_index"]):
        raise ValueError("side and top predictions must use the same frame order")
    return {
        "side_view_id": int(side["view_id"][0]),
        "top_view_id": int(top["view_id"][0]),
        "paired_frame_count": int(side["frame_index"].numel()),
        "frame_order_equal": True,
    }


def run_diagnostic(
    model_dir: Path,
    *,
    alpha: float = 0.1,
    calibration_fraction: float = 0.5,
    seed: int = 42,
) -> dict[str, object]:
    side_path = model_dir / "predictions_side.pt"
    top_path = model_dir / "predictions_top.pt"
    side, side_shape = _materialize(side_path)
    top, top_shape = _materialize(top_path)
    view_contract = _validate_view_pair(side, top)
    if side["mean"].shape[-1] != 45 or top["mean"].shape[-1] != 45:
        raise ValueError("ITOP predictions must have 45 output coordinates")
    calibration, side_eval = _split_indices(
        side["frame_index"],
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    region = fit_split_conformal(
        side["mean"][calibration],
        side_shape[calibration],
        side["target"][calibration],
        alpha=alpha,
    )
    calibration_summary = evaluate_region(
        region,
        side["mean"][calibration],
        side_shape[calibration],
        side["target"][calibration],
    )
    side_summary = evaluate_region(
        region,
        side["mean"][side_eval],
        side_shape[side_eval],
        side["target"][side_eval],
    )
    top_summary = evaluate_region(
        region, top["mean"], top_shape, top["target"]
    )
    return {
        "study": "ITOP post-hoc split-conformal shape diagnostic",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch_version": torch.__version__,
        "model_dir": str(model_dir),
        "model_dir_files": {
            "predictions_side.pt": _sha256(side_path),
            "predictions_top.pt": _sha256(top_path),
        },
        "output_representation": "V=15(1o), d=45",
        "shape_semantics": "reconstructed graph-precision scatter S=Q^{-1}",
        "view_contract": view_contract,
        "alpha": alpha,
        "calibration_fraction": calibration_fraction,
        "seed": seed,
        "calibration_size": int(calibration.numel()),
        "side_evaluation_size": int(side_eval.numel()),
        "top_evaluation_size": int(top["frame_index"].numel()),
        "region": {
            "rank": region.rank,
            "threshold": region.threshold,
        },
        "calibration": calibration_summary,
        "side_iid_holdout": side_summary,
        "top_cross_view_transfer": top_summary,
        "official_benchmark_warning": (
            "This consumes part of the official side test file for post-hoc "
            "calibration. It is not a replacement for the untouched ITOP "
            "benchmark and gives no conformal guarantee under cross-view shift."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_diagnostic(
        args.model_dir,
        alpha=args.alpha,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
