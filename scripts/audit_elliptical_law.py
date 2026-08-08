"""Run E0 single-elliptical-law falsification from released artifacts."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from data.dielectric_dataset import get_dielectric_irreps_loaders
from evaluation.elliptical import (
    elliptical_falsification_from_whitened,
    falsification_decision,
    stratified_elliptical_falsification,
)
from evaluation.metrics import symmetric_whitened_residuals
from scripts.dielectric_runtime import (
    collect_dielectric_predictions,
    configure_inference_contract,
    inference_contract_from_args,
    load_dielectric_checkpoint,
    load_dielectric_data_args,
    load_run_record,
)
from scripts.itop_reproducibility import (
    atomic_write_json,
    sha256_file,
    source_provenance,
)
from scripts.train_itop import _build_model

HYPOTHESIS = (
    "A single fixed-nu elliptical Student-t is contradicted by whitened residual "
    "projection, direction, or radius-direction structure."
)


def _json_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required provenance artifact is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "record": json.loads(path.read_text(encoding="utf-8")),
    }


def _tertiles(values: torch.Tensor, name: str) -> dict[str, torch.Tensor]:
    values = values.float()
    lower, upper = torch.quantile(values, torch.tensor([1.0 / 3.0, 2.0 / 3.0]))
    return {
        f"{name}_low": values <= lower,
        f"{name}_mid": (values > lower) & (values <= upper),
        f"{name}_high": values > upper,
    }


def _audit_whitened(
    whitened: torch.Tensor,
    strata: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    kwargs = {
        "reference": "student_t",
        "student_t_dof": args.student_t_dof,
        "num_directions": args.directions,
        "permutations": args.permutations,
        "seed": args.seed,
    }
    overall = elliptical_falsification_from_whitened(whitened, **kwargs)
    stratified = stratified_elliptical_falsification(whitened, strata, **kwargs)
    return {
        "overall": overall,
        "decision": falsification_decision(overall, alpha=args.alpha),
        "strata": {
            name: {
                **audit,
                **(
                    {"decision": falsification_decision(audit, alpha=args.alpha)}
                    if audit.get("status") != "insufficient_samples"
                    else {}
                ),
            }
            for name, audit in stratified.items()
        },
    }


def _dielectric_descriptors(dataset) -> dict[str, torch.Tensor]:
    atom_count, element_count, spatial_extent = [], [], []
    for index in range(len(dataset)):
        graph = dataset[index]
        positions = graph.pos.float()
        centered = positions - positions.mean(dim=0, keepdim=True)
        atom_count.append(float(positions.shape[0]))
        element_count.append(float(torch.unique(graph.z).numel()))
        spatial_extent.append(
            float(torch.sqrt(centered.square().sum(dim=-1).mean()).item())
        )
    return {
        "atom_count": torch.tensor(atom_count),
        "element_count": torch.tensor(element_count),
        "spatial_extent": torch.tensor(spatial_extent),
    }


def _audit_dielectric(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = args.checkpoint_dir.resolve()
    model, spec, _ = load_dielectric_checkpoint(checkpoint_dir, args.device)
    data_args = load_dielectric_data_args(checkpoint_dir)
    contract = load_run_record(checkpoint_dir).get("inference_contract")
    if contract is None:
        contract = inference_contract_from_args(data_args, args.device)
    configure_inference_contract(contract)
    _, _, loader = get_dielectric_irreps_loaders(
        data_dir=data_args.data_dir,
        batch_size=data_args.batch_size,
        num_workers=getattr(data_args, "num_workers", 0),
        persistent_workers=getattr(data_args, "persistent_workers", False),
        pin_memory=getattr(data_args, "pin_memory", False),
        prefetch_factor=getattr(data_args, "prefetch_factor", None),
        lmax=data_args.lmax,
        storage=getattr(data_args, "dataset_storage", "files"),
        shard_cache_size=getattr(data_args, "shard_cache_size", 2),
    )
    predictions = collect_dielectric_predictions(
        model, loader, args.device, inference_contract=contract
    )
    whitened = symmetric_whitened_residuals(
        predictions["mu_irreps"],
        predictions["y_irreps"],
        predictions["scale_irreps"],
    )
    descriptors = _dielectric_descriptors(loader.dataset)
    if any(len(values) != whitened.shape[0] for values in descriptors.values()):
        raise RuntimeError("dielectric descriptor and prediction counts differ")
    strata: dict[str, torch.Tensor] = {}
    for name, values in descriptors.items():
        strata.update(_tertiles(values, name))
    args.student_t_dof = float(spec.student_t_dof)
    return {
        "schema_version": 1,
        "study": "E0 dielectric single-elliptical-law falsification",
        "hypothesis": HYPOTHESIS,
        "existing_evidence": (
            "Unified checkpoint has poor radial coverage and second-moment "
            "whitening, but lacked projection, pure-direction, and independence tests."
        ),
        "intervention": "artifact-only symmetric-whitened residual diagnostics",
        "controlled_variables": {
            "checkpoint": str(checkpoint_dir),
            "runner_source": source_provenance(Path(__file__).resolve().parents[1]),
            "checkpoint_sha256": sha256_file(checkpoint_dir / "best_model.pt"),
            "split": "test",
            "coordinate_space": "compiled_irreps_0e_plus_2e",
            "student_t_dof": args.student_t_dof,
            "inference_contract": contract,
        },
        "descriptors": {name: values.tolist() for name, values in descriptors.items()},
        "audit": _audit_whitened(whitened, strata, args),
    }


def _itop_whitened(
    run_dir: Path, view: str, chunk_size: int
) -> tuple[torch.Tensor, dict]:
    training_args = Namespace(
        **json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    )
    model, _ = _build_model(training_args)
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    prediction = torch.load(
        run_dir / f"predictions_{view}.pt", map_location="cpu", weights_only=True
    )
    expected_samples = 4863
    required = ("mean", "target", "params", "visible_joints")
    missing = [key for key in required if key not in prediction]
    if missing:
        raise KeyError(f"{view} predictions lack required fields: {missing}")
    counts = {key: int(prediction[key].shape[0]) for key in required}
    if set(counts.values()) != {expected_samples}:
        raise RuntimeError(
            f"{view} predictions violate the {expected_samples}-sample contract: {counts}"
        )
    if not all(
        bool(torch.isfinite(prediction[key]).all())
        for key in ("mean", "target", "params")
    ):
        raise ValueError(f"{view} predictions contain non-finite values")
    records = []
    with torch.inference_mode():
        for start in range(0, prediction["mean"].shape[0], chunk_size):
            stop = start + chunk_size
            scale = model.spd_map(prediction["params"][start:stop].float()).double()
            records.append(
                symmetric_whitened_residuals(
                    prediction["mean"][start:stop].double(),
                    prediction["target"][start:stop].double(),
                    scale,
                )
            )
    whitened = torch.cat(records)
    if whitened.shape != prediction["mean"].shape or not bool(
        torch.isfinite(whitened).all()
    ):
        raise RuntimeError(f"{view} whitening produced invalid output")
    return whitened, prediction


def _audit_itop(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    training_args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    args.student_t_dof = float(training_args["student_t_dof"])
    views = {}
    for view in ("side", "top"):
        whitened, prediction = _itop_whitened(run_dir, view, args.chunk_size)
        visible_count = prediction["visible_joints"].sum(dim=-1).float()
        strata = {
            "severely_occluded_0_to_5_visible": visible_count <= 5,
            "partially_visible_6_to_10": (visible_count >= 6) & (visible_count <= 10),
            "mostly_visible_11_to_14": (visible_count >= 11) & (visible_count <= 14),
            "fully_visible_15": visible_count == 15,
        }
        views[view] = {
            "visible_joint_count_summary": {
                "min": int(visible_count.min()),
                "median": float(visible_count.median()),
                "max": int(visible_count.max()),
            },
            **_audit_whitened(whitened, strata, args),
        }
    return {
        "schema_version": 1,
        "study": "E0 ITOP single-elliptical-law falsification",
        "hypothesis": HYPOTHESIS,
        "existing_evidence": (
            "Side/Top coverage and uncertainty ranking are poor or unstable, but "
            "saved results lacked projection, pure-direction, and independence tests."
        ),
        "intervention": "artifact-only symmetric-whitened residual diagnostics",
        "controlled_variables": {
            "run_dir": str(run_dir),
            "runner_source": source_provenance(Path(__file__).resolve().parents[1]),
            "checkpoint_sha256": sha256_file(run_dir / "best_model.pt"),
            "prediction_sha256": {
                view: sha256_file(run_dir / f"predictions_{view}.pt")
                for view in ("side", "top")
            },
            "training_args": _json_artifact(run_dir / "args.json"),
            "training_environment": _json_artifact(run_dir / "environment.json"),
            "model": training_args["model"],
            "student_t_dof": args.student_t_dof,
            "sample_contract": {"side": 4863, "top": 4863},
        },
        "views": views,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directions", type=int, default=64)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--alpha", type=float, default=0.01)
    subparsers = parser.add_subparsers(dest="task", required=True)
    dielectric = subparsers.add_parser("dielectric")
    dielectric.add_argument("--checkpoint_dir", type=Path, required=True)
    dielectric.add_argument("--device", default="cuda")
    itop = subparsers.add_parser("itop")
    itop.add_argument("--run_dir", type=Path, required=True)
    itop.add_argument("--chunk_size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    record = _audit_dielectric(args) if args.task == "dielectric" else _audit_itop(args)
    atomic_write_json(record, args.output)
    print(json.dumps({"study": record["study"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
