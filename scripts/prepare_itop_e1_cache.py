"""Adapt a released ITOP Full-t head to the task-neutral frozen E1 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from data.itop_dataset import itop_train_validation_indices
from scripts.itop_reproducibility import (
    atomic_write_json,
    sha256_file,
    source_provenance,
)
from scripts.train_itop import _build_model


def _save_tensor(payload: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sample_ids(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    frame = payload["frame_index"].long()
    view = payload["view_id"].long()
    identifiers = view * (1 << 32) + frame
    if torch.unique(identifiers).numel() != identifiers.numel():
        raise ValueError("ITOP frame/view identifiers are not unique")
    return identifiers


@torch.inference_mode()
def _extract(
    model,
    payload: dict[str, torch.Tensor],
    indices: list[int] | None,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if indices is None:
        selected = torch.arange(payload["features"].shape[0])
    else:
        selected = torch.tensor(indices, dtype=torch.long)
    records = {name: [] for name in ("features", "mean", "params", "target")}
    sample_ids = _sample_ids(payload).index_select(0, selected)
    model.eval()
    for start in range(0, selected.numel(), batch_size):
        batch_indices = selected[start : start + batch_size]
        features = payload["features"].index_select(0, batch_indices).to(device)
        graph_batch = torch.arange(features.shape[0], device=device)
        mean = model.joint_head.mean_head(features, graph_batch)
        params = model.joint_head.operator_head.forward_parameters(
            features, graph_batch
        )
        records["features"].append(features.float().cpu())
        records["mean"].append(mean.float().cpu())
        records["params"].append(params.float().cpu())
        records["target"].append(
            payload["target"].index_select(0, batch_indices).float()
        )
    return {
        **{name: torch.cat(values) for name, values in records.items()},
        "sample_id": sample_ids,
    }


def _split_record(payload: dict[str, torch.Tensor], path: Path) -> dict:
    identifiers = payload["sample_id"].contiguous().numpy().tobytes()
    return {
        "count": int(payload["sample_id"].numel()),
        "sha256": sha256_file(path),
        "sample_id_sha256": hashlib.sha256(identifiers).hexdigest(),
        "sample_id_min": int(payload["sample_id"].min()),
        "sample_id_max": int(payload["sample_id"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--feature_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E1 cache: {args.output_dir}")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")

    checkpoint_path = args.checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_kind") != "full_student_t":
        raise ValueError("ITOP E1 requires a released Full Student-t checkpoint")
    source_args = argparse.Namespace(**checkpoint["args"])
    model, plan = _build_model(source_args)
    if plan is None:
        raise RuntimeError("Full Student-t checkpoint did not reconstruct a plan")
    compilation = plan.compilation
    output_dimension = compilation.output_spec.dim
    if (
        compilation.covariance_parameter_count
        != output_dimension * (output_dimension + 1) // 2
    ):
        raise ValueError("ITOP E1 requires all Full covariance coordinates")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device)

    feature_metadata_path = args.feature_cache / "metadata.json"
    feature_metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    expected_backbone = feature_metadata["backbone_checkpoint_sha256"]
    if expected_backbone != sha256_file(source_args.backbone_checkpoint):
        raise ValueError("ITOP feature cache and Full-t backbone hashes differ")
    side_train_path = args.feature_cache / "side_train.pt"
    side_test_path = args.feature_cache / "side_test.pt"
    top_test_path = args.feature_cache / "top_test.pt"
    side_train = torch.load(side_train_path, map_location="cpu", weights_only=True)
    side_test = torch.load(side_test_path, map_location="cpu", weights_only=True)
    top_test = torch.load(top_test_path, map_location="cpu", weights_only=True)
    train_indices, validation_indices = itop_train_validation_indices(
        side_train["features"].shape[0], seed=int(checkpoint["seed"])
    )
    sources = {
        "train": (side_train, train_indices),
        "val": (side_train, validation_indices),
        "test": (side_test, None),
        "ood": (top_test, None),
    }

    args.output_dir.mkdir(parents=True)
    split_records = {}
    for split, (payload, indices) in sources.items():
        frozen = _extract(
            model,
            payload,
            indices,
            device=device,
            batch_size=args.batch_size,
        )
        path = args.output_dir / f"{split}.pt"
        _save_tensor(frozen, path)
        split_records[split] = _split_record(frozen, path)

    metadata = {
        "schema_version": 1,
        "kind": "frozen_distribution_features",
        "task_adapter": "itop_full",
        "source_checkpoint": {
            "path": str(args.checkpoint_dir.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "source_compilation": compilation.as_dict(),
        "feature_irreps": str(compilation.seed_irreps),
        "output_irreps": str(compilation.output_spec.irreps),
        "parameter_irreps": str(compilation.operator_family.parameter_irreps),
        "parameter_count": compilation.covariance_parameter_count,
        "operator_family": compilation.operator_family.as_dict(),
        "spd_map": {"kind": "matrix_exp", "representation_metric": "none"},
        "student_t_dof": float(source_args.student_t_dof),
        "inference_contract": {
            "backbone_precision": feature_metadata["backbone_precision"],
            "operator_precision": "fp32",
            "frozen_pooled_features": True,
        },
        "dataset": {
            "feature_metadata_sha256": sha256_file(feature_metadata_path),
            "side_train_sha256": sha256_file(side_train_path),
            "side_test_sha256": sha256_file(side_test_path),
            "top_test_sha256": sha256_file(top_test_path),
        },
        "selection_contract": {
            "validation": "Side held-out split",
            "test": "Side IID",
            "ood": "Top cross-view; never used for selection",
        },
        "splits": split_records,
        "source": source_provenance(Path(__file__).resolve().parents[1]),
    }
    atomic_write_json(metadata, args.output_dir / "metadata.json")
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "splits": split_records}, indent=2
        )
    )


if __name__ == "__main__":
    main()
