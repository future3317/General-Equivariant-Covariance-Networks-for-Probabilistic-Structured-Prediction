"""Freeze one dielectric Full-t checkpoint into the task-neutral E1 schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from compatibility.torch_geometric import PyGDataLoader
from data.dielectric_dataset import DielectricIrrepsDataset
from representations.operator_lowering import project_parameter_bindings
from scripts.dielectric_runtime import (
    configure_inference_contract,
    dataset_provenance,
    load_dielectric_checkpoint,
    load_dielectric_data_args,
    load_run_record,
    sha256_file,
)
from scripts.itop_reproducibility import atomic_write_json, source_provenance


def _save_tensor(payload: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def _extract(model, loader, device: torch.device, contract: dict[str, Any]) -> dict:
    records: dict[str, list[torch.Tensor]] = {
        "features": [],
        "mean": [],
        "params": [],
        "target": [],
        "sample_id": [],
    }
    use_bf16 = contract["backbone_precision"] == "bf16" and device.type == "cuda"
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            node_features, graph_batch = model.backbone(batch)
        compiled = model.joint_head._compiled_features(
            node_features.float(), graph_batch
        )
        mean = model.joint_head.mean_projection(compiled)
        params = project_parameter_bindings(model.joint_head, compiled)
        records["features"].append(compiled.float().cpu())
        records["mean"].append(mean.float().cpu())
        records["params"].append(params.float().cpu())
        records["target"].append(batch.y_irreps.float().cpu())
        records["sample_id"].append(batch.sample_id.long().reshape(-1).cpu())
    return {name: torch.cat(values) for name, values in records.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E1 cache: {args.output_dir}")
    device = torch.device(args.device)
    model, spec, compilation = load_dielectric_checkpoint(args.checkpoint_dir, device)
    full_parameter_count = (
        compilation.output_spec.dim * (compilation.output_spec.dim + 1) // 2
    )
    if (
        spec.distribution != "student_t"
        or compilation.covariance_parameter_count != full_parameter_count
        or len(compilation.operator_family.parameter_bindings) != 1
    ):
        raise ValueError(
            "E1 dielectric cache requires the released Full Student-t model"
        )
    run_record = load_run_record(args.checkpoint_dir)
    contract = run_record.get("inference_contract")
    if not isinstance(contract, dict):
        raise TypeError("source checkpoint lacks an inference contract")
    configure_inference_contract(contract)
    data_args = load_dielectric_data_args(args.checkpoint_dir)
    datasets = {
        split: DielectricIrrepsDataset(
            data_args.data_dir,
            split,
            lmax=data_args.lmax,
            storage=getattr(data_args, "dataset_storage", "files"),
            shard_cache_size=getattr(data_args, "shard_cache_size", 2),
        )
        for split in ("train", "val", "test")
    }
    args.output_dir.mkdir(parents=True)
    projection_path = args.output_dir / "operator_projection.pt"
    _save_tensor(model.joint_head.covariance_projection.state_dict(), projection_path)
    split_records = {}
    for split, dataset in datasets.items():
        loader = PyGDataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        payload = _extract(model, loader, device, contract)
        path = args.output_dir / f"{split}.pt"
        _save_tensor(payload, path)
        split_records[split] = {
            "count": len(dataset),
            "sha256": sha256_file(path),
            "sample_id_sha256": hashlib.sha256(
                payload["sample_id"].contiguous().numpy().tobytes()
            ).hexdigest(),
            "sample_id_min": int(payload["sample_id"].min()),
            "sample_id_max": int(payload["sample_id"].max()),
        }
    metadata = {
        "schema_version": 1,
        "kind": "frozen_distribution_features",
        "task_adapter": "dielectric_full",
        "source_checkpoint": {
            "path": str(args.checkpoint_dir.resolve()),
            "sha256": sha256_file(args.checkpoint_dir / "best_model.pt"),
        },
        "source_run_spec_sha256": sha256_file(args.checkpoint_dir / "run_spec.json"),
        "source_compilation": compilation.as_dict(),
        "feature_irreps": str(compilation.active_target_irreps),
        "output_irreps": str(compilation.output_spec.irreps),
        "parameter_irreps": str(compilation.operator_family.parameter_irreps),
        "parameter_count": compilation.covariance_parameter_count,
        "operator_projection": {
            "path": projection_path.name,
            "sha256": sha256_file(projection_path),
        },
        "operator_family": compilation.operator_family.as_dict(),
        "spd_map": {
            "kind": spec.covariance_parameterization,
            "shape_min": spec.shape_min,
            "shape_max": spec.shape_max,
            "volume_min": spec.volume_min,
            "volume_max": spec.volume_max,
            "log_variance_min": spec.log_variance_min,
            "log_variance_max": spec.log_variance_max,
            "representation_metric": spec.representation_metric,
            "metric_scalar": spec.metric_scalar,
            "metric_l2": spec.metric_l2,
        },
        "student_t_dof": spec.student_t_dof,
        "inference_contract": contract,
        "dataset": dataset_provenance(data_args.data_dir),
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
