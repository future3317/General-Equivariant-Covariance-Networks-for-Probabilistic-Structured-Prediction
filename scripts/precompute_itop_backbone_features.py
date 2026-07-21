"""Cache pooled features from one deterministic ITOP backbone checkpoint."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
from tqdm import tqdm

from data.itop_dataset import get_itop_split_loader
from data.itop_features import ITOPFeatureDataset, sha256_file
from models.pooling import mean_pool
from scripts.train_itop import build_itop_backbone


@torch.inference_mode()
def _extract(backbone, loader, device: torch.device, *, use_bf16: bool) -> dict:
    records: dict[str, list[torch.Tensor]] = {
        "features": [],
        "target": [],
        "visible_joints": [],
        "frame_index": [],
        "view_id": [],
    }
    backbone.eval()
    for batch in tqdm(loader, desc="cache frozen backbone"):
        batch = batch.to(device, non_blocking=True)
        enabled = use_bf16 and device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=enabled,
        ):
            node_features, graph_batch = backbone(batch)
        # Match training exactly: the BF16 backbone output crosses the typed
        # readout boundary in FP32 before graph pooling.
        records["features"].append(mean_pool(node_features.float(), graph_batch).cpu())
        records["target"].append(batch.y_pose.cpu())
        records["visible_joints"].append(batch.visible_joints.cpu())
        records["frame_index"].append(batch.frame_index.cpu())
        records["view_id"].append(batch.view_id.cpu())
    return {name: torch.cat(values) for name, values in records.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_dir")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    checkpoint_hash = sha256_file(args.checkpoint)
    metadata_path = args.output_dir / "metadata.json"
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise NotADirectoryError(args.output_dir)
        if not metadata_path.is_file():
            raise FileExistsError(
                f"feature cache directory is incomplete: {args.output_dir}"
            )
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing["backbone_checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("output directory belongs to a different checkpoint")
        for name in ("side_train", "side_test", "top_test"):
            ITOPFeatureDataset(args.output_dir / f"{name}.pt")
        print(f"complete feature cache already exists: {args.output_dir}")
        return
    args.output_dir.mkdir(parents=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_kind"] != "deterministic":
        raise ValueError("backbone feature cache requires a deterministic checkpoint")
    training_args = Namespace(**checkpoint["args"])
    if args.data_dir is not None:
        training_args.data_dir = args.data_dir
    device = torch.device(args.device)
    backbone = build_itop_backbone(training_args)
    backbone.load_state_dict(checkpoint["backbone_state"], strict=True)
    backbone = backbone.to(device)
    loader_kwargs = dict(
        data_dir=training_args.data_dir,
        batch_size=args.batch_size,
        num_points=training_args.num_points,
        num_neighbors=training_args.num_neighbors,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
    )
    splits = {
        "side_train": ("side", "train"),
        "side_test": ("side", "test"),
        "top_test": ("top", "test"),
    }
    counts = {}
    for name, (view, split) in splits.items():
        output_path = args.output_dir / f"{name}.pt"
        loader = get_itop_split_loader(view=view, split=split, **loader_kwargs)
        payload = _extract(
            backbone,
            loader,
            device,
            use_bf16=training_args.backbone_precision == "bf16",
        )
        counts[name] = int(payload["features"].shape[0])
        temporary = output_path.with_suffix(".pt.partial")
        torch.save(payload, temporary)
        temporary.replace(output_path)
    metadata = {
        "schema_version": 1,
        "kind": "frozen_pooled_itop_backbone_features",
        "backbone_checkpoint": str(args.checkpoint.resolve()),
        "backbone_checkpoint_sha256": checkpoint_hash,
        "backbone_irreps": str(backbone.irreps_out),
        "feature_dimension": backbone.irreps_out.dim,
        "backbone_precision": training_args.backbone_precision,
        "num_points": training_args.num_points,
        "num_neighbors": training_args.num_neighbors,
        "counts": counts,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
