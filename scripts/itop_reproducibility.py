"""Small, explicit provenance helpers for the controlled ITOP protocol.

The training code records two different things on purpose:

* the scientific contract (what was requested), and
* the provenance (which source, caches, and input checkpoints supplied it).

Keeping these records separate makes a result auditable without turning the
training entry point into a general experiment-management framework.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from data.itop_dataset import itop_cache_dir

CONTRACT_VERSION = 1
GEOMETRY_CACHE_FILES = (
    "metadata.json",
    "points.npy",
    "neighbors.npy",
    "joints.npy",
    "visible_joints.npy",
    "centroids.npy",
    "frame_indices.npy",
)
FEATURE_CACHE_FILES = ("metadata.json", "side_train.pt", "side_test.pt", "top_test.pt")


def atomic_write_json(payload: Any, path: str | Path) -> None:
    """Publish JSON without leaving a truncated final file after interruption."""
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_hash(contract: dict[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_provenance(repo_root: str | Path) -> dict[str, Any]:
    """Capture source identity, including an explicit dirty-tree indicator."""
    root = str(Path(repo_root).resolve())

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", root, *arguments], text=True
        ).strip()

    commit = git("rev-parse", "HEAD")
    status = subprocess.check_output(
        ["git", "-C", root, "status", "--porcelain"], text=True
    )
    tracked = git("ls-files", "-s")
    diff = subprocess.check_output(
        ["git", "-C", root, "diff", "--binary", "HEAD"], text=False
    )
    untracked = git("ls-files", "--others", "--exclude-standard")
    source_payload = {
        "commit": commit,
        "tracked_index": tracked,
        "status": status,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked": untracked,
    }
    source_hash = hashlib.sha256(
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "commit": commit,
        "dirty": bool(status.strip() or untracked.strip()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "diff_sha256": source_payload["diff_sha256"],
        "untracked_files": untracked.splitlines() if untracked else [],
        "source_hash": source_hash,
    }


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required ITOP provenance file is missing: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def geometry_cache_provenance(
    data_dir: str | Path,
    *,
    num_points: int,
    num_neighbors: int,
    train_cache_sample_limit: int | None,
) -> dict[str, Any]:
    """Hash every immutable geometry-cache artifact used by ITOP training."""
    root = Path(data_dir)
    records: dict[str, Any] = {}
    for view, split, sample_limit in (
        ("side", "train", train_cache_sample_limit),
        ("side", "test", None),
        ("top", "test", None),
    ):
        cache = itop_cache_dir(
            root, view, split, num_points, num_neighbors, sample_limit
        )
        metadata_path = cache / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"ITOP geometry cache is missing: {cache}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_limit = sample_limit
        if metadata.get("sample_limit") != expected_limit:
            raise ValueError(
                f"ITOP cache sample_limit mismatch for {cache}: "
                f"{metadata.get('sample_limit')} != {expected_limit}"
            )
        records[f"{view}_{split}"] = {
            "path": str(cache.resolve()),
            "sample_limit": sample_limit,
            "metadata": metadata,
            "files": {
                name: _file_record(cache / name) for name in GEOMETRY_CACHE_FILES
            },
        }
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "caches": records,
        "dataset_cache_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def feature_cache_provenance(cache_dir: str | Path) -> dict[str, Any]:
    root = Path(cache_dir)
    files = {name: _file_record(root / name) for name in FEATURE_CACHE_FILES}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(root.resolve()),
        "files": files,
        "feature_cache_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def checkpoint_provenance(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    checkpoint = Path(path)
    return {
        "path": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
    }


def training_contract(
    args, device: torch.device, *, freeze: dict[str, Any]
) -> dict[str, Any]:
    """Return the fixed ITOP training semantics for one stage."""
    split_seed = getattr(args, "split_seed", None)
    if split_seed is None:
        split_seed = args.seed
    return {
        "version": CONTRACT_VERSION,
        "model": {
            "model_kind": args.model,
            "phase": args.phase,
            "student_t_dof": args.student_t_dof,
            "representation_metric": args.representation_metric,
        },
        "representation": {
            "output": "15 x 1o centered 3D joints",
            "num_joints": 15,
            "coordinate_dim": 3,
        },
        "data": {
            "train_view": "side",
            "validation": "fixed-seed 90/10 split of side-train",
            "split_seed": split_seed,
            "test_views": ["side", "top"],
            "num_points": args.num_points,
            "num_neighbors": args.num_neighbors,
            "train_cache_sample_limit": args.train_cache_sample_limit,
            "label_centering": False,
            "augmentation": False,
            "split_mixing": False,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2)",
            "early_stopping_patience": args.patience,
            "selection": (
                "validation_mpjpe_cm"
                if args.model == "deterministic"
                else "validation_nll"
            ),
            "training_objective": (
                "mse_mean_plus_detached_feature_residual_nll"
                if getattr(args, "faithful_joint", False)
                else (
                    "mean_squared_error"
                    if args.model == "deterministic"
                    else "joint_negative_log_likelihood"
                )
            ),
            "gradient_routing": (
                {
                    "backbone_and_mean": "mean_squared_error_only",
                    "covariance_projection": "detached_feature_residual_nll_only",
                    "compiled_operator_lifting": "no_gradient_in_faithful_objective",
                }
                if getattr(args, "faithful_joint", False)
                else "ordinary_objective_autograd"
            ),
        },
        "precision": {
            "backbone": "BF16 autocast on CUDA"
            if args.backbone_precision == "bf16"
            else "FP32",
            "operator_readout_spd_likelihood": "FP32",
            "tf32": False,
            "cudnn_benchmark": False,
            "distributed": False,
            "bitwise_cuda_reproducibility": (
                "not claimed; exact seed/split/sampler/checkpoint contract is recorded"
            ),
        },
        "randomness": {
            "seed": args.seed,
            "sampler": "RandomSampler with seed*1000003+epoch",
            "worker_seeding": "torch.initial_seed propagated to Python and NumPy",
            "num_workers": args.num_workers,
        },
        "backend": {
            "tensor_product_backend": args.tp_backend,
            "cueq_method": args.cueq_method,
            "compile_tensor_products": args.compile_tp,
        },
        "freeze": freeze,
    }
