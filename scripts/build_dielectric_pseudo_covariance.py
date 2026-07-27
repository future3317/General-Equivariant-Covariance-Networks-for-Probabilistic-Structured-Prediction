# ruff: noqa: E402
"""Construct audited, train-only isotropic pseudo-covariance targets.

No directional covariance cache can be built by this entry point: invariant
kNN alone has no neighbour-to-query O(3) transport.  The output is a residual
covariance target, never a Student-t scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.dielectric_dataset import DielectricIrrepsDataset  # noqa: E402
from data.paths import dataset_dir  # noqa: E402
from data.pseudo_covariance import (
    PSEUDO_CACHE_VERSION,
    build_isotropic_pseudo_covariance,
    invariant_structure_embedding,
    validate_oof_residual_payload,
)  # noqa: E402
from scripts.dielectric_runtime import dataset_provenance, sha256_file, source_provenance  # noqa: E402


def build_pseudo_covariance(
    oof_residuals_path: str | Path,
    *, data_dir: str | Path | None,
    output: str | Path,
    k: int,
    tau: float,
    shrinkage: float,
    epsilon: float,
    lmax: int = 2,
    storage: str = "files",
    shard_cache_size: int = 2,
) -> dict:
    source = Path(oof_residuals_path)
    oof = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(oof, dict):
        raise ValueError("OOF residual cache must be a dictionary")
    validate_oof_residual_payload(oof)
    root = dataset_dir(data_dir, "mp_dielectric")
    dataset = DielectricIrrepsDataset(root, "train", lmax=lmax, storage=storage, shard_cache_size=shard_cache_size)
    if len(dataset) != oof["residuals"].shape[0]:
        raise ValueError("OOF residual cache length differs from the current train split")
    embeddings = torch.stack([invariant_structure_embedding(dataset[index]) for index in range(len(dataset))])
    result = build_isotropic_pseudo_covariance(
        oof["residuals"], embeddings, k=k, tau=tau, shrinkage=shrinkage, epsilon=epsilon
    )
    covariance = result["covariance"]
    eigvals = torch.linalg.eigvalsh(covariance)
    metadata = {
        "version": PSEUDO_CACHE_VERSION,
        "split": "train",
        "mode": "isotropic_only",
        "coordinate_semantics": "residual_covariance",
        "target_is_student_t_scale": False,
        "transport_certificate": None,
        "embedding": {
            "name": "deterministic_structure_invariant_v1",
            "invariance": "translation, O(3), atom_permutation",
            "uses_labels": False,
            "fold_specific": False,
            "dimension": int(embeddings.shape[1]),
        },
        "knn": {"k": k, "tau": tau, "weight": "exp(-squared_distance/tau)", "self_excluded": True},
        "shrinkage": {"lambda": shrinkage, "epsilon": epsilon, "formula": "(1-lambda)Sigma + lambda Tr(Sigma)/d I + epsilon I"},
        "diagnostics": {
            "effective_neighbours": {"mean": float(result["effective_neighbours"].mean()), "min": float(result["effective_neighbours"].min()), "max": float(result["effective_neighbours"].max())},
            "rank": 6,
            "condition_number": {"mean": 1.0, "max": 1.0},
            "eigenvalue_min": float(eigvals.min()),
            "eigenvalue_max": float(eigvals.max()),
            "directional_supervision": "none; full local covariance is diagnostic-only because it lacks transport",
        },
        "oof_residual_cache": {"path": str(source), "sha256": sha256_file(source), "checkpoint_chain_sha256": oof.get("checkpoint_chain_sha256")},
        "dataset": dataset_provenance(root),
        "source": source_provenance(_ROOT),
    }
    payload = {**metadata, **result}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof_residuals", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--storage", choices=("files", "shards"), default="files")
    parser.add_argument("--shard_cache_size", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build_pseudo_covariance(args.oof_residuals, data_dir=args.data_dir, output=args.output, k=args.k, tau=args.tau, shrinkage=args.shrinkage, epsilon=args.epsilon, lmax=args.lmax, storage=args.storage, shard_cache_size=args.shard_cache_size), indent=2))


if __name__ == "__main__":
    main()
