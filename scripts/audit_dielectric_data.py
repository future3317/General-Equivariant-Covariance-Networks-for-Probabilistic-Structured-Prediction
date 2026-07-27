"""Profile precomputed dielectric splits for integrity and target shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.dielectric_dataset import DielectricIrrepsDataset
from data.paths import dataset_dir


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    return {
        str(level): float(torch.quantile(values, level))
        for level in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
    }


def _profile(dataset: DielectricIrrepsDataset) -> dict:
    targets, nodes, edges = [], [], []
    invalid = {"target": 0, "positions": 0, "edge_index": 0}
    target_fingerprints: set[bytes] = set()
    for index in range(len(dataset)):
        graph = dataset[index]
        target = graph.y_irreps.reshape(-1).double()
        targets.append(target)
        nodes.append(int(graph.pos.shape[0]))
        edges.append(int(graph.edge_index.shape[1]))
        invalid["target"] += int(not bool(torch.isfinite(target).all()))
        invalid["positions"] += int(not bool(torch.isfinite(graph.pos).all()))
        invalid["edge_index"] += int(
            graph.edge_index.numel() == 0
            or int(graph.edge_index.min()) < 0
            or int(graph.edge_index.max()) >= graph.pos.shape[0]
        )
        target_fingerprints.add(target.float().numpy().tobytes())

    target_matrix = torch.stack(targets)
    node_tensor, edge_tensor = torch.tensor(nodes), torch.tensor(edges)
    return {
        "num_graphs": len(dataset),
        "invalid_graphs": invalid,
        "duplicate_exact_targets": int(len(dataset) - len(target_fingerprints)),
        "node_count_quantiles": _quantiles(node_tensor.double()),
        "edge_count_quantiles": _quantiles(edge_tensor.double()),
        "target_irrep_mean": target_matrix.mean(0).tolist(),
        "target_irrep_std": target_matrix.std(0, unbiased=False).tolist(),
        "target_irrep_quantiles": {
            str(level): torch.quantile(target_matrix, level, dim=0).tolist()
            for level in (0.01, 0.5, 0.99)
        },
        "target_norm_quantiles": _quantiles(target_matrix.norm(dim=-1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--storage", choices=("files", "shards"), default="files")
    parser.add_argument("--lmax", type=int, default=2)
    args = parser.parse_args()

    root = dataset_dir(args.data_dir, "mp_dielectric")
    profiles = {
        split: _profile(
            DielectricIrrepsDataset(
                root, split, lmax=args.lmax, storage=args.storage
            )
        )
        for split in ("train", "val", "test")
    }
    val_mean = torch.tensor(profiles["val"]["target_irrep_mean"])
    val_std = torch.tensor(profiles["val"]["target_irrep_std"]).clamp_min(1e-12)
    test_mean = torch.tensor(profiles["test"]["target_irrep_mean"])
    test_std = torch.tensor(profiles["test"]["target_irrep_std"])
    profiles["val_to_test_target_shift"] = {
        "mean_shift_in_val_std": ((test_mean - val_mean) / val_std).tolist(),
        "std_ratio_test_over_val": (test_std / val_std).tolist(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profiles, indent=2))
    print(json.dumps(profiles["val_to_test_target_shift"], indent=2))


if __name__ == "__main__":
    main()
