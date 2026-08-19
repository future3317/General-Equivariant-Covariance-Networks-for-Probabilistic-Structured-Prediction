"""Pre-generate degree-sequence-matched ITOP tree controls."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path


def _validate_tree(
    edges: Iterable[tuple[int, int]], num_nodes: int
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(source), int(target)) for source, target in edges)
    if num_nodes < 2 or len(normalized) != num_nodes - 1:
        raise ValueError("reference must be a tree with num_nodes-1 edges")
    adjacency = [[] for _ in range(num_nodes)]
    seen: set[tuple[int, int]] = set()
    for source, target in normalized:
        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            raise ValueError("tree edge is outside the node range")
        if source == target:
            raise ValueError("tree cannot contain a self edge")
        edge = (min(source, target), max(source, target))
        if edge in seen:
            raise ValueError("tree cannot contain duplicate undirected edges")
        seen.add(edge)
        adjacency[source].append(target)
        adjacency[target].append(source)
    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    if len(visited) != num_nodes:
        raise ValueError("reference must be connected and therefore a tree")
    return normalized


def _degree_sequence(edges: tuple[tuple[int, int], ...], num_nodes: int) -> list[int]:
    degrees = [0] * num_nodes
    for source, target in edges:
        degrees[source] += 1
        degrees[target] += 1
    return degrees


def generate_degree_matched_tree(
    reference_edges: Iterable[tuple[int, int]], *, num_nodes: int, seed: int
) -> tuple[tuple[int, int], ...]:
    """Sample uniformly from trees with the reference labeled degree sequence.

    A labeled tree is in bijection with its Prüfer sequence. Repeating node
    ``i`` exactly ``degree[i] - 1`` times and uniformly shuffling that multiset
    samples the degree-matched tree without a finite-step Markov-chain mixing
    assumption.
    """
    reference = _validate_tree(reference_edges, num_nodes)
    rng = random.Random(int(seed))
    degrees = _degree_sequence(reference, num_nodes)
    prufer = [node for node, degree in enumerate(degrees) for _ in range(degree - 1)]
    rng.shuffle(prufer)
    remaining = degrees[:]
    edges: list[tuple[int, int]] = []
    for node in prufer:
        leaf = min(index for index, degree in enumerate(remaining) if degree == 1)
        edges.append((leaf, node))
        remaining[leaf] -= 1
        remaining[node] -= 1
    leaves = [index for index, degree in enumerate(remaining) if degree == 1]
    if len(leaves) != 2:
        raise RuntimeError("invalid Prüfer decode for degree-matched tree")
    edges.append((leaves[0], leaves[1]))
    return _validate_tree(edges, num_nodes)


def topology_manifest(
    reference_edges: Iterable[tuple[int, int]], *, num_nodes: int, count: int, seed: int
) -> list[dict]:
    """Return all pre-generated controls without outcome-based filtering."""
    reference = _validate_tree(reference_edges, num_nodes)
    degrees = _degree_sequence(reference, num_nodes)
    records = []
    for index in range(count):
        topology_seed = int(seed) + index
        edges = generate_degree_matched_tree(
            reference, num_nodes=num_nodes, seed=topology_seed
        )
        records.append(
            {
                "index": index,
                "topology_seed": topology_seed,
                "num_nodes": num_nodes,
                "reference_edges": [list(edge) for edge in reference],
                "edges": [list(edge) for edge in edges],
                "labeled_degree_sequence": degrees,
                "degree_sequence": sorted(degrees),
                "sampler": "uniform_labeled_degree_sequence_prufer",
                "outcome_filtered": False,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    from data.itop_dataset import ITOP_SKELETON_EDGES

    payload = {
        "schema_version": 1,
        "kind": "itop_degree_sequence_matched_tree_manifest",
        "count": args.count,
        "seed": args.seed,
        "outcome_filtered": False,
        "records": topology_manifest(
            ITOP_SKELETON_EDGES, num_nodes=15, count=args.count, seed=args.seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
