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
    """Sample one labeled tree with fixed degrees by deterministic edge swaps.

    The sampler starts from the registered tree and accepts only simple,
    connected two-edge switches.  It therefore preserves the labeled degree
    sequence while making the null distribution an explicit edge-swap process,
    rather than silently changing the node labels through a new tree draw.
    """
    reference = _validate_tree(reference_edges, num_nodes)
    rng = random.Random(int(seed))
    current = list(reference)
    edge_count = len(current)
    accepted = 0
    target_swaps = max(8, 4 * num_nodes)
    max_attempts = target_swaps * 40
    for _ in range(max_attempts):
        first, second = rng.sample(range(edge_count), 2)
        a, b = current[first]
        c, d = current[second]
        if rng.random() < 0.5:
            candidates = ((a, c, b, d), (a, d, b, c))
        else:
            candidates = ((a, c, b, d), (a, d, b, c))[::-1]
        proposal = None
        for left, right, other_left, other_right in candidates:
            candidate = list(current)
            candidate[first] = (left, right)
            candidate[second] = (other_left, other_right)
            try:
                normalized = _validate_tree(candidate, num_nodes)
            except ValueError:
                continue
            if normalized != tuple(current):
                proposal = list(normalized)
                break
        if proposal is None:
            continue
        current = proposal
        accepted += 1
        if accepted >= target_swaps:
            break
    if accepted == 0:
        raise RuntimeError("edge-swap sampler made no valid topology move")
    return tuple(current)


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
                "sampler": "connected_degree_preserving_edge_swap",
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
