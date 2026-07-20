"""Typed graph structure for repeated equivariant output variables."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from compatibility.e3nn import o3


@dataclass(frozen=True)
class EquivariantOutputGraph:
    """A graph whose nodes carry one copy of the same O(3) irrep.

    The current backend intentionally requires one multiplicity-one local
    irrep.  This covers articulated 3D points (``1o``) while keeping the flat
    output layout node-major, so every node occupies one contiguous block.
    """

    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    node_irrep: o3.Irrep | str = "1o"
    node_names: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        irrep = o3.Irrep(self.node_irrep)
        object.__setattr__(self, "node_irrep", irrep)

        normalized_edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for raw_source, raw_target in self.edges:
            source, target = int(raw_source), int(raw_target)
            if source == target:
                raise ValueError("graph precision does not accept self edges")
            if not (0 <= source < self.num_nodes and 0 <= target < self.num_nodes):
                raise ValueError(f"edge {(source, target)} is outside the node range")
            undirected = (min(source, target), max(source, target))
            if undirected in seen:
                raise ValueError(f"duplicate undirected edge: {undirected}")
            seen.add(undirected)
            normalized_edges.append((source, target))
        object.__setattr__(self, "edges", tuple(normalized_edges))

        if self.node_names is not None and len(self.node_names) != self.num_nodes:
            raise ValueError("node_names must contain one name per node")

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def block_dim(self) -> int:
        return self.node_irrep.dim

    @property
    def output_dim(self) -> int:
        return self.num_nodes * self.block_dim

    @property
    def output_irreps(self) -> o3.Irreps:
        return o3.Irreps([(self.num_nodes, self.node_irrep)])

    @property
    def num_potentials(self) -> int:
        return self.num_nodes + self.num_edges

    @property
    def is_tree(self) -> bool:
        """Whether the undirected output graph is connected and acyclic."""
        if self.num_edges != self.num_nodes - 1:
            return False
        adjacency: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for source, target in self.edges:
            adjacency[source].append(target)
            adjacency[target].append(source)
        visited = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return len(visited) == self.num_nodes

    def incidence_matrix(
        self,
        *,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        incidence = torch.zeros(
            self.num_edges,
            self.num_nodes,
            dtype=dtype,
            device=device,
        )
        for edge_index, (source, target) in enumerate(self.edges):
            incidence[edge_index, source] = -1.0
            incidence[edge_index, target] = 1.0
        return incidence

    def representation_matrix(self, rotation: torch.Tensor) -> torch.Tensor:
        local = self.node_irrep.D_from_matrix(rotation)
        identity = torch.eye(
            self.num_nodes,
            dtype=local.dtype,
            device=local.device,
        )
        return torch.kron(identity, local)

    def as_dict(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "edges": [list(edge) for edge in self.edges],
            "node_irrep": str(self.node_irrep),
            "node_names": list(self.node_names)
            if self.node_names is not None
            else None,
            "block_dim": self.block_dim,
            "output_dim": self.output_dim,
            "is_tree": self.is_tree,
        }
