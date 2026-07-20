"""Graph-structured equivariant precision assembled from local SPD blocks."""

from __future__ import annotations

import torch

from representations.graph_structure import EquivariantOutputGraph
from spd_maps.base import SPDMap, symmetrize


class GraphStructuredPrecisionMap(SPDMap):
    r"""Assemble a global precision from unary and relational potentials.

    Raw parameters contain ``J + E`` symmetric matrices.  Matrix exponentials
    produce unary precisions ``U_j`` and edge precisions ``W_e`` and the global
    precision is

    ``Q = BlockDiag(U) + (B kron I)^T BlockDiag(W) (B kron I)``.

    ``logdet`` and ``precision_action`` operate in precision coordinates, so
    training never materializes ``Q^{-1}``.  ``forward`` computes covariance
    only when explicitly requested by evaluation or sampling code.
    """

    def __init__(self, graph: EquivariantOutputGraph):
        super().__init__()
        self.graph = graph
        self.register_buffer("incidence", graph.incidence_matrix())
        self.register_buffer(
            "edge_sources",
            torch.tensor([source for source, _ in graph.edges], dtype=torch.long),
        )
        self.register_buffer(
            "edge_targets",
            torch.tensor([target for _, target in graph.edges], dtype=torch.long),
        )
        self._tree_parent, self._tree_parent_edge, self._tree_postorder = (
            self._build_tree_elimination_plan()
        )

    def _build_tree_elimination_plan(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Precompute a rooted post-order plan for exact tree elimination."""
        if not self.graph.is_tree:
            return (), (), ()
        adjacency: list[list[tuple[int, int]]] = [
            [] for _ in range(self.graph.num_nodes)
        ]
        for edge_index, (source, target) in enumerate(self.graph.edges):
            adjacency[source].append((target, edge_index))
            adjacency[target].append((source, edge_index))

        parent = [-1] * self.graph.num_nodes
        parent_edge = [-1] * self.graph.num_nodes
        order = [0]
        for node in order:
            for neighbor, edge_index in adjacency[node]:
                if neighbor == parent[node]:
                    continue
                parent[neighbor] = node
                parent_edge[neighbor] = edge_index
                order.append(neighbor)
        return tuple(parent), tuple(parent_edge), tuple(reversed(order[1:]))

    def _validate(self, params: torch.Tensor) -> None:
        expected = (
            self.graph.num_potentials,
            self.graph.block_dim,
            self.graph.block_dim,
        )
        if params.shape[-3:] != expected:
            raise ValueError(
                f"graph precision parameters must end in {expected}, got {params.shape[-3:]}"
            )

    def local_precisions(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unary and edge SPD precision blocks."""
        self._validate(params)
        blocks = torch.linalg.matrix_exp(symmetrize(params))
        return (
            blocks[..., : self.graph.num_nodes, :, :],
            blocks[..., self.graph.num_nodes :, :, :],
        )

    def _precision_from_blocks(
        self,
        params: torch.Tensor,
        unary: torch.Tensor,
        relational: torch.Tensor,
    ) -> torch.Tensor:
        dimension = self.graph.output_dim
        precision = params.new_zeros((*params.shape[:-3], dimension, dimension))

        def block_slice(node: int) -> slice:
            start = node * self.graph.block_dim
            return slice(start, start + self.graph.block_dim)

        for node in range(self.graph.num_nodes):
            node_slice = block_slice(node)
            precision[..., node_slice, node_slice] += unary[..., node, :, :]

        for edge_index, (source, target) in enumerate(self.graph.edges):
            source_slice = block_slice(source)
            target_slice = block_slice(target)
            block = relational[..., edge_index, :, :]
            precision[..., source_slice, source_slice] += block
            precision[..., target_slice, target_slice] += block
            precision[..., source_slice, target_slice] -= block
            precision[..., target_slice, source_slice] -= block
        return symmetrize(precision)

    def precision(self, params: torch.Tensor) -> torch.Tensor:
        unary, relational = self.local_precisions(params)
        return self._precision_from_blocks(params, unary, relational)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        precision = self.precision(params)
        cholesky = torch.linalg.cholesky(precision)
        return torch.cholesky_inverse(cholesky)

    def logdet(self, params: torch.Tensor) -> torch.Tensor:
        """Return ``log det Sigma = -log det Q``."""
        unary, relational = self.local_precisions(params)
        return self._logdet_from_blocks(params, unary, relational)

    @staticmethod
    def _cholesky_logdet(matrix: torch.Tensor) -> torch.Tensor:
        cholesky = torch.linalg.cholesky(matrix)
        return 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)

    def _tree_logdet_precision(
        self,
        unary: torch.Tensor,
        relational: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``log det Q`` by exact block Schur elimination on a tree."""
        diagonal = [unary[..., node, :, :] for node in range(self.graph.num_nodes)]
        for edge_index, (source, target) in enumerate(self.graph.edges):
            edge_block = relational[..., edge_index, :, :]
            diagonal[source] = diagonal[source] + edge_block
            diagonal[target] = diagonal[target] + edge_block

        logdet = unary.new_zeros(unary.shape[:-3])
        for node in self._tree_postorder:
            pivot = symmetrize(diagonal[node])
            cholesky = torch.linalg.cholesky(pivot)
            logdet = logdet + 2.0 * torch.log(
                torch.diagonal(cholesky, dim1=-2, dim2=-1)
            ).sum(-1)
            edge_index = self._tree_parent_edge[node]
            edge_block = relational[..., edge_index, :, :]
            correction = edge_block @ torch.cholesky_solve(edge_block, cholesky)
            parent = self._tree_parent[node]
            diagonal[parent] = diagonal[parent] - correction

        return logdet + self._cholesky_logdet(symmetrize(diagonal[0]))

    def _logdet_from_blocks(
        self,
        params: torch.Tensor,
        unary: torch.Tensor,
        relational: torch.Tensor,
    ) -> torch.Tensor:
        if self.graph.is_tree:
            return -self._tree_logdet_precision(unary, relational)
        precision = self._precision_from_blocks(params, unary, relational)
        return -self._cholesky_logdet(precision)

    def _precision_action_from_blocks(
        self,
        unary: torch.Tensor,
        relational: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        node_residual = residual.reshape(
            *residual.shape[:-1], self.graph.num_nodes, self.graph.block_dim
        )
        unary_action = torch.einsum(
            "...ji,...jik,...jk->...",
            node_residual,
            unary,
            node_residual,
        )

        if self.graph.num_edges == 0:
            return unary_action
        differences = (
            node_residual[..., self.edge_targets, :]
            - node_residual[..., self.edge_sources, :]
        )
        relational_action = torch.einsum(
            "...ei,...eik,...ek->...",
            differences,
            relational,
            differences,
        )
        return unary_action + relational_action

    def precision_action(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate ``r^T Q r`` directly from local graph potentials."""
        if residual.shape[-1] != self.graph.output_dim:
            raise ValueError(
                f"residual last dim {residual.shape[-1]} != {self.graph.output_dim}"
            )
        unary, relational = self.local_precisions(params)
        return self._precision_action_from_blocks(unary, relational, residual)

    def statistics(
        self, params: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse one set of local matrix exponentials for both NLL statistics."""
        if residual.shape[-1] != self.graph.output_dim:
            raise ValueError(
                f"residual last dim {residual.shape[-1]} != {self.graph.output_dim}"
            )
        unary, relational = self.local_precisions(params)
        return (
            self._logdet_from_blocks(params, unary, relational),
            self._precision_action_from_blocks(unary, relational, residual),
        )

    def sample(
        self,
        mean: torch.Tensor,
        params: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Draw Gaussian samples using triangular solves in precision space."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        precision = self.precision(params)
        cholesky = torch.linalg.cholesky(precision)
        noise = torch.randn(
            *mean.shape[:-1],
            self.graph.output_dim,
            num_samples,
            dtype=mean.dtype,
            device=mean.device,
        )
        residual = torch.linalg.solve_triangular(
            cholesky.transpose(-1, -2), noise, upper=True
        )
        return mean.unsqueeze(-1) + residual
