"""Equivariant graph backbone based on e3nn tensor products."""

from __future__ import annotations

import math
import torch
from compatibility.e3nn import FullyConnectedNet, Gate, o3, soft_one_hot_linspace
from torch_scatter import scatter, scatter_softmax


class ShiftedSoftPlus(torch.nn.Module):
    """Shifted softplus: softplus(x) - log(2)."""

    def __init__(self, beta: float = 1, threshold: float = 20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self._log2 = math.log(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(x) - self._log2


class EquivariantActivation(torch.nn.Module):
    """Equivariant activation using e3nn's Gate module."""

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)

        act = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates = {1: ShiftedSoftPlus(), -1: torch.tanh}

        scalars_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps if ir.l == 0]
        ).simplify()
        gated_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps if ir.l > 0]
        )

        if len(scalars_irreps) > 0 and len(gated_irreps) > 0:
            irreps_gates = o3.Irreps(
                [(mul, "0e") for mul, _ in gated_irreps]
            ).simplify()
            scalar_acts = [act[ir.p] for _, ir in scalars_irreps]
            gate_acts = [act_gates[ir.p] for _, ir in irreps_gates]

            self.gate = Gate(
                scalars_irreps,
                scalar_acts,
                irreps_gates,
                gate_acts,
                gated_irreps,
            )
            self.input_proj = o3.Linear(self.irreps, scalars_irreps + irreps_gates + gated_irreps)
            self.has_gate = True
            self.irreps_out = self.gate.irreps_out
        else:
            self.has_gate = False
            self.irreps_out = self.irreps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_gate:
            return self.gate(self.input_proj(x))
        return x


class EquivariantMessagePassing(torch.nn.Module):
    """Single E(3)-equivariant message-passing layer."""

    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_hidden: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        num_basis: int = 10,
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_node_hidden,
            shared_weights=False,
            internal_weights=False,
        )
        self.fc = FullyConnectedNet(
            [num_basis, 32, self.tp.weight_numel],
            act=torch.nn.functional.silu,
        )
        self.self_interaction = o3.Linear(self.irreps_node_hidden, self.irreps_node_hidden)
        self.skip = o3.Linear(self.irreps_node_input, self.irreps_node_hidden)
        self.act = EquivariantActivation(self.irreps_node_hidden)
        self.irreps_out = self.act.irreps_out

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_length_embedding: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.fc(edge_length_embedding)
        messages = (
            self.tp(node_feats[edge_src], edge_attr, weights)
            * edge_weight.unsqueeze(-1)
        )
        sum_messages = scatter(
            messages,
            edge_dst,
            dim=0,
            dim_size=node_feats.size(0),
            reduce="sum",
        )
        ones = torch.ones(messages.size(0), device=messages.device)
        node_counts = scatter(
            ones,
            edge_dst,
            dim=0,
            dim_size=node_feats.size(0),
            reduce="sum",
        )
        agg_messages = sum_messages / torch.sqrt(node_counts + 1e-8).unsqueeze(-1)
        agg_messages = self.self_interaction(agg_messages)
        output = self.act(agg_messages + self.skip(node_feats))
        return output


class EquivariantBackbone(torch.nn.Module):
    """Equivariant graph backbone.

    Consumes a PyG-like ``Data`` object with fields ``node_features``,
    ``edge_index``, ``edge_sh`` (spherical harmonics), ``edge_rbf`` (radial
    basis), and ``edge_weights``. Returns per-node equivariant features and the
    batch vector.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        max_radius: float = 6.0,
        num_basis: int = 8,
        lmax: int = 2,
        num_layers: int = 2,
        atom_feature_dim: int = 49,
        atom_features: str = "manual",
    ):
        super().__init__()
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.lmax = lmax
        self.atom_feature_dim = atom_feature_dim
        self.atom_features = atom_features

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        irreps_node_hidden = self._build_hidden_irreps(hidden_dim, lmax)

        if atom_features == "learnable":
            self.atom_embedding = torch.nn.Embedding(119, atom_feature_dim)
            irreps_embedded = f"{atom_feature_dim}x0e"
        else:
            irreps_embedded = f"{atom_feature_dim}x0e"

        self.layers = torch.nn.ModuleList()
        current_irreps = irreps_embedded
        for _ in range(num_layers):
            conv = EquivariantMessagePassing(
                current_irreps,
                irreps_node_hidden,
                self.irreps_sh,
                num_basis,
            )
            self.layers.append(conv)
            current_irreps = conv.irreps_out

        self.post_linear1 = o3.Linear(current_irreps, current_irreps)
        self.post_act = EquivariantActivation(current_irreps)
        self.post_linear2 = o3.Linear(self.post_act.irreps_out, self.post_act.irreps_out)
        self.irreps_out = self.post_linear2.irreps_out

    def _build_hidden_irreps(self, hidden_dim: int, lmax: int) -> o3.Irreps:
        irreps_list = []
        for l in range(lmax + 1):
            mul = max(1, hidden_dim // (2 ** l))
            if l == 0:
                irreps_list.append((mul, (l, 1)))
            else:
                mul_each = max(1, mul // 2)
                irreps_list.append((mul_each, (l, 1)))
                irreps_list.append((mul_each, (l, -1)))
        return o3.Irreps(irreps_list).simplify()

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atom_features == "learnable":
            if not hasattr(data, "z"):
                raise ValueError("learnable atom_features requires data.z (atomic numbers)")
            node_feats = self.atom_embedding(data.z)
        else:
            node_feats = data.node_features
            assert node_feats.shape[-1] == self.atom_feature_dim, (
                f"node_features dim {node_feats.shape[-1]} != expected {self.atom_feature_dim}"
            )
        batch = data.batch
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]
        edge_attr = data.edge_sh
        edge_length_embedding = data.edge_rbf
        edge_weights = data.edge_weights

        for layer in self.layers:
            node_feats = layer(
                node_feats,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedding,
                edge_weights,
            )

        node_feats = self.post_linear1(node_feats)
        node_feats = self.post_act(node_feats)
        node_feats = self.post_linear2(node_feats)
        return node_feats, batch
