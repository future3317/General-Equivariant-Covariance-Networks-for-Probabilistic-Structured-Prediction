"""Equivariant graph backbone with auditable tensor-product backends."""

from __future__ import annotations

import importlib.util
import math
from typing import Literal

import torch
from compatibility.cuequivariance import load_cuequivariance
from compatibility.e3nn import FullyConnectedNet, Gate, o3


TENSOR_PRODUCT_BACKENDS = ("e3nn", "cueq")
CUEQ_METHODS = ("naive", "fused_tp")
TensorProductBackend = Literal["e3nn", "cueq"]
CueqMethod = Literal["naive", "fused_tp"]


def _validate_tensor_product_options(backend: str, method: str) -> None:
    if backend not in TENSOR_PRODUCT_BACKENDS:
        raise ValueError(
            f"tp_backend must be one of {TENSOR_PRODUCT_BACKENDS}, got {backend!r}"
        )
    if method not in CUEQ_METHODS:
        raise ValueError(f"cueq_method must be one of {CUEQ_METHODS}, got {method!r}")
    if backend != "cueq" and method != "naive":
        raise ValueError("cueq_method is only applicable when tp_backend='cueq'")


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
        gated_irreps = o3.Irreps([(mul, ir) for mul, ir in self.irreps if ir.l > 0])

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
            self.input_proj = o3.Linear(
                self.irreps, scalars_irreps + irreps_gates + gated_irreps
            )
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
        *,
        tp_backend: TensorProductBackend = "e3nn",
        cueq_method: CueqMethod = "naive",
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        _validate_tensor_product_options(tp_backend, cueq_method)
        self.tp_backend = tp_backend
        self.cueq_method = cueq_method
        self.tp = self._build_tensor_product()
        self._compiled_tp = None
        self.fc = FullyConnectedNet(
            [num_basis, 32, self.tp.weight_numel],
            act=torch.nn.functional.silu,
        )
        self.self_interaction = o3.Linear(
            self.irreps_node_hidden, self.irreps_node_hidden
        )
        self.skip = o3.Linear(self.irreps_node_input, self.irreps_node_hidden)
        self.act = EquivariantActivation(self.irreps_node_hidden)
        self.irreps_out = self.act.irreps_out

    def _build_tensor_product(self) -> torch.nn.Module:
        if self.tp_backend == "e3nn":
            return o3.FullyConnectedTensorProduct(
                self.irreps_node_input,
                self.irreps_edge_attr,
                self.irreps_node_hidden,
                irrep_normalization="component",
                path_normalization="element",
                shared_weights=False,
                internal_weights=False,
            )
        try:
            cue, cuet = load_cuequivariance()
        except ImportError as error:
            raise RuntimeError(
                "tp_backend='cueq' requires cuequivariance and cuequivariance-torch"
            ) from error
        if (
            self.cueq_method == "fused_tp"
            and importlib.util.find_spec("cuequivariance_ops_torch") is None
        ):
            raise RuntimeError(
                "cueq_method='fused_tp' requires cuequivariance_ops_torch. "
                "NVIDIA currently publishes its cu12 wheel for Linux; use "
                "cueq_method='naive' with --compile_tp on Windows."
            )
        return cuet.FullyConnectedTensorProduct(
            cue.Irreps(cue.O3, str(self.irreps_node_input)),
            cue.Irreps(cue.O3, str(self.irreps_edge_attr)),
            cue.Irreps(cue.O3, str(self.irreps_node_hidden)),
            layout=cue.mul_ir,
            shared_weights=False,
            internal_weights=False,
            use_fallback=self.cueq_method == "naive",
            method=self.cueq_method,
        )

    def compile_tensor_product(self, *, dynamic: bool = True) -> None:
        """Compile only the selected tensor product while preserving its state dict.

        Tensor products accept a variable edge-batch dimension, hence dynamic
        shapes are enabled by default. Compilation is explicit because the
        first invocation has a non-negligible cold-start cost.
        """
        device = next(self.parameters()).device
        if device.type == "cuda" and importlib.util.find_spec("triton") is None:
            raise RuntimeError(
                "CUDA torch.compile requires a working Triton installation; "
                "the current environment has none. Install a PyTorch-compatible "
                "Triton build or leave --compile_tp disabled."
            )
        compiled = torch.compile(self.tp, fullgraph=True, dynamic=dynamic)
        object.__setattr__(self, "_compiled_tp", compiled)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_length_embedding: torch.Tensor,
        edge_weight: torch.Tensor,
        node_norm: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.fc(edge_length_embedding)
        tensor_product = self.tp if self._compiled_tp is None else self._compiled_tp
        messages = tensor_product(
            node_feats[edge_src], edge_attr, weights
        ) * edge_weight.unsqueeze(-1)
        sum_messages = messages.new_zeros((node_feats.size(0), messages.size(-1)))
        sum_messages.index_add_(0, edge_dst, messages)
        agg_messages = sum_messages * node_norm.unsqueeze(-1)
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
        tp_backend: TensorProductBackend = "e3nn",
        cueq_method: CueqMethod = "naive",
    ):
        super().__init__()
        _validate_tensor_product_options(tp_backend, cueq_method)
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.lmax = lmax
        self.atom_feature_dim = atom_feature_dim
        self.atom_features = atom_features
        self.tp_backend = tp_backend
        self.cueq_method = cueq_method

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
                tp_backend=tp_backend,
                cueq_method=cueq_method,
            )
            self.layers.append(conv)
            current_irreps = conv.irreps_out

        self.post_linear1 = o3.Linear(current_irreps, current_irreps)
        self.post_act = EquivariantActivation(current_irreps)
        self.post_linear2 = o3.Linear(
            self.post_act.irreps_out, self.post_act.irreps_out
        )
        self.irreps_out = self.post_linear2.irreps_out

    def compile_tensor_products(self, *, dynamic: bool = True) -> None:
        """Enable ``torch.compile`` for each edge tensor product only."""
        for layer in self.layers:
            layer.compile_tensor_product(dynamic=dynamic)

    def _build_hidden_irreps(self, hidden_dim: int, lmax: int) -> o3.Irreps:
        irreps_list = []
        for angular_momentum in range(lmax + 1):
            mul = max(1, hidden_dim // (2**angular_momentum))
            if angular_momentum == 0:
                irreps_list.append((mul, (angular_momentum, 1)))
            else:
                mul_each = max(1, mul // 2)
                irreps_list.append((mul_each, (angular_momentum, 1)))
                irreps_list.append((mul_each, (angular_momentum, -1)))
        return o3.Irreps(irreps_list).simplify()

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        if self.atom_features == "learnable":
            if not hasattr(data, "z"):
                raise ValueError(
                    "learnable atom_features requires data.z (atomic numbers)"
                )
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
        node_counts = torch.bincount(edge_dst, minlength=node_feats.size(0)).to(
            dtype=node_feats.dtype
        )
        node_norm = torch.rsqrt(node_counts + 1e-8)

        for layer in self.layers:
            node_feats = layer(
                node_feats,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedding,
                edge_weights,
                node_norm,
            )

        node_feats = self.post_linear1(node_feats)
        node_feats = self.post_act(node_feats)
        node_feats = self.post_linear2(node_feats)
        return node_feats, batch
