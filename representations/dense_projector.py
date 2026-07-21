"""Checkpoint-preserving dense-projector lowering for e3nn tensor products.

The executor retains the flat weight coordinates and path normalization of
``e3nn.o3.FullyConnectedTensorProduct``.  It changes only the contraction
schedule: multiplicity mixing is performed before a small frozen angular
projector.  This makes it suitable for lowering an already trained spherical
CG checkpoint without changing the represented function.
"""

from __future__ import annotations

import math

import torch
from compatibility.e3nn import o3


class MultiplicityFirstDenseTensorProduct(torch.nn.Module):
    """Dense-projector implementation of a fully connected tensor product.

    With ``contraction_rank=None``, ``weight`` has exactly the same flat layout
    as e3nn's fully connected tensor product, so state dictionaries load
    without a learned or numerical basis conversion.  A positive rank creates
    independent CP factors and is explicitly approximate whenever it is below
    the exact rank of at least one multiplicity matrix.
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps | str,
        irreps_in2: o3.Irreps | str,
        irreps_out: o3.Irreps | str,
        *,
        contraction_rank: int | None = None,
    ):
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        reference = o3.FullyConnectedTensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
            path_normalization="element",
        )
        self._paths = self._collect_paths(reference)
        self.max_exact_rank = max(path["exact_rank"] for path in self._paths)
        if contraction_rank is not None and contraction_rank < 1:
            raise ValueError("contraction_rank must be positive")
        if contraction_rank is not None and contraction_rank >= self.max_exact_rank:
            contraction_rank = None
        self.contraction_rank = contraction_rank
        self.is_exact = contraction_rank is None
        self._register_couplings()

        if self.is_exact:
            self.weight_numel = reference.weight_numel
            self.weight = torch.nn.Parameter(reference.weight.detach().clone())
            self.left_factors = torch.nn.ParameterList()
            self.right_factors = torch.nn.ParameterList()
        else:
            self.weight_numel = 0
            self.register_parameter("weight", None)
            self.left_factors = torch.nn.ParameterList()
            self.right_factors = torch.nn.ParameterList()
            assert self.contraction_rank is not None
            for path in self._paths:
                left_mul = self.irreps_in1[path["i_in1"]].mul
                right_mul = self.irreps_in2[path["i_in2"]].mul
                out_mul = self.irreps_out[path["i_out"]].mul
                rank = min(self.contraction_rank, path["exact_rank"])
                left = torch.empty(out_mul, rank, left_mul)
                right = torch.empty(out_mul, rank, right_mul)
                torch.nn.init.normal_(left, std=1.0 / math.sqrt(left_mul * rank))
                torch.nn.init.normal_(right, std=1.0 / math.sqrt(right_mul * rank))
                self.left_factors.append(torch.nn.Parameter(left))
                self.right_factors.append(torch.nn.Parameter(right))
        del reference

    @staticmethod
    def _collect_paths(reference: torch.nn.Module) -> list[dict]:
        paths: list[dict] = []
        offset = 0
        for instruction in reference.instructions:
            if instruction.connection_mode != "uvw" or not instruction.has_weight:
                raise ValueError(
                    "dense-projector lowering requires weighted uvw instructions"
                )
            size = math.prod(instruction.path_shape)
            left_mul, right_mul, _ = instruction.path_shape
            paths.append(
                {
                    "i_in1": instruction.i_in1,
                    "i_in2": instruction.i_in2,
                    "i_out": instruction.i_out,
                    "start": offset,
                    "stop": offset + size,
                    "shape": tuple(instruction.path_shape),
                    "path_weight": float(instruction.path_weight),
                    "exact_rank": min(left_mul, right_mul),
                }
            )
            offset += size
        if not paths:
            raise ValueError("tensor product has no executable paths")
        return paths

    def _register_couplings(self) -> None:
        keys = {
            (
                self.irreps_in1[path["i_in1"]].ir.l,
                self.irreps_in2[path["i_in2"]].ir.l,
                self.irreps_out[path["i_out"]].ir.l,
            )
            for path in self._paths
        }
        for left_l, right_l, output_l in keys:
            coupling = (
                o3.wigner_3j(
                    left_l,
                    right_l,
                    output_l,
                    dtype=torch.float64,
                )
                .permute(2, 0, 1)
                .contiguous()
            )
            self.register_buffer(
                f"_coupling_{left_l}_{right_l}_{output_l}_float64",
                coupling,
                persistent=False,
            )
            self.register_buffer(
                f"_coupling_{left_l}_{right_l}_{output_l}_float32",
                coupling.float(),
                persistent=False,
            )

    def _coupling(self, path: dict, dtype: torch.dtype) -> torch.Tensor:
        left_l = self.irreps_in1[path["i_in1"]].ir.l
        right_l = self.irreps_in2[path["i_in2"]].ir.l
        output_l = self.irreps_out[path["i_out"]].ir.l
        suffix = "float64" if dtype == torch.float64 else "float32"
        coupling = getattr(self, f"_coupling_{left_l}_{right_l}_{output_l}_{suffix}")
        return coupling if coupling.dtype == dtype else coupling.to(dtype=dtype)

    @staticmethod
    def _project(
        left: torch.Tensor,
        mixed: torch.Tensor,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        pair = (left.unsqueeze(-3).unsqueeze(-1) * mixed.unsqueeze(-2)).flatten(-2)
        return torch.matmul(pair, coupling.flatten(1).T).sum(dim=-2)

    def _exact_path(
        self,
        path: dict,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight[path["start"] : path["stop"]].reshape(path["shape"])
        mixed = torch.einsum("uvw,...vj->...wuj", weight * path["path_weight"], right)
        return self._project(left, mixed, self._coupling(path, left.dtype))

    def _factorized_path(
        self,
        path_index: int,
        path: dict,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        left_mixed = torch.einsum(
            "wru,...ui->...wri", self.left_factors[path_index], left
        )
        right_mixed = torch.einsum(
            "wrv,...vj->...wrj", self.right_factors[path_index], right
        )
        pair = (left_mixed.unsqueeze(-1) * right_mixed.unsqueeze(-2)).flatten(-2)
        projected = torch.matmul(pair, self._coupling(path, left.dtype).flatten(1).T)
        return projected.sum(dim=-2) * path["path_weight"]

    def load_e3nn_weights(self, tensor_product: torch.nn.Module) -> None:
        """Copy the exact flat weight vector from an e3nn reference module."""
        if not self.is_exact:
            raise RuntimeError("truncated lowering has no exact e3nn weight map")
        if o3.Irreps(tensor_product.irreps_in1) != self.irreps_in1:
            raise ValueError("first input irreps do not match")
        if o3.Irreps(tensor_product.irreps_in2) != self.irreps_in2:
            raise ValueError("second input irreps do not match")
        if o3.Irreps(tensor_product.irreps_out) != self.irreps_out:
            raise ValueError("output irreps do not match")
        if tensor_product.weight.numel() != self.weight.numel():
            raise ValueError("flat weight layouts do not match")
        with torch.no_grad():
            self.weight.copy_(tensor_product.weight)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Ignore only deterministic e3nn code-generation buffers."""
        generated_prefix = prefix + "_compiled_main_left_right."
        for key in list(state_dict):
            if key == prefix + "output_mask" or key.startswith(generated_prefix):
                state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        if input1.shape[-1] != self.irreps_in1.dim:
            raise ValueError(
                f"expected first input dimension {self.irreps_in1.dim}, "
                f"got {input1.shape[-1]}"
            )
        if input2.shape[-1] != self.irreps_in2.dim:
            raise ValueError(
                f"expected second input dimension {self.irreps_in2.dim}, "
                f"got {input2.shape[-1]}"
            )
        left_inputs = [
            input1[..., irrep_slice].reshape(
                *input1.shape[:-1], multiplicity, irrep.dim
            )
            for irrep_slice, (multiplicity, irrep) in zip(
                self.irreps_in1.slices(), self.irreps_in1
            )
        ]
        right_inputs = [
            input2[..., irrep_slice].reshape(
                *input2.shape[:-1], multiplicity, irrep.dim
            )
            for irrep_slice, (multiplicity, irrep) in zip(
                self.irreps_in2.slices(), self.irreps_in2
            )
        ]
        outputs: list[torch.Tensor | None] = [None] * len(self.irreps_out)
        for path_index, path in enumerate(self._paths):
            left = left_inputs[path["i_in1"]]
            right = right_inputs[path["i_in2"]]
            contribution = (
                self._exact_path(path, left, right)
                if self.is_exact
                else self._factorized_path(path_index, path, left, right)
            )
            current = outputs[path["i_out"]]
            outputs[path["i_out"]] = (
                contribution if current is None else current + contribution
            )
        pieces = []
        for index, (multiplicity, irrep) in enumerate(self.irreps_out):
            value = outputs[index]
            if value is None:
                value = input1.new_zeros(*input1.shape[:-1], multiplicity, irrep.dim)
            pieces.append(value.flatten(start_dim=-2))
        return torch.cat(pieces, dim=-1)
