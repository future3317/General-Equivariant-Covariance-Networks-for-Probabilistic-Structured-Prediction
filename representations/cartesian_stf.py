"""Exact STF-coordinate/dense-projector execution for rank-2 covariance heads.

The public compiler remains basis agnostic.  This module is its specialized
execution backend for ``V = 0e + 2e`` and seed features containing only
``0e``, ``1o`` and ``2e`` channels.  It replaces the runtime Clebsch--Gordan
tensor square by dense multiplicity contractions followed by small frozen
projectors in e3nn's orthonormal real-irrep coordinates.  Cartesian STF2
coordinates provide the exact interpretation and operator bijection; the hot
path does not materialize fourth-order Cartesian tensors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from compatibility.e3nn import CartesianTensor, o3

from representations.symmetric_square import O3SymmetricOperatorBasis


RANK2_OUTPUT_IRREPS = o3.Irreps("1x0e + 1x2e")
RANK2_OPERATOR_IRREPS = o3.Irreps("2x0e + 2x2e + 1x4e")
_SUPPORTED_SEED_TYPES = {o3.Irrep("0e"), o3.Irrep("1o"), o3.Irrep("2e")}


def is_rank2_stf_output(irreps: o3.Irreps | str) -> bool:
    """Return whether ``irreps`` is exactly one ``0e`` plus one ``2e``."""
    counts: dict[o3.Irrep, int] = {}
    for multiplicity, irrep in o3.Irreps(irreps):
        counts[irrep] = counts.get(irrep, 0) + multiplicity
    return counts == {o3.Irrep("0e"): 1, o3.Irrep("2e"): 1}


def supports_cartesian_stf_seed(irreps: o3.Irreps | str) -> bool:
    """Return whether every seed irrep has an implemented Cartesian path."""
    seed = o3.Irreps(irreps)
    return bool(seed) and all(irrep in _SUPPORTED_SEED_TYPES for _, irrep in seed)


def _normalized_projector(left_l: int, right_l: int, output_l: int) -> torch.Tensor:
    """Build a row-orthonormal CG projector in e3nn real-irrep coordinates."""
    coupling = o3.wigner_3j(left_l, right_l, output_l, dtype=torch.float64)
    return math.sqrt(2 * output_l + 1) * coupling.permute(2, 0, 1).contiguous()


class Rank2CartesianSTFOperatorBasis(torch.nn.Module):
    r"""The explicit ``(a,b,P,Q,H)`` realization of ``Sym(0e + 2e)``.

    In the orthonormal Cartesian STF2 basis, the canonical components assemble
    the operator

    .. math::
        A(s,T) = (a s + \langle Q,T\rangle,
        sQ + bT + \Pi_2(PT+TP) + H:T).

    The component order is ``a, b, P[5], Q[5], H[9]``.  A fixed invertible
    change of basis maps e3nn's reduced-tensor-product coefficients to these
    components, so ``assemble`` is numerically identical to
    :class:`O3SymmetricOperatorBasis` while exposing the Cartesian formula.
    No dense fourth-order Cartesian tensor is stored.
    """

    component_slices = {
        "a": slice(0, 1),
        "b": slice(1, 2),
        "P": slice(2, 7),
        "Q": slice(7, 12),
        "H": slice(12, 21),
    }

    def __init__(self):
        super().__init__()
        self.output_irreps = RANK2_OUTPUT_IRREPS
        self._operator_irreps = RANK2_OPERATOR_IRREPS
        self._output_dim = 6
        self._operator_dim = 21

        cartesian = CartesianTensor("ij=ji")
        coordinate_basis = torch.eye(6, dtype=torch.float64)
        stf2_basis = torch.stack(
            [cartesian.to_cartesian(vector) for vector in coordinate_basis[1:]]
        )
        projector_l0 = _normalized_projector(2, 2, 0).flatten(1)
        projector_l2 = _normalized_projector(2, 2, 2).flatten(1)
        projector_l4 = _normalized_projector(2, 2, 4).flatten(1)

        canonical_basis = self._build_canonical_basis(
            cartesian, stf2_basis, projector_l4.reshape(9, 5, 5)
        )
        spherical = O3SymmetricOperatorBasis(RANK2_OUTPUT_IRREPS)
        spherical_basis = spherical.basis.to(dtype=torch.float64)
        components_to_irreps = torch.einsum(
            "qij,cij->qc", spherical_basis, canonical_basis
        )
        irreps_to_components = torch.linalg.inv(components_to_irreps)
        module_dtype = torch.get_default_dtype()
        stf2_basis = stf2_basis.to(dtype=module_dtype)
        projector_l0 = projector_l0.to(dtype=module_dtype)
        projector_l2 = projector_l2.to(dtype=module_dtype)
        projector_l4 = projector_l4.to(dtype=module_dtype)
        canonical_basis = canonical_basis.to(dtype=module_dtype)
        spherical_basis = spherical_basis.to(dtype=module_dtype)
        components_to_irreps = components_to_irreps.to(dtype=module_dtype)
        irreps_to_components = irreps_to_components.to(dtype=module_dtype)

        # ``_basis`` intentionally matches O3SymmetricOperatorBasis' state key,
        # so existing spherical-head checkpoints remain strictly loadable.
        self.register_buffer("_basis", spherical_basis)
        self.register_buffer("stf2_basis", stf2_basis, persistent=False)
        self.register_buffer("P0", projector_l0, persistent=False)
        self.register_buffer("P2", projector_l2, persistent=False)
        self.register_buffer("P4", projector_l4, persistent=False)
        self.register_buffer("_canonical_basis", canonical_basis, persistent=False)
        self.register_buffer(
            "_components_to_irreps", components_to_irreps, persistent=False
        )
        self.register_buffer(
            "_irreps_to_components", irreps_to_components, persistent=False
        )

    @staticmethod
    def _build_canonical_basis(
        cartesian: CartesianTensor,
        stf2_basis: torch.Tensor,
        projector_l4: torch.Tensor,
    ) -> torch.Tensor:
        basis: list[torch.Tensor] = []
        scalar = torch.zeros(6, 6, dtype=torch.float64)
        scalar[0, 0] = 1.0
        basis.append(scalar)

        identity = torch.zeros(6, 6, dtype=torch.float64)
        identity[1:, 1:] = torch.eye(5, dtype=torch.float64)
        basis.append(identity)

        eye3 = torch.eye(3, dtype=torch.float64)
        for P in stf2_basis:
            columns = []
            for T in stf2_basis:
                product = P @ T + T @ P
                product = product - torch.trace(product) * eye3 / 3.0
                columns.append(cartesian.from_cartesian(product)[1:])
            operator = torch.zeros(6, 6, dtype=torch.float64)
            operator[1:, 1:] = torch.stack(columns, dim=1)
            basis.append(operator)

        for index in range(5):
            operator = torch.zeros(6, 6, dtype=torch.float64)
            operator[0, index + 1] = 1.0
            operator[index + 1, 0] = 1.0
            basis.append(operator)

        for contraction in projector_l4:
            operator = torch.zeros(6, 6, dtype=torch.float64)
            operator[1:, 1:] = contraction
            basis.append(operator)
        return torch.stack(basis)

    @property
    def operator_irreps(self) -> o3.Irreps:
        return self._operator_irreps

    @property
    def operator_dim(self) -> int:
        return self._operator_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def basis(self) -> torch.Tensor:
        """Return the spherical coefficient basis for API compatibility."""
        return self._basis

    @property
    def canonical_basis(self) -> torch.Tensor:
        return self._canonical_basis

    def components_from_irreps(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[-1] != self._operator_dim:
            raise ValueError(
                f"expected {self._operator_dim} operator coefficients, "
                f"got {coefficients.shape[-1]}"
            )
        return torch.einsum("...q,cq->...c", coefficients, self._irreps_to_components)

    def irreps_from_components(self, components: torch.Tensor) -> torch.Tensor:
        if components.shape[-1] != self._operator_dim:
            raise ValueError(
                f"expected {self._operator_dim} Cartesian components, "
                f"got {components.shape[-1]}"
            )
        return torch.einsum("...c,qc->...q", components, self._components_to_irreps)

    def assemble_components(self, components: torch.Tensor) -> torch.Tensor:
        """Assemble the explicit ``(a,b,P,Q,H)`` operator."""
        if components.shape[-1] != self._operator_dim:
            raise ValueError(
                f"expected {self._operator_dim} Cartesian components, "
                f"got {components.shape[-1]}"
            )
        operator = torch.einsum("...c,cij->...ij", components, self._canonical_basis)
        return 0.5 * (operator + operator.transpose(-1, -2))

    def assemble(self, coefficients: torch.Tensor) -> torch.Tensor:
        # Use the orthonormal spherical basis for the numerical fast path.  The
        # explicit Cartesian route above is the same linear map but incurs two
        # extra small basis changes in float32.
        operator = torch.einsum("...q,qij->...ij", coefficients, self._basis)
        return 0.5 * (operator + operator.transpose(-1, -2))

    def project(self, operator: torch.Tensor) -> torch.Tensor:
        if operator.shape[-2:] != (self._output_dim, self._output_dim):
            raise ValueError(
                f"operator shape {operator.shape[-2:]} != "
                f"({self._output_dim}, {self._output_dim})"
            )
        return torch.einsum("...ij,qij->...q", operator, self._basis)


class MultiplicityFirstCartesianTensorSquare(torch.nn.Module):
    r"""Exact multiplicity-first dense-projector square for the rank-2 target.

    With ``contraction_rank=None`` the learnable flat weight has exactly the
    same layout and normalization as :class:`e3nn.o3.TensorSquare`.  Self paths
    are assembled into symmetric channel matrices and evaluated as ``W=I W``;
    cross paths use the smaller multiplicity as their exact factorization rank.
    Thus channel mixing is scheduled before each frozen angular projector.

    A positive ``contraction_rank`` selects a CP-factorized path.  It is only
    accepted when it is smaller than at least one exact path rank and is
    explicitly marked non-equivalent by ``is_exact=False``.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps | str,
        irreps_out: o3.Irreps | str = RANK2_OPERATOR_IRREPS,
        *,
        contraction_rank: int | None = None,
    ):
        super().__init__()
        if not supports_cartesian_stf_seed(irreps_in):
            raise ValueError(
                "Cartesian-STF tensor square supports only 0e, 1o and 2e "
                f"seed channels, got {o3.Irreps(irreps_in)}"
            )
        self.irreps_in = o3.Irreps(irreps_in).simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        if any(
            irrep not in {o3.Irrep("0e"), o3.Irrep("2e"), o3.Irrep("4e")}
            for _, irrep in self.irreps_out
        ):
            raise ValueError("Cartesian-STF tensor square outputs must be 0e, 2e or 4e")

        reference = o3.TensorSquare(self.irreps_in, irreps_out=self.irreps_out)
        self._instruction_paths = self._collect_paths(reference.instructions)
        self.max_exact_rank = max(
            path["exact_rank"] for path in self._instruction_paths
        )
        if contraction_rank is not None and contraction_rank < 1:
            raise ValueError("contraction_rank must be positive")
        if contraction_rank is not None and contraction_rank >= self.max_exact_rank:
            contraction_rank = None
        self.contraction_rank = contraction_rank
        self.is_exact = contraction_rank is None

        module_dtype = torch.get_default_dtype()
        self.register_buffer(
            "P0",
            _normalized_projector(2, 2, 0).flatten(1).to(dtype=module_dtype),
            persistent=False,
        )
        self.register_buffer(
            "P2",
            _normalized_projector(2, 2, 2).flatten(1).to(dtype=module_dtype),
            persistent=False,
        )
        self.register_buffer(
            "P4",
            _normalized_projector(2, 2, 4).flatten(1).to(dtype=module_dtype),
            persistent=False,
        )
        self._register_couplings()
        self._register_pair_bases()

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
            for path in self._instruction_paths:
                left_mul = self.irreps_in[path["i_in1"]].mul
                right_mul = self.irreps_in[path["i_in2"]].mul
                out_mul = self.irreps_out[path["i_out"]].mul
                rank = min(self.contraction_rank, path["exact_rank"])
                left = torch.empty(out_mul, rank, left_mul)
                right = torch.empty(out_mul, rank, right_mul)
                torch.nn.init.normal_(left, std=1.0 / math.sqrt(left_mul * rank))
                torch.nn.init.normal_(right, std=1.0 / math.sqrt(right_mul * rank))
                self.left_factors.append(torch.nn.Parameter(left))
                self.right_factors.append(torch.nn.Parameter(right))
        del reference

    def _collect_paths(self, instructions: Sequence) -> list[dict]:
        paths: dict[tuple[int, int, int], dict] = {}
        offset = 0
        for instruction in instructions:
            size = math.prod(instruction.path_shape)
            key = (
                instruction.i_in1,
                instruction.i_in2,
                instruction.i_out,
            )
            path = paths.setdefault(
                key,
                {
                    "i_in1": instruction.i_in1,
                    "i_in2": instruction.i_in2,
                    "i_out": instruction.i_out,
                },
            )
            path[instruction.connection_mode] = (
                offset,
                offset + size,
                tuple(instruction.path_shape),
                float(instruction.path_weight),
            )
            offset += size
        collected = []
        for path in paths.values():
            left_mul = self.irreps_in[path["i_in1"]].mul
            right_mul = self.irreps_in[path["i_in2"]].mul
            path["exact_rank"] = (
                left_mul if path["i_in1"] == path["i_in2"] else min(left_mul, right_mul)
            )
            collected.append(path)
        return collected

    def _register_couplings(self) -> None:
        keys = {
            (
                self.irreps_in[path["i_in1"]].ir.l,
                self.irreps_in[path["i_in2"]].ir.l,
                self.irreps_out[path["i_out"]].ir.l,
            )
            for path in self._instruction_paths
        }
        for left_l, right_l, output_l in keys:
            name = f"_coupling_{left_l}_{right_l}_{output_l}"
            normalized = _normalized_projector(left_l, right_l, output_l)
            raw = (normalized / math.sqrt(2 * output_l + 1)).to(
                dtype=torch.get_default_dtype()
            )
            self.register_buffer(name, raw, persistent=False)

    def _register_pair_bases(self) -> None:
        multiplicities = {
            self.irreps_in[path["i_in1"]].mul
            for path in self._instruction_paths
            if path["i_in1"] == path["i_in2"] and "u<vw" in path
        }
        for multiplicity in multiplicities:
            indices = torch.triu_indices(multiplicity, multiplicity, offset=1)
            pair_basis = torch.zeros(
                indices.shape[1],
                multiplicity,
                multiplicity,
                dtype=torch.get_default_dtype(),
            )
            if indices.shape[1] > 0:
                pair_ids = torch.arange(indices.shape[1])
                pair_basis[pair_ids, indices[0], indices[1]] = 0.5
                pair_basis[pair_ids, indices[1], indices[0]] = 0.5
            self.register_buffer(
                f"_pair_basis_{multiplicity}", pair_basis, persistent=False
            )

    def _coupling(self, path: dict) -> torch.Tensor:
        left_l = self.irreps_in[path["i_in1"]].ir.l
        right_l = self.irreps_in[path["i_in2"]].ir.l
        output_l = self.irreps_out[path["i_out"]].ir.l
        return getattr(self, f"_coupling_{left_l}_{right_l}_{output_l}")

    def _weight_view(self, specification: tuple) -> tuple[torch.Tensor, float]:
        start, stop, shape, path_weight = specification
        return self.weight[start:stop].reshape(shape), path_weight

    def load_e3nn_weights(self, tensor_square: torch.nn.Module) -> None:
        """Copy a spherical-CG TensorSquare weight without conversion loss."""
        if not self.is_exact:
            raise RuntimeError("truncated contraction has no exact e3nn weight map")
        if o3.Irreps(tensor_square.irreps_in) != self.irreps_in:
            raise ValueError("TensorSquare input irreps do not match")
        if o3.Irreps(tensor_square.irreps_out) != self.irreps_out:
            raise ValueError("TensorSquare output irreps do not match")
        if tensor_square.weight.numel() != self.weight.numel():
            raise ValueError("TensorSquare weight layout does not match")
        with torch.no_grad():
            self.weight.copy_(tensor_square.weight)

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
        """Accept deterministic code-generation buffers from e3nn checkpoints."""
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

    def _exact_path(self, path: dict, inputs: list[torch.Tensor]) -> torch.Tensor:
        left = inputs[path["i_in1"]]
        right = inputs[path["i_in2"]]
        coupling = self._coupling(path)
        if path["i_in1"] != path["i_in2"]:
            weights, path_weight = self._weight_view(path["uvw"])
            mixed = torch.einsum("uvw,...vj->...wuj", weights * path_weight, right)
            return self._project_exact(left, mixed, coupling, path)

        diagonal, diagonal_scale = self._weight_view(path["uuw"])
        channel_matrix = torch.diag_embed((diagonal * diagonal_scale).transpose(0, 1))
        if "u<vw" in path:
            off_diagonal, off_scale = self._weight_view(path["u<vw"])
            pair_basis = getattr(self, f"_pair_basis_{left.shape[-2]}").to(
                dtype=off_diagonal.dtype
            )
            channel_matrix = channel_matrix + torch.einsum(
                "pw,puv->wuv", off_diagonal * off_scale, pair_basis
            )
        mixed = torch.einsum("wuv,...vj->...wuj", channel_matrix, right)
        return self._project_exact(left, mixed, coupling, path)

    def _project_exact(
        self,
        left: torch.Tensor,
        mixed: torch.Tensor,
        coupling: torch.Tensor,
        path: dict,
    ) -> torch.Tensor:
        """Apply one angular map after exact multiplicity contraction."""
        left_l = self.irreps_in[path["i_in1"]].ir.l
        right_l = self.irreps_in[path["i_in2"]].ir.l
        output_l = self.irreps_out[path["i_out"]].ir.l
        if left_l == 0:
            # The only supported cross path is 0 x 2 -> 2.  e3nn's component
            # normalization contributes 1/sqrt(2l+1).
            return (left[..., :, 0].unsqueeze(-2).unsqueeze(-1) * mixed).sum(
                dim=-2
            ) / math.sqrt(2 * output_l + 1)
        if output_l == 0 and left_l == right_l:
            return (left.unsqueeze(-3) * mixed).sum(dim=(-2, -1)).unsqueeze(
                -1
            ) / math.sqrt(2 * left_l + 1)
        pair = (left.unsqueeze(-3).unsqueeze(-1) * mixed.unsqueeze(-2)).flatten(-2)
        return torch.matmul(pair, coupling.flatten(1).T).sum(dim=-2)

    def _factorized_path(
        self, path_index: int, path: dict, inputs: list[torch.Tensor]
    ) -> torch.Tensor:
        left = inputs[path["i_in1"]]
        right = inputs[path["i_in2"]]
        left_mixed = torch.einsum(
            "wru,...ui->...wri", self.left_factors[path_index], left
        )
        right_mixed = torch.einsum(
            "wrv,...vj->...wrj", self.right_factors[path_index], right
        )
        coupling = self._coupling(path)
        pair = (left_mixed.unsqueeze(-1) * right_mixed.unsqueeze(-2)).flatten(-2)
        projected = torch.matmul(pair, coupling.flatten(1).T)
        return projected.sum(dim=-2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"expected input dimension {self.irreps_in.dim}, "
                f"got {features.shape[-1]}"
            )
        inputs = [
            features[..., irrep_slice].reshape(
                *features.shape[:-1], multiplicity, irrep.dim
            )
            for irrep_slice, (multiplicity, irrep) in zip(
                self.irreps_in.slices(), self.irreps_in
            )
        ]
        outputs: list[torch.Tensor | None] = [None] * len(self.irreps_out)
        for path_index, path in enumerate(self._instruction_paths):
            contribution = (
                self._exact_path(path, inputs)
                if self.is_exact
                else self._factorized_path(path_index, path, inputs)
            )
            output_index = path["i_out"]
            previous = outputs[output_index]
            outputs[output_index] = (
                contribution if previous is None else previous + contribution
            )
        if any(output is None for output in outputs):
            raise RuntimeError("an STF output irrep has no admissible input path")
        return torch.cat(
            [output.flatten(start_dim=-2) for output in outputs if output is not None],
            dim=-1,
        )

    def extra_repr(self) -> str:
        mode = "exact" if self.is_exact else f"truncated_rank={self.contraction_rank}"
        return f"{self.irreps_in} -> {self.irreps_out}, {mode}"
