"""Layouts for packing repeated copies of an O(3) representation."""

from __future__ import annotations

import torch
from compatibility.e3nn import o3


class RepeatedIrrepLayout:
    """Describe and pack ``copies`` copies of one representation.

    e3nn stores multiplicities grouped by irrep type.  Covariance factors and
    graph potentials instead need a leading copy axis.  This object owns the
    single conversion between those layouts and validates its input dimension.
    """

    def __init__(self, irreps: o3.Irreps | str, copies: int):
        self.irreps = o3.Irreps(irreps)
        self.copies = int(copies)
        if self.copies < 1:
            raise ValueError("copies must be positive")
        if self.irreps.dim < 1:
            raise ValueError("irreps must not be empty")

        self.expanded_irreps = o3.Irreps(
            [(multiplicity * self.copies, irrep) for multiplicity, irrep in self.irreps]
        )
        source_slices: list[list[slice]] = [[] for _ in range(self.copies)]
        cursor = 0
        for multiplicity, irrep in self.irreps:
            for repeated_copy in range(multiplicity * self.copies):
                start = cursor + repeated_copy * irrep.dim
                source_slices[repeated_copy % self.copies].append(
                    slice(start, start + irrep.dim)
                )
            cursor += multiplicity * self.copies * irrep.dim
        self._source_slices = tuple(tuple(copy_slices) for copy_slices in source_slices)

    def pack(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Return coefficients as ``(..., copies, irreps.dim)``."""
        if coefficients.shape[-1] != self.expanded_irreps.dim:
            raise ValueError(
                f"coefficients last dim {coefficients.shape[-1]} != "
                f"expanded irreps dim {self.expanded_irreps.dim}"
            )
        return torch.stack(
            [
                torch.cat(
                    [coefficients[..., source] for source in copy_slices],
                    dim=-1,
                )
                for copy_slices in self._source_slices
            ],
            dim=-2,
        )
