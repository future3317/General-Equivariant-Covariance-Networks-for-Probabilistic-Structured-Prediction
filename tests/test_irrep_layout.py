"""Tests for the shared repeated-irrep packing layout."""

import pytest
import torch

from representations.irrep_layout import RepeatedIrrepLayout


def test_repeated_irrep_layout_packs_copy_axis():
    layout = RepeatedIrrepLayout("2x0e + 1x1o", copies=3)
    coefficients = torch.arange(
        2 * layout.expanded_irreps.dim, dtype=torch.float64
    ).reshape(2, -1)
    packed = layout.pack(coefficients)
    assert packed.shape == (2, 3, layout.irreps.dim)

    unpacked_columns = packed.transpose(-1, -2)
    assert unpacked_columns.shape == (2, layout.irreps.dim, 3)


def test_repeated_irrep_layout_validates_copy_count_and_dimension():
    with pytest.raises(ValueError, match="copies"):
        RepeatedIrrepLayout("0e", copies=0)
    layout = RepeatedIrrepLayout("0e + 1o", copies=2)
    with pytest.raises(ValueError, match="last dim"):
        layout.pack(torch.randn(4, layout.expanded_irreps.dim + 1))
