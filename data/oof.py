"""Shared deterministic split contract for cross-fitted dielectric runs."""

from __future__ import annotations

import torch


def fold_assignments(size: int, folds: int, seed: int) -> torch.Tensor:
    """Return stable split-local fold ids used by both training and OOF cache."""
    if size < 1 or folds < 2 or folds > size:
        raise ValueError("cross-fitting requires 2 <= folds <= dataset size")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(size, generator=generator)
    assignments = torch.empty(size, dtype=torch.long)
    assignments[permutation] = torch.arange(size, dtype=torch.long) % folds
    return assignments
