"""Warning-clean imports for PyTorch Geometric on PyTorch 2.12.

The installed ``torch_sparse`` extension still defines a small set of helpers
with import-time ``torch.jit.script`` decorators.  They remain functional, but
PyTorch 2.12 emits a deprecation warning while importing them.  Restrict the
suppression to this third-party import so warnings from project code continue
to fail strict test runs.
"""

from __future__ import annotations

import warnings


_TORCHSCRIPT_DEPRECATION = (
    r"`torch\.jit\.script` is deprecated\. "
    r"Please switch to `torch\.compile` or `torch\.export`\."
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=_TORCHSCRIPT_DEPRECATION,
        category=DeprecationWarning,
    )
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader


__all__ = ["Data", "PyGDataLoader"]
