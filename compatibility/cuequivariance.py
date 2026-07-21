"""Narrow compatibility adapter for cuEquivariance's eager PyTorch path."""

from __future__ import annotations

import importlib
from types import ModuleType
import warnings

import torch


# cuEquivariance 0.10 creates a short-lived capability-probe temporary
# directory without retaining its context manager.  Its interpreter-shutdown
# warning is external to this project; keep the filter limited to tempfile's
# exact message so project ResourceWarnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"Implicitly cleaning up <TemporaryDirectory",
    category=ResourceWarning,
)


def load_cuequivariance() -> tuple[ModuleType, ModuleType]:
    """Import cuEquivariance after bridging a renamed private FX predicate.

    cuEquivariance 0.10 calls ``is_fx_symbolic_tracing`` while PyTorch 2.8
    exposes the same predicate as ``is_fx_tracing``.  The alias only restores
    that predicate name; it does not select a different kernel or fallback.
    """
    symbolic_trace = torch.fx._symbolic_trace
    if not hasattr(symbolic_trace, "is_fx_symbolic_tracing"):
        symbolic_trace.is_fx_symbolic_tracing = symbolic_trace.is_fx_tracing

    cue = importlib.import_module("cuequivariance")
    cuet = importlib.import_module("cuequivariance_torch")
    return cue, cuet
