"""Configure e3nn for PyTorch versions where TorchScript is deprecated.

e3nn 0.5.7 defaults to compiling generated FX graphs with
``torch.jit.script``.  PyTorch 2.12 deprecates that API and emits a warning for
every generated equivariant layer.  e3nn's supported eager code-generation
mode executes the same FX graphs without passing them through TorchScript.
"""

from __future__ import annotations

import warnings


_TORCHSCRIPT_DEPRECATION = (
    r"`torch\.jit\.script` is deprecated\. "
    r"Please switch to `torch\.compile` or `torch\.export`\."
)

# A few e3nn modules still use import-time TorchScript decorators.  Keep this
# suppression local to the upstream import; project warnings remain visible.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=_TORCHSCRIPT_DEPRECATION,
        category=DeprecationWarning,
    )
    import e3nn as _e3nn

    _e3nn.set_optimization_defaults(jit_mode="eager")

    from e3nn import o3
    from e3nn.io import CartesianTensor
    from e3nn.math import soft_one_hot_linspace
    from e3nn.nn import FullyConnectedNet, Gate


__all__ = [
    "CartesianTensor",
    "FullyConnectedNet",
    "Gate",
    "o3",
    "soft_one_hot_linspace",
]
