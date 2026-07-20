"""Test-wide runtime initialization."""

# Configure e3nn before test modules import it directly.  This keeps PyTorch
# 2.12's deprecated TorchScript path out of module construction.
from compatibility import e3nn as _e3nn_runtime  # noqa: F401
from compatibility import torch_geometric as _pyg_runtime  # noqa: F401
