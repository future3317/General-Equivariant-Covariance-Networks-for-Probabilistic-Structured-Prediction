"""Test-wide runtime initialization."""

import os
import sys

if sys.platform == "win32":
    # The local egnn environment can load both Intel and LLVM OpenMP runtimes
    # through its scientific stack.  Keep this compatibility workaround local
    # to the test process so exact SciPy diagnostics do not abort the suite.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Configure e3nn before test modules import it directly.  This keeps PyTorch
# 2.12's deprecated TorchScript path out of module construction.
from compatibility import e3nn as _e3nn_runtime  # noqa: E402,F401
from compatibility import torch_geometric as _pyg_runtime  # noqa: E402,F401
