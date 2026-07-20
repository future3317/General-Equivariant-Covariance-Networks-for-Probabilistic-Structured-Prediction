"""Explicit dataset-root configuration shared by loaders and CLI scripts."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "EQUIVCOMPILER_DATA_ROOT"


def data_root() -> Path:
    """Return the configured project dataset root.

    Dataset locations are intentionally not inferred from the checkout path:
    the Windows workstation and Linux server keep the same logical layout on
    different filesystems.
    """
    value = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"{DATA_ROOT_ENV} is not set; point it to the directory containing "
            "ITOP, modelnet40, mp_dielectric, and mp_elastic"
        )
    return Path(value).expanduser()


def dataset_dir(path: str | Path | None, name: str) -> Path:
    """Resolve an explicit dataset directory or one below the configured root."""
    if path is not None:
        return Path(path).expanduser()
    return data_root() / name
