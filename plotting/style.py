"""Unified plotting style for TPAMI paper figures.

This module centralizes fonts, colors, sizes and helper utilities so that
all result figures in the repository share the same publication-ready look.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Color palette (soft, color-blind friendly, consistent with ICML figures)
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#E07A5F",  # muted coral / salmon
    "secondary": "#81B29A",  # sage green
    "tertiary": "#3D405B",  # dark slate
    "accent": "#F2CC8F",  # warm yellow
    "teal": "#5FBDBD",  # soft teal
    "purple": "#9D8DF1",  # light purple
    "olive": "#A3A847",  # olive green
    "gray": "#A8A8A8",  # neutral gray
    "light_gray": "#D9D9D9",  # light gray
    "dark_gray": "#4A4A4A",  # dark gray
}

# Sequential palette for methods / lines
PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["teal"],
    COLORS["olive"],
    COLORS["purple"],
    COLORS["accent"],
    COLORS["tertiary"],
    COLORS["gray"],
]

# ---------------------------------------------------------------------------
# Default style parameters
# ---------------------------------------------------------------------------
DEFAULT_RC = {
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    # Sizes suitable for IEEE TPAMI single- and double-column figures.
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    # Lines / markers
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    # Saving
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.compression": 9,
}


def setup_tpami_style() -> None:
    """Apply the unified IEEE TPAMI style to matplotlib."""
    plt.style.use("default")
    rcParams.update(DEFAULT_RC)


def get_color(index: int) -> str:
    """Return a color from the cyclic palette."""
    return PALETTE[index % len(PALETTE)]


def label_panels(
    axes,
    labels: Sequence[str] | None = None,
    x: float = -0.18,
    y: float = 1.05,
    fontsize: int = 14,
    fontweight: str = "bold",
) -> None:
    """Add (a), (b), ... labels to a sequence of axes.

    Parameters
    ----------
    axes : iterable of Axes
    labels : sequence of str, optional
        Custom labels. If None, uses lowercase letters.
    x, y : float
        Position in axes coordinates.
    """
    if labels is None:
        labels = [f"({s})" for s in string.ascii_lowercase]
    for ax, label in zip(axes, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            va="bottom",
            ha="right",
        )


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int | None = None,
) -> None:
    """Save a figure in multiple formats.

    The output directory is created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, format=fmt)


def cm2inch(*values: float) -> tuple[float, ...]:
    """Convert centimeters to inches for figure sizing."""
    return tuple(v / 2.54 for v in values)
