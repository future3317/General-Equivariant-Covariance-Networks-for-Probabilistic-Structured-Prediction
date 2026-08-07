"""Unified plotting style for TPAMI paper figures.

This module centralizes fonts, colors, sizes and helper utilities so that
all result figures in the repository share the same publication-ready look.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Color palette.  The paper figures use the same restrained midnight-blue /
# champagne-gold language as the ICML predecessor.  The named entries are
# semantic so individual figure scripts do not need to carry local palettes.
# ---------------------------------------------------------------------------
COLORS = {
    "midnight_blue": "#002060",
    "champagne_gold": "#D4AF37",
    "champagne_light": "#F5D0A9",
    "navy_light": "#5B79A8",
    "dark_gray": "#3F4650",
    "gray": "#8C939D",
    "light_gray": "#D9DEE7",
}

# Backwards-compatible semantic aliases used by the older figure generators.
# They intentionally resolve to the same two-color system rather than to a
# second, unrelated palette.
COLORS.update(
    {
        "primary": COLORS["midnight_blue"],
        "secondary": COLORS["champagne_gold"],
        "tertiary": COLORS["navy_light"],
        "accent": COLORS["champagne_light"],
        # Publication-semantic aliases used by the refreshed figures.  The
        # legacy names above remain stable for older result generators.
        "blue_main": "#0F4D92",
        "blue_secondary": "#3775BA",
        "green_3": "#4F9F59",
        "red_2": "#B85C5A",
        "red_strong": "#B64342",
        "neutral": "#CFCECE",
        "teal": "#287F8C",
        "violet": "#7B4C9A",
    }
)

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "tpami_midnight_density",
    ["#F7F8FB", "#C9D4E6", "#5B79A8", "#002060"],
)

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "tpami_blue_gold_diverging",
    [COLORS["champagne_gold"], "#F7F8FB", COLORS["midnight_blue"]],
)

# Sequential palette for methods / lines
PALETTE = [
    COLORS["blue_main"],
    COLORS["green_3"],
    COLORS["red_strong"],
    COLORS["teal"],
    COLORS["violet"],
    COLORS["gray"],
]

# ---------------------------------------------------------------------------
# Default style parameters
# ---------------------------------------------------------------------------
DEFAULT_RC = {
    # Use a bundled sans-serif font so rendering is deterministic across the
    # local and server environments; retain mathtext for equations.
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    # Sizes suitable for IEEE TPAMI single- and double-column figures.
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    # Lines / markers
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.16,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    # Legend
    "legend.frameon": False,
    # Saving
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.compression": 9,
    "svg.fonttype": "none",
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
