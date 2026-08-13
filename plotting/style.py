"""Unified plotting style for TPAMI paper figures.

This module centralizes fonts, colors, sizes and helper utilities so that
all result figures in the repository share the same publication-ready look.

The baseline follows the Academic Figure Skill Nature/Cell/Science guidelines:
- Arial/Helvetica sans-serif, 8 pt base, 7 pt tick labels
- Clean left/bottom spines, no top/right spines
- Restrained, print-safe semantic palette
- Vector-first PDF export with TrueType font embedding
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
# ---------------------------------------------------------------------------
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# ---------------------------------------------------------------------------
# Named color aliases used by the figure-generation scripts.
# All aliases resolve into the restrained palette above so the repository
# remains color-consistent across panels.
# ---------------------------------------------------------------------------
COLORS = {
    "midnight_blue": CATEGORICAL[0],   # primary reference / control
    "champagne_gold": CATEGORICAL[3],  # secondary / comparison
    "champagne_light": "#E0E0E0",      # light fills / confidence bands
    "navy_light": CATEGORICAL_EXTENDED[6],
    "dark_gray": GREY,
    "gray": "#BBBBBB",
    "light_gray": "#E8E8E8",
}

# Backwards-compatible semantic aliases used by older figure generators.
COLORS.update(
    {
        "primary": COLORS["midnight_blue"],
        "secondary": COLORS["champagne_gold"],
        "tertiary": COLORS["navy_light"],
        "accent": COLORS["champagne_light"],
        # Publication-semantic aliases used by the refreshed figures.
        "blue_main": CATEGORICAL[0],
        "blue_secondary": CATEGORICAL_EXTENDED[6],
        "green_3": CATEGORICAL[2],
        "red_2": CATEGORICAL_EXTENDED[7],
        "red_strong": CATEGORICAL[1],
        "neutral": "#CFCECE",
        "teal": CATEGORICAL[3],
        "violet": CATEGORICAL[4],
    }
)

# Explicit semantic aliases used by manuscript-level comparison figures.
# Keeping these names separate from legacy aliases prevents a renderer from
# accidentally assigning a new color when the number of arms changes.
FAMILY_COLORS = {
    "full": CATEGORICAL[4],
    "low_rank": CATEGORICAL[3],
    "block": CATEGORICAL[2],
    "isotropic": CATEGORICAL[5],
    "graph": CATEGORICAL[2],
}
LAW_COLORS = {
    "gaussian": CATEGORICAL[5],
    "student_t": CATEGORICAL[1],
}

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "cns_sequential_density",
    SEQUENTIAL,
)

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "cns_blue_white_red",
    DIVERGING,
)

# Sequential palette for methods / lines, following CNS order.
PALETTE = [
    COLORS["blue_main"],
    COLORS["red_strong"],
    COLORS["green_3"],
    COLORS["teal"],
    COLORS["violet"],
    COLORS["gray"],
]

# ---------------------------------------------------------------------------
# Academic Figure Skill Typography Baseline -- COPY VERBATIM
# ---------------------------------------------------------------------------
DEFAULT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.frameon": False,
    # Export
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.05,
    "figure.dpi": 300,
    # Grid: off by default; when enabled explicitly, keep it subtle
    "axes.grid": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.3,
    "grid.color": "#E0E0E0",
    "axes.axisbelow": True,
}


def setup_tpami_style() -> None:
    """Apply the unified IEEE TPAMI / CNS publication style to matplotlib."""
    plt.style.use("default")
    rcParams.update(DEFAULT_RC)


def get_color(index: int) -> str:
    """Return a color from the cyclic palette."""
    return PALETTE[index % len(PALETTE)]


def label_panels(
    axes,
    labels: Sequence[str] | None = None,
    x: float = -0.12,
    y: float = 1.04,
    fontsize: int = 9,
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

    The output directory is created automatically. PNGs are rendered at the
    configured figure DPI (300); PDFs keep text as editable TrueType fonts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, format=fmt)


def cm2inch(*values: float) -> tuple[float, ...]:
    """Convert centimeters to inches for figure sizing."""
    return tuple(v / 2.54 for v in values)
