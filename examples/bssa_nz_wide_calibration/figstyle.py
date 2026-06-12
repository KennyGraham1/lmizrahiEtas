"""Shared publication-quality figure style for the BSSA manuscript figures.

Imported by analyze_sweep.py (figures 1-4) and analyze_experiments.py
(figures 5-8) so every panel uses one consistent palette, font, and layout.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Neutral ink colors.
NAVY = "#102A43"
TEXT = "#2A3F54"
GRID = "#DCE3EA"
SUBTLE = "#5C7185"

# Scenario order and human-readable labels (LaTeX-safe).
SCENARIO_ORDER = [
    "baseline", "low_mu_k0", "high_mu_k0", "mc_4p3", "mc_4p5",
    "window_1980", "window_2000",
]
DISPLAY_NAMES = {
    "baseline": "1960 baseline",
    "low_mu_k0": "1960 low init.",
    "high_mu_k0": "1960 high init.",
    "mc_4p3": r"$M_c$ 4.3",
    "mc_4p5": r"$M_c$ 4.5",
    "window_1980": "1980 window",
    "window_2000": "2000 window",
}

# Colorblind-aware, consistent scenario palette.
COLORS = {
    "baseline": "#2C6FB0",     # blue
    "low_mu_k0": "#3FA7A2",    # teal
    "high_mu_k0": "#5AA86B",   # green
    "mc_4p3": "#E08A2B",       # orange
    "mc_4p5": "#C9A227",       # gold
    "window_1980": "#8E5BA6",  # purple
    "window_2000": "#D1495B",  # red
}
# Training-window colors (figure 5), consistent with the scenario palette.
WINDOW_COLORS = {"1960": "#2C6FB0", "1980": "#8E5BA6", "2000": "#D1495B"}

OBSERVED = "#1A1A1A"
REFERENCE_LINE = "#33475B"


def apply() -> None:
    """Apply the shared rcParams. Call once at the top of each figure script."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.edgecolor": "#9FB3C8",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": NAVY,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 400,
        "figure.dpi": 150,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.35,
    })


def panel_label(ax, text: str, pad: int = 7) -> None:
    """Left-aligned bold navy panel title, e.g. '(a) Point estimates'."""
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold",
                 color=NAVY, pad=pad)


def scenario_legend(fig, ncol: int = 4, y: float = -0.02):
    """A single shared scenario legend below the panels."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=COLORS[s], marker="o", linestyle="-",
               markersize=5, linewidth=1.8, label=DISPLAY_NAMES[s])
        for s in SCENARIO_ORDER
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=ncol,
                     frameon=False, bbox_to_anchor=(0.5, y),
                     handletextpad=0.5, columnspacing=1.3)
    return leg
