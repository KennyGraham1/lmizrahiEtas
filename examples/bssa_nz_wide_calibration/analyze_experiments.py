#!/usr/bin/env python3
"""Tables and figures for the four follow-up experiments.

Consumes the CSVs written by ``experiments/*.py`` and produces manuscript-ready
figures (figure5--figure8) plus ``tables/key_results_experiments.json``. Each
section is skipped with a warning if its input table is missing, so the script
can be run incrementally as the experiments complete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLE_DIR = HERE / "tables"
FIGURE_DIR = HERE / "figures"
SIM_ROOT = HERE.parents[1] / "examples" / "simulations_nz_wide"

import sys as _sys
if str(HERE) not in _sys.path:
    _sys.path.insert(0, str(HERE))
import figstyle

WINDOW_COLORS = figstyle.WINDOW_COLORS
SCEN_ORDER = figstyle.SCENARIO_ORDER
SCEN_LABEL = figstyle.DISPLAY_NAMES
SCEN_COLOR = figstyle.COLORS


def setup():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    figstyle.apply()


def _save(fig, stem):
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_multi_origin(key):
    path = TABLE_DIR / "multi_origin_branching.csv"
    if not path.exists():
        print("skip multi_origin (no table)"); return
    df = pd.read_csv(path)
    df["window_start"] = df["window_start"].astype(str)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axhspan(1.0, df["branching_ratio"].max() + 0.005, color="#F2D7D5", alpha=0.35,
               zorder=0)
    ax.axhspan(df["branching_ratio"].min() - 0.005, 1.0, color="#D6E4F0", alpha=0.35,
               zorder=0)
    for window in ["1960", "1980", "2000"]:
        sub = df[df["window_start"] == window].sort_values("origin")
        if sub.empty:
            continue
        years = sub["origin"].str[:4].astype(int)
        ax.plot(years, sub["branching_ratio"], "o-", color=WINDOW_COLORS[window],
                label=f"{window} start", linewidth=2.0, markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    ax.axhline(1.0, color=figstyle.REFERENCE_LINE, linestyle="--", linewidth=1.2,
               zorder=3)
    ax.text(2013.0, 1.001, "supercritical", fontsize=8, color="#A03A48",
            style="italic", va="bottom")
    ax.text(2013.0, 0.999, "subcritical", fontsize=8, color="#3B6EA5",
            style="italic", va="top")
    ax.set_xlabel("Forecast origin (year)")
    ax.set_ylabel(r"Fitted branching ratio $n$")
    ax.set_xticks([2013, 2015, 2017, 2019, 2021])
    figstyle.panel_label(ax, "Branching ratio vs training window, across origins")
    ax.legend(title="Training start", frameon=False, loc="center right")
    _save(fig, "figure5_multi_origin")

    # Key results.
    rec = {}
    for window in ["1960", "1980", "2000"]:
        sub = df[df["window_start"] == window]
        rec[f"window_{window}_branching_by_origin"] = {
            r.origin[:4]: float(r.branching_ratio) for r in sub.itertuples()
        }
        rec[f"window_{window}_all_supercritical"] = bool((sub["branching_ratio"] >= 1).all())
        rec[f"window_{window}_all_subcritical"] = bool((sub["branching_ratio"] < 1).all())
    rec["n_origins"] = int(df["origin"].nunique())
    key["multi_origin"] = rec
    print(f"figure5_multi_origin: {df['origin'].nunique()} origins x 3 windows")


def fig_branching_uncertainty(key):
    path = TABLE_DIR / "branching_uncertainty.csv"
    if not path.exists():
        print("skip branching_uncertainty (no table)"); return
    df = pd.read_csv(path)
    df["__o"] = df["scenario"].map({s: i for i, s in enumerate(SCEN_ORDER)})
    df = df.sort_values("__o")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(len(df))
    colors = [SCEN_COLOR[s] for s in df["scenario"]]
    xhi = float((df["branching_ratio"] + 1.96 * df["se_branching_ratio"]).max())
    xlo = float((df["branching_ratio"] - 1.96 * df["se_branching_ratio"]).min())
    ax.axvspan(1.0, xhi + 0.004, color="#F2D7D5", alpha=0.4, zorder=0)
    ax.text(1.001, len(df) - 0.4, "supercritical", fontsize=8, color="#A03A48",
            style="italic", ha="left", va="center")
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.plot([r["ci95_low"], r["ci95_high"]], [yi, yi], color=SCEN_COLOR[r["scenario"]],
                linewidth=2.2, alpha=0.55, zorder=2, solid_capstyle="round")
    ax.scatter(df["branching_ratio"], y, c=colors, s=62, edgecolor="white",
               linewidth=0.8, zorder=3)
    ax.axvline(1.0, color=figstyle.REFERENCE_LINE, linestyle="--", linewidth=1.2,
               zorder=1)
    ax.set_yticks(y, [SCEN_LABEL[s] for s in df["scenario"]])
    ax.set_xlabel(r"Branching ratio $n$  (point estimate, 95\% CI)".replace("\\%", "%"))
    ax.set_xlim(xlo - 0.006, xhi + 0.006)
    figstyle.panel_label(ax, "Branching-ratio uncertainty")
    ax.invert_yaxis()
    ax.grid(axis="x", color=figstyle.GRID, linewidth=0.7, alpha=0.9)
    _save(fig, "figure6_branching_uncertainty")

    key["branching_uncertainty"] = {
        r.scenario: {
            "n": float(r.branching_ratio), "se": float(r.se_branching_ratio),
            "ci95": [float(r.ci95_low), float(r.ci95_high)],
            "excludes_unity": bool(r.excludes_unity),
        } for r in df.itertuples()
    }
    print("figure6_branching_uncertainty written")


def fig_bounded_magnitude(key):
    path = TABLE_DIR / "bounded_branching.csv"
    if not path.exists():
        print("skip bounded_magnitude (no table)"); return
    df = pd.read_csv(path)
    mmax_cols = sorted([c for c in df.columns if c.startswith("n_mmax_")],
                       key=lambda c: float(c.split("_")[-1]))
    mmax_vals = [float(c.split("_")[-1]) for c in mmax_cols]

    has_tail = any(SIM_ROOT.glob("nzbnd_window_2000_mmax*"))
    ncols = 2 if has_tail else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8.4 if has_tail else 5.0, 3.7))
    axes = np.atleast_1d(axes)
    if has_tail:
        fig.subplots_adjust(wspace=0.32)

    ax = axes[0]
    for s in SCEN_ORDER:
        row = df[df["scenario"] == s]
        if row.empty:
            continue
        ys = [float(row[c].iloc[0]) for c in mmax_cols] + [float(row["n_unbounded"].iloc[0])]
        ax.plot(mmax_vals, ys[:-1], "o-", color=SCEN_COLOR[s], markersize=4.5,
                linewidth=1.7, markeredgecolor="white", markeredgewidth=0.5,
                label=SCEN_LABEL[s])
        ax.scatter([mmax_vals[-1] + 0.5], [ys[-1]], color=SCEN_COLOR[s],
                   marker="*", s=80, edgecolor="white", linewidth=0.4, zorder=4)
    ax.axhline(1.0, color=figstyle.REFERENCE_LINE, linestyle="--", linewidth=1.1)
    ax.set_xlabel(r"Maximum magnitude $M_{\max}$  (star: unbounded)")
    ax.set_ylabel(r"Branching ratio $n$")
    figstyle.panel_label(ax, r"(a) Admissibility vs $M_{\max}$")
    ax.legend(frameon=False, ncol=2, fontsize=7.2, loc="center left",
              bbox_to_anchor=(0.0, 0.42))

    if has_tail:
        ax2 = axes[1]
        unb_dir = SIM_ROOT / "nz_wide_calibration_window_2000_20210101_000000"
        bnd_dirs = sorted(SIM_ROOT.glob("nzbnd_window_2000_mmax*"))
        unb = pd.read_csv(unb_dir / "forecasts_1826days.csv",
                          usecols=["catalog_id", "magnitude"])
        unb_max = unb.groupby("catalog_id")["magnitude"].max()
        bins = np.arange(4.5, 11.6, 0.25)
        ax2.hist(unb_max, bins=bins, color="#9AA7B2", alpha=0.85, label="Unbounded",
                 zorder=2)
        if bnd_dirs:
            bnd = pd.read_csv(bnd_dirs[-1] / "forecasts_1826days.csv",
                              usecols=["catalog_id", "magnitude"])
            bnd_max = bnd.groupby("catalog_id")["magnitude"].max()
            mm_tag = bnd_dirs[-1].name.split("mmax")[-1].split("_")[0].replace("p", ".")
            ax2.hist(bnd_max, bins=bins, color=SCEN_COLOR["window_2000"], alpha=0.6,
                     label=rf"Truncated $M_{{\max}}={mm_tag}$", zorder=3)
        ax2.axvline(7.2, color=figstyle.OBSERVED, linestyle="--", linewidth=1.2,
                    label="Observed max (M7.2)", zorder=4)
        ax2.set_xlabel("Catalog maximum magnitude (5-yr)")
        ax2.set_ylabel("Number of catalogs")
        figstyle.panel_label(ax2, "(b) Maximum-magnitude distribution")
        ax2.legend(frameon=False, fontsize=8)

    _save(fig, "figure7_bounded_magnitude")
    key["bounded_magnitude"] = {
        r.scenario: {"n_unbounded": float(r.n_unbounded),
                     **{f"n_mmax_{mmax_vals[i]:g}": float(r[mmax_cols[i]])
                        for i in range(len(mmax_cols))}}
        for _, r in df.iterrows()
    }
    print("figure7_bounded_magnitude written")


def fig_reference(key):
    path = TABLE_DIR / "reference_comparison.csv"
    if not path.exists():
        print("skip reference (no table)"); return
    df = pd.read_csv(path).sort_values("duration_days")
    yrs = df["duration_days"] / 365.25
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))
    fig.subplots_adjust(wspace=0.3)
    has_spatial = "info_gain_per_eq_spatial" in df.columns
    if has_spatial:
        axes[0].bar(yrs - 0.18, df["info_gain_per_eq_spatial"], width=0.34,
                    color=SCEN_COLOR["window_2000"], edgecolor="white",
                    linewidth=0.7, zorder=3, label="Normalized spatial")
        axes[0].bar(yrs + 0.18, df["info_gain_per_eq"], width=0.34,
                    color="#9AA7B2", edgecolor="white", linewidth=0.7, zorder=3,
                    label="Rate-based")
        axes[0].legend(loc="upper right", fontsize=8)
        ymax = max(df["info_gain_per_eq_spatial"].max(), df["info_gain_per_eq"].max())
    else:
        axes[0].bar(yrs, df["info_gain_per_eq"], width=0.55,
                    color=SCEN_COLOR["window_2000"], edgecolor="white",
                    linewidth=0.7, zorder=3)
        ymax = df["info_gain_per_eq"].max()
    axes[0].axhline(0.0, color=figstyle.TEXT, linewidth=1.0, zorder=2)
    axes[0].set_xlabel("Forecast horizon (yr)")
    axes[0].set_ylabel("Information gain per earthquake (nats)")
    axes[0].set_ylim(0, ymax * 1.22)
    figstyle.panel_label(axes[0], "(a) ETAS skill over reference")

    axes[1].plot(yrs, df["observed_count"], "s-", color=figstyle.OBSERVED,
                 linewidth=1.8, markersize=5, label="Observed")
    axes[1].plot(yrs, df["etas_mean_count"], "o-", color=SCEN_COLOR["window_2000"],
                 linewidth=2.0, markersize=5, markeredgecolor="white",
                 markeredgewidth=0.5, label="ETAS 2000-window")
    axes[1].plot(yrs, df["ref_mean_count"], "^-", color=SCEN_COLOR["baseline"],
                 linewidth=2.0, markersize=5, markeredgecolor="white",
                 markeredgewidth=0.5, label="Smoothed-Poisson reference")
    axes[1].set_xlabel("Forecast horizon (yr)")
    axes[1].set_ylabel("Mean forecast count")
    figstyle.panel_label(axes[1], "(b) Count forecasts vs observed")
    axes[1].legend(loc="upper left")
    _save(fig, "figure8_reference")

    key["reference"] = {
        "info_gain_per_eq": {int(r.duration_days): float(r.info_gain_per_eq)
                             for r in df.itertuples()},
        "etas_positive_ig_all_horizons": bool((df["info_gain_per_eq"] > 0).all()),
        "etas_n_passes": int(df["etas_n_consistent"].sum()),
        "ref_n_passes": int(df["ref_n_consistent"].sum()),
        "etas_s_passes": int(df["etas_s_consistent"].sum()),
        "ref_s_passes": int(df["ref_s_consistent"].sum()),
    }
    print("figure8_reference written")


def main():
    setup()
    key = {}
    fig_multi_origin(key)
    fig_branching_uncertainty(key)
    fig_bounded_magnitude(key)
    fig_reference(key)
    with (TABLE_DIR / "key_results_experiments.json").open("w") as fh:
        json.dump(key, fh, indent=2)
    print(f"\nWrote {TABLE_DIR / 'key_results_experiments.json'}")
    print(json.dumps(key, indent=2)[:2000])


if __name__ == "__main__":
    main()
