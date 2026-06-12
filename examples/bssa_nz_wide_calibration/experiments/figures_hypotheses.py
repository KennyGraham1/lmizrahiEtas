#!/usr/bin/env python3
"""Publication figures for the six novel-hypothesis experiments (figures 9-14).

Reads the result tables in ../tables and writes figures to ../figures, matching
the shared figstyle used by the rest of the manuscript. Each figure is guarded by
the presence of its table, so the script can be run incrementally as experiments
finish.

  fig9  H1  synthetic injection: fitted n per degradation variant vs generating n
  fig10 H6  calibration-slope diagnostic: cascade-growth exponent vs fitted n
  fig11 H3  sequence dominance refuted + true drivers (cluster fraction, N)
  fig12 H4  depth-stratified branching ratios (Simpson's paradox test)
  fig13 H5  open-boundary flux: measured and boundary-corrected n vs cutoff L
  fig14 H2  background-flexibility ladder: n vs bandwidth, CV-selected flexibility
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
sys.path.insert(0, str(BSSA))
import figstyle  # noqa: E402

TABLES = BSSA / "tables"
FIGS = BSSA / "figures"
SUPER = "#D1495B"   # supercritical / hot
SUB = "#2C6FB0"     # subcritical / cool
INK = figstyle.NAVY


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png", flush=True)


def _h1_panel(ax, df, title):
    order = ["control", "detection", "magnitude", "both"]
    labels = {"control": "control", "detection": "era\nincompl.",
              "magnitude": "scale\ndrift", "both": "both"}
    df = df[df["ok"]] if "ok" in df else df
    n_true = float(df["n_true"].iloc[0])
    data = [df[df["variant"] == v]["n_refit"].dropna().values for v in order]
    parts = ax.violinplot(data, positions=range(len(order)), showmeans=False,
                          showextrema=False, widths=0.8)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(SUB if order[i] == "control" else SUPER)
        b.set_alpha(0.35); b.set_edgecolor("none")
    ctrl_mean = float(np.mean(data[0]))
    for i, d in enumerate(data):
        x = np.random.normal(i, 0.05, size=len(d))
        ax.scatter(x, d, s=16, color=INK, alpha=0.7, zorder=3, edgecolor="white", linewidth=0.4)
        ax.scatter([i], [np.mean(d)], s=130, marker="_", color="black", zorder=4, linewidth=2.2)
        if i > 0:
            ax.annotate(f"+{np.mean(d)-ctrl_mean:.3f}", (i, np.mean(d)),
                        textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=7.5, color=SUPER)
    ax.axhline(1.0, color=SUPER, ls="--", lw=1.2, zorder=1)
    ax.axhline(n_true, color=SUB, ls=":", lw=1.3, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[v] for v in order])
    figstyle.panel_label(ax, title)
    return n_true


def fig9_synthetic():
    fm = TABLES / "synthetic_injection_matched.csv"
    fu = TABLES / "synthetic_injection_uniform.csv"
    if not fm.exists() and not fu.exists():
        print("  [skip fig9] no synthetic_injection_{matched,uniform}.csv"); return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=False)
    nt = None
    if fm.exists():
        nt = _h1_panel(axes[0], pd.read_csv(fm),
                       "(a) Manuscript pipeline (clustered bg)")
    if fu.exists():
        _h1_panel(axes[1], pd.read_csv(fu),
                  "(b) Uniform bg, exact times (robustness)")
    axes[0].set_ylabel("fitted branching ratio $\\hat{n}$ (1960-start refit)")
    if nt is not None:
        axes[0].text(3.4, 1.003, "$n=1$", color=SUPER, ha="right", va="bottom", fontsize=8)
        axes[0].text(3.4, nt - 0.003, f"generating $n={nt:.2f}$", color=SUB,
                     ha="right", va="top", fontsize=8)
    fig.suptitle("Observational degradation adds a fixed increment to fitted $n$",
                 fontsize=11, color=INK, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "figure9_synthetic_injection")


def fig10_calibration_slope():
    f = TABLES / "calibration_slope.csv"
    if not f.exists():
        print("  [skip fig10] calibration_slope.csv not found"); return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    # (a) cascade growth exponent vs n
    ax = axes[0]
    for _, r in df.iterrows():
        c = figstyle.COLORS.get(r["scenario"], INK)
        ax.scatter(r["n"], r["sim_growth_exponent"], s=70, color=c, zorder=3,
                   edgecolor="white", linewidth=0.6)
    # regression line
    x, y = df["n"].values, df["sim_growth_exponent"].values
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color=figstyle.SUBTLE, lw=1.2, ls="--", zorder=2)
    from scipy.stats import pearsonr
    r_ = pearsonr(x, y)[0]
    ax.axvline(1.0, color=SUPER, ls=":", lw=1.0)
    ax.set_xlabel("fitted branching ratio $n$")
    ax.set_ylabel("simulated count-growth exponent\n$d\\,\\ln \\bar N_{sim}/d\\,\\ln T$")
    figstyle.panel_label(ax, f"(a) Cascade growth tracks $n$  ($r={r_:.2f}$)")
    # (b) obs/sim log-log slope vs n, Mc=4.1 only
    ax = axes[1]
    mc41 = df[~df["scenario"].isin(["mc_4p3", "mc_4p5"])]
    for _, r in mc41.iterrows():
        c = figstyle.COLORS.get(r["scenario"], INK)
        ax.scatter(r["n"], r["loglog_slope"], s=70, color=c, zorder=3,
                   edgecolor="white", linewidth=0.6)
    x2, y2 = mc41["n"].values, mc41["loglog_slope"].values
    if len(x2) > 2:
        b1, b0 = np.polyfit(x2, y2, 1)
        xs = np.linspace(x2.min(), x2.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=figstyle.SUBTLE, lw=1.2, ls="--", zorder=2)
        r2_ = pearsonr(x2, y2)[0]
    else:
        r2_ = np.nan
    ax.axvline(1.0, color=SUPER, ls=":", lw=1.0)
    ax.set_xlabel("fitted branching ratio $n$")
    ax.set_ylabel("observed/simulated ratio\nlog-log slope")
    figstyle.panel_label(ax, f"(b) Calibration slope vs $n$ ($M_c$=4.1; $r={r2_:.2f}$)")
    fig.tight_layout()
    _save(fig, "figure10_calibration_slope")


def fig11_sequence():
    f = TABLES / "sequence_dominance.csv"
    fd = TABLES / "sequence_dominance_drivers.csv"
    if not f.exists():
        print("  [skip fig11] sequence_dominance.csv not found"); return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    cmap = {"1960": figstyle.WINDOW_COLORS["1960"], "1980": figstyle.WINDOW_COLORS["1980"],
            "2000": figstyle.WINDOW_COLORS["2000"]}
    # (a) n vs largest-cluster fraction (refutation: negative)
    ax = axes[0]
    df["window"] = df["window"].astype(str)
    for w, g in df.groupby("window"):
        ax.scatter(g["largest_cluster_fraction"], g["n"], s=60, color=cmap.get(w, INK),
                   label=f"{w} start", zorder=3, edgecolor="white", linewidth=0.6)
    x, y = df["largest_cluster_fraction"].values, df["n"].values
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color=figstyle.SUBTLE, lw=1.2, ls="--", zorder=2)
    from scipy.stats import pearsonr
    r_ = pearsonr(x, y)[0]
    ax.axhline(1.0, color=SUPER, ls=":", lw=1.0)
    ax.set_xlabel("largest-cluster event fraction")
    ax.set_ylabel("fitted branching ratio $n$")
    figstyle.panel_label(ax, f"(a) Count dominance: $r={r_:.2f}$ (wrong sign)")
    ax.legend(fontsize=8, loc="upper right")
    # (b) n vs N (true driver)
    ax = axes[1]
    if fd.exists():
        dd = pd.read_csv(fd); dd["window"] = dd["window"].astype(str)
        for w, g in dd.groupby("window"):
            ax.scatter(g["N"], g["n"], s=60, color=cmap.get(w, INK), zorder=3,
                       edgecolor="white", linewidth=0.6, label=f"{w} start")
        # highlight Kaikoura 2017 in the 2000 window
        kk = dd[(dd["window"] == "2000") & (dd["origin"].astype(str).str.startswith("2017"))]
        if len(kk):
            ax.scatter(kk["N"], kk["n"], s=200, facecolor="none", edgecolor="black",
                       linewidth=1.6, zorder=4)
            ax.annotate("Kaikoura\nenters window", (kk["N"].iloc[0], kk["n"].iloc[0]),
                        textcoords="offset points", xytext=(8, 10), fontsize=8, color=INK)
        x, y = dd["N"].values, dd["n"].values
        r2 = pearsonr(x, y)[0]
        ax.axhline(1.0, color=SUPER, ls=":", lw=1.0)
        ax.set_xlabel("training catalog size $N$")
        ax.set_ylabel("fitted branching ratio $n$")
        figstyle.panel_label(ax, f"(b) Catalog size is the driver ($r={r2:.2f}$)")
    fig.tight_layout()
    _save(fig, "figure11_sequence_dominance")


def fig12_depth():
    f = TABLES / "depth_stratified.csv"
    if not f.exists():
        print("  [skip fig12] depth_stratified.csv not found"); return
    df = pd.read_csv(f)
    df["window"] = df["window"].astype(str)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    strata = ["pooled", "crustal", "deep"]
    windows = sorted(df["window"].unique())
    width = 0.8 / len(strata)
    scol = {"pooled": INK, "crustal": SUB, "deep": "#5AA86B"}
    for si, s in enumerate(strata):
        xs, ns, los, his = [], [], [], []
        for wi, w in enumerate(windows):
            row = df[(df["window"] == w) & (df["stratum"] == s)]
            if not len(row): continue
            x = wi + (si - 1) * width
            xs.append(x); ns.append(row["n"].iloc[0])
            los.append(row["n"].iloc[0] - row["ci_low"].iloc[0])
            his.append(row["ci_high"].iloc[0] - row["n"].iloc[0])
        ax.errorbar(xs, ns, yerr=[los, his], fmt="o", color=scol[s], label=s,
                    capsize=3, markersize=7, zorder=3)
    ax.axhline(1.0, color=SUPER, ls="--", lw=1.2)
    ax.axhspan(1.0, ax.get_ylim()[1], color=SUPER, alpha=0.05)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f"{w} window" for w in windows])
    ax.set_ylabel("branching ratio $n$ (95% CI)")
    ax.legend(title="depth stratum", fontsize=9)
    figstyle.panel_label(ax, "Depth-stratified branching ratios")
    _save(fig, "figure12_depth_stratified")


def fig13_boundary():
    f = TABLES / "boundary_flux.csv"
    if not f.exists():
        print("  [skip fig13] boundary_flux.csv not found"); return
    df = pd.read_csv(f).sort_values("L")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    yerr = [df["n_measured"] - df["ci_low"], df["ci_high"] - df["n_measured"]]
    ax.errorbar(df["L"], df["n_measured"], yerr=yerr, fmt="o-", color=SUB,
                capsize=3, markersize=7, label="measured $n(L)$", zorder=3)
    if "n_corrected" in df:
        ax.plot(df["L"], df["n_corrected"], "s--", color=SUPER, markersize=6,
                label="boundary-corrected $n(L)$", zorder=3)
    ax.axhline(1.0, color="black", ls=":", lw=1.0)
    # Mark the published 180E boundary (the dip).
    dip = df.loc[df["n_measured"].idxmin()]
    ax.scatter([dip["L"]], [dip["n_measured"]], s=180, facecolor="none",
               edgecolor="black", linewidth=1.6, zorder=4)
    ax.annotate("published 180$^\\circ$E\nbisects East Cape sequence",
                (dip["L"], dip["n_measured"]), textcoords="offset points",
                xytext=(6, -28), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax.set_xlabel("eastern boundary longitude $L$ ($^\\circ$E)")
    ax.set_ylabel("2000-window branching ratio $n$")
    figstyle.panel_label(ax, "Subcriticality is a boundary-position artefact")
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, "figure13_boundary_flux")


def fig14_ladder():
    f = TABLES / "background_ladder.csv"
    if not f.exists():
        print("  [skip fig14] background_ladder.csv not found"); return
    df = pd.read_csv(f)
    fin = df[np.isfinite(df["sigma_deg"])].sort_values("sigma_deg", ascending=False)
    homog = df[~np.isfinite(df["sigma_deg"])]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    # Homogeneous background (least flexible) at the left as a separate category,
    # then the smoothed-covariate fits. The branching ratio is BIMODAL: ~1.03 with
    # homogeneous mu, ~0 the instant any data-derived spatial background is admitted.
    xs = list(range(len(fin) + 1))
    labels = ["homogeneous"] + [f"{s:g}" for s in fin["sigma_deg"]]
    ys = ([homog["n_branching"].iloc[0]] if len(homog) else [np.nan]) + list(fin["n_branching"])
    pind = ([0.0] if len(homog) else [np.nan]) + list(fin["mean_P_induced"])
    ax.plot(xs, ys, "o-", color=INK, markersize=8, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color=INK)
    ax.axhline(1.0, color=SUPER, ls="--", lw=1.2)
    ax.fill_between([-0.5, len(xs) - 0.5], 1.0, 1.12, color=SUPER, alpha=0.05)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("background model: homogeneous $\\mu$, then smoothed covariate at bandwidth $\\sigma$ (deg)")
    ax.set_ylabel("branching ratio $n$")
    if "cv_holdout_loglik_per_event" in df and fin["cv_holdout_loglik_per_event"].notna().any():
        best = fin.loc[fin["cv_holdout_loglik_per_event"].idxmax()]
        bx = labels.index(f"{best['sigma_deg']:g}")
        ax.annotate("out-of-sample\nCV optimum", (bx, best["n_branching"]),
                    textcoords="offset points", xytext=(0, 28), ha="center",
                    fontsize=8, color="#2E7D32",
                    arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=0.9))
    figstyle.panel_label(ax, "Branching ratio is bimodal in the background model")
    _save(fig, "figure14_background_ladder")


def main():
    figstyle.apply()
    np.random.seed(0)
    print("Generating hypothesis figures...")
    for fn in (fig9_synthetic, fig10_calibration_slope, fig11_sequence, fig12_depth,
               fig13_boundary, fig14_ladder):
        try:
            fn()
        except Exception as exc:
            print(f"  [error] {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
