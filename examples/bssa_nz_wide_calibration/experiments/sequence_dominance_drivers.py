#!/usr/bin/env python3
"""H3 follow-up: what actually drives the fitted branching ratio?

The naive sequence-dominance law (eta increases with the largest cluster's
event-COUNT fraction) is refuted by sequence_dominance.py: across the 15
window x origin fits, eta correlates NEGATIVELY with largest-cluster fraction
(r ~ -0.82), because the count fraction is mechanically an inverse proxy for
catalog size (short windows have fewer events, so a fixed-size sequence is a
larger fraction). And within the short 2000 window the Kaikoura-driven 2017
spike in eta is NOT matched by a spike in count fraction.

This script tests the two candidate drivers that the refutation points to:
  (1) catalog size  N            -- the cross-window trend
  (2) largest training magnitude -- the within-window 2017 spike (the 2016
      Kaikoura M7.8 enters the 2000 window only at the 2017 origin)
plus a productivity-weighted dominance metric that, unlike a raw count
fraction, gives a single large mainshock its ETAS weight:
      prod_dominance = max_i exp(a*(m_i - mc)) / sum_i exp(a*(m_i - mc))
using a representative productivity exponent a (default 1.15, the 2000-window
fit). This is the magnitude analogue of the count-based cluster fraction.

Outputs tables/sequence_dominance_drivers.csv and prints the correlations.
"""

from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent

TABLE_DIR = BSSA / "tables"
CATALOG_PATH = ROOT / "input_data" / "nzcat.csv"
POLYGON_PATH = ROOT / "input_data" / "nz_polygon.npy"
MC = 4.1
A_PROD = 1.15  # representative ETAS productivity exponent (2000-window fit)

WINDOW_START = {"1960": dt.datetime(1960, 1, 1),
                "1980": dt.datetime(1980, 1, 1),
                "2000": dt.datetime(2000, 1, 1)}


def main():
    from scipy import stats
    from matplotlib.path import Path as MplPath

    mo = pd.read_csv(TABLE_DIR / "multi_origin_branching.csv")
    mo["window_start"] = mo["window_start"].astype(str)

    cat = pd.read_csv(CATALOG_PATH, index_col=0, parse_dates=["time"])
    coords = np.load(POLYGON_PATH)
    poly = MplPath(np.column_stack([coords[:, 1], coords[:, 0]]))
    inside = poly.contains_points(np.column_stack([cat["longitude"], cat["latitude"]]))
    cat = cat[inside & (cat["magnitude"] >= MC)]

    rows = []
    for _, r in mo.iterrows():
        w = str(r["window_start"])
        origin = pd.Timestamp(r["origin"])
        start = WINDOW_START[w]
        train = cat[(cat["time"] >= start) & (cat["time"] < origin)]
        mags = train["magnitude"].to_numpy()
        w_prod = np.exp(A_PROD * (mags - MC))
        rows.append({
            "window": w, "origin": r["origin"], "n": float(r["branching_ratio"]),
            "N": int(len(train)), "logN": float(np.log(len(train))),
            "max_mag": float(mags.max()) if len(mags) else np.nan,
            "n_ge_7": int((mags >= 7.0).sum()),
            "prod_dominance": float(w_prod.max() / w_prod.sum()) if len(mags) else np.nan,
            "has_kaikoura": bool((train["time"] >= dt.datetime(2016, 11, 13)).any()
                                 and (mags >= 7.7).any()),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "sequence_dominance_drivers.csv", index=False)

    print("Per-fit drivers (15 window x origin fits):")
    print(frame.to_string(index=False))

    def corr(x):
        good = frame[[x, "n"]].dropna()
        pr = stats.pearsonr(good[x], good["n"])
        sr = stats.spearmanr(good[x], good["n"])
        return pr, sr

    print("\n=== Correlations of fitted n with candidate drivers (15 fits) ===")
    for x in ["N", "logN", "max_mag", "prod_dominance", "n_ge_7"]:
        pr, sr = corr(x)
        print(f"  n vs {x:15s}: Pearson r={pr[0]:+.3f} (p={pr[1]:.4g}), "
              f"Spearman rho={sr.correlation:+.3f} (p={sr.pvalue:.4g})")

    # Within-2000-window: does max_mag explain the 2017 spike?
    w2000 = frame[frame["window"] == "2000"].sort_values("origin")
    print("\n=== Within the 2000 window (the Kaikoura test) ===")
    print(w2000[["origin", "n", "N", "max_mag", "has_kaikoura"]].to_string(index=False))
    pr = stats.pearsonr(w2000["max_mag"], w2000["n"])
    print(f"  within-2000  n vs max_mag: Pearson r={pr[0]:+.3f} (p={pr[1]:.4g})")
    kk = w2000[w2000["has_kaikoura"]]["n"].mean()
    nokk = w2000[~w2000["has_kaikoura"]]["n"].mean()
    print(f"  mean n with Kaikoura in window  = {kk:.4f}")
    print(f"  mean n without Kaikoura         = {nokk:.4f}")
    print(f"  --> Kaikoura inclusion shifts the 2000-window n by {kk - nokk:+.4f}")


if __name__ == "__main__":
    main()
