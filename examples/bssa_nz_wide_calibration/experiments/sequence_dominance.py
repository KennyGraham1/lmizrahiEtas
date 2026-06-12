#!/usr/bin/env python3
"""Experiment H3: the sequence-dominance law for the fitted ETAS branching ratio.

HYPOTHESIS
----------
The fitted branching ratio ``n`` (a.k.a. ``eta``) is an *increasing* function of
the largest aftershock sequence's fractional share of the training catalog. In
other words, the apparent "training-window sensitivity" of ``n`` is really
*sequence dominance* in disguise: when one big mainshock-aftershock cascade makes
up a larger fraction of the events used to fit the model, the EM inversion infers
more triggering and pushes ``n`` toward (or above) unity.

The key data point is the (2000-window, 2017-origin) fit, where ``n`` jumps to
1.023 while the neighbouring origins of the same window sit near ~0.97. The 2017
origin is the first training window to fully contain the 2016 M7.8 Kaikoura
sequence, so if the hypothesis holds, that spike should lie *on* the regression
line of ``n`` versus largest-cluster-fraction rather than being an outlier.

METHOD
------
No new inversions are run. We reuse the 15 fitted ``n`` values (3 windows x 5
origins) from ``tables/multi_origin_branching.csv``. For each (window, origin):

1.  Reconstruct the TRAINING (target) catalog from ``input_data/nzcat.csv``:
    events with ``window_start_date <= time < origin_date`` and
    ``magnitude >= mc`` (mc = 4.1), restricted to the published source polygon
    ``input_data/nz_polygon.npy`` using a matplotlib ``Path`` on ``[lon, lat]``,
    exactly as ``run_nz_wide_forecast.load_catalog`` does.

2.  Decluster that catalog with the Zaliapin & Ben-Zion (2013) nearest-neighbour
    method, which is *independent of the ETAS fit* (so the regression is not
    circular). For every event j, its nearest neighbour among earlier events i is
    the one minimising the rescaled proximity

        eta_ij = t_ij * (r_ij)^df * 10^(-b * m_i)      (t_ij > 0, else +inf)

    with t_ij = (t_j - t_i) in years, r_ij = haversine epicentral distance in km,
    df = 1.6 (fractal dimension), b = 1.0 (Gutenberg-Richter b), m_i the parent
    candidate's magnitude. We store eta_j = min_i eta_ij and the chosen parent.

3.  Classify each nearest-neighbour link as background vs clustered by the
    bimodality of ``log10(eta_j)``: a link is "clustered" when
    ``log10(eta_j) < threshold``. The default reproducible threshold is -4.5 (a
    standard NZ-region value); we also compute an Otsu/midpoint threshold on the
    histogram of ``log10(eta_j)`` and report which we used. Clustered links are
    traced via union-find to form clusters; background events (link above the
    threshold, or the earliest event with no parent) seed singleton clusters.

4.  Report the largest-cluster size, ``largest_cluster_fraction`` (largest cluster
    size / n_target_events), the top-3 cluster fraction, and the clustered
    (non-background) fraction.

5.  Regress ``n`` on ``largest_cluster_fraction`` across all 15 fits (OLS): slope,
    intercept, R^2, Pearson & Spearman correlations with p-values, and the
    residual of the (2000, 2017) point relative to the fitted line.

Outputs
-------
``tables/sequence_dominance.csv``  (one row per window/origin)

Smoke test: run with ``--smoke`` to process a single window/origin (default
2000/2021) and write to ``tables/sequence_dominance_smoke.csv``; this proves the
declustering and the per-fit metrics work without doing all 15.

Full run: invoke with no smoke flags to process all 15 fits and run the
regression.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- sys.path bootstrap (copied verbatim from existing experiment scripts) ----
HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

TABLE_DIR = BSSA / "tables"
INPUT_DATA_DIR = ROOT / "input_data"
CATALOG_PATH = INPUT_DATA_DIR / "nzcat.csv"
POLYGON_PATH = INPUT_DATA_DIR / "nz_polygon.npy"
MULTI_ORIGIN_TABLE = TABLE_DIR / "multi_origin_branching.csv"

# Completeness magnitude and Zaliapin & Ben-Zion (2013) declustering constants.
MC = 4.1
DF = 1.6           # fractal dimension of the epicentre distribution
GR_B = 1.0         # Gutenberg-Richter b-value
DEFAULT_THRESHOLD = -4.5  # standard NZ-region log10(eta) split for background/cluster

# window_start label -> calendar start date of the training window.
WINDOW_START_DATES = {
    "1960": "1960-01-01",
    "1980": "1980-01-01",
    "2000": "2000-01-01",
}

DAYS_PER_YEAR = 365.25
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Catalog reconstruction
# ---------------------------------------------------------------------------
def load_polygon_filtered_catalog(catalog_path: Path, polygon_path: Path) -> pd.DataFrame:
    """Load nzcat.csv and keep only events inside the source polygon.

    Mirrors ``run_nz_wide_forecast.load_catalog``: the polygon .npy stores
    ``[lat, lon]`` vertices, and the containment test is run on ``[lon, lat]``.
    """
    catalog = pd.read_csv(catalog_path, index_col=0, parse_dates=["time"])
    catalog.sort_values(by="time", inplace=True)
    if os.path.exists(polygon_path):
        from matplotlib.path import Path as MplPath

        coords = np.load(polygon_path)  # [lat, lon]
        poly_path = MplPath(np.column_stack([coords[:, 1], coords[:, 0]]))  # [lon, lat]
        points = np.column_stack([catalog["longitude"], catalog["latitude"]])
        mask = poly_path.contains_points(points)
        catalog = catalog.loc[mask]
    return catalog.copy()


def training_catalog(
    full_catalog: pd.DataFrame,
    window_start_date: str,
    origin_date: str,
    mc: float = MC,
) -> pd.DataFrame:
    """Slice the training/target window: [window_start, origin), magnitude >= mc."""
    start = pd.Timestamp(window_start_date)
    stop = pd.Timestamp(origin_date)
    mask = (
        (full_catalog["time"] >= start)
        & (full_catalog["time"] < stop)
        & (full_catalog["magnitude"] >= mc)
    )
    return full_catalog.loc[mask].sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Zaliapin & Ben-Zion (2013) nearest-neighbour declustering
# ---------------------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle epicentral distance (km). Inputs in degrees; broadcasts."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_neighbour_eta(
    catalog: pd.DataFrame,
    df: float = DF,
    b: float = GR_B,
) -> tuple[np.ndarray, np.ndarray]:
    """Zaliapin-Ben-Zion nearest-neighbour proximity for each event.

    For each event j (ordered by time) we search all *earlier* events i and find
    the parent candidate minimising

        eta_ij = t_ij * r_ij^df * 10^(-b * m_i),   t_ij in years (>0).

    Returns
    -------
    eta_j : array of shape (N,)
        The minimum proximity for each event; +inf for the earliest event
        (no earlier neighbour exists).
    parent : int array of shape (N,)
        Index (into the reset-index catalog) of the nearest-neighbour parent, or
        -1 when no parent exists (earliest event / background by construction).

    The O(N^2) search is vectorised over i for each j; with N <= ~17.5k events the
    largest training window completes in seconds.
    """
    n = len(catalog)
    times = catalog["time"].to_numpy()
    # event times in (fractional) years relative to the first event.
    t_years = (times - times[0]) / np.timedelta64(1, "D") / DAYS_PER_YEAR
    lat = catalog["latitude"].to_numpy(dtype=float)
    lon = catalog["longitude"].to_numpy(dtype=float)
    mag = catalog["magnitude"].to_numpy(dtype=float)

    eta_j = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)

    # Precompute the magnitude weight 10^(-b * m_i) for every potential parent i.
    mag_weight = np.power(10.0, -b * mag)

    for j in range(1, n):
        t_ij = t_years[j] - t_years[:j]  # all strictly >= 0; ties (== 0) excluded below
        valid = t_ij > 0.0
        if not np.any(valid):
            continue
        r_ij = _haversine_km(lat[:j], lon[:j], lat[j], lon[j])
        # Guard the r^df term against r == 0 (collocated events): use a tiny floor
        # so the distance factor does not annihilate the proximity.
        r_ij = np.maximum(r_ij, 1e-6)
        eta_ij = np.full(j, np.inf)
        eta_ij[valid] = (
            t_ij[valid] * np.power(r_ij[valid], df) * mag_weight[:j][valid]
        )
        k = int(np.argmin(eta_ij))
        eta_j[j] = eta_ij[k]
        parent[j] = k
    return eta_j, parent


def _otsu_threshold(values: np.ndarray, bins: int = 60) -> float:
    """Otsu between-class-variance threshold on a 1-D sample (finite values only)."""
    v = values[np.isfinite(values)]
    if v.size < 2:
        return float("nan")
    counts, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    if total == 0:
        return float("nan")
    w = counts.astype(float) / total
    cum_w = np.cumsum(w)
    cum_mean = np.cumsum(w * centers)
    global_mean = cum_mean[-1]
    # Between-class variance for each possible split position.
    denom = cum_w * (1.0 - cum_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (global_mean * cum_w - cum_mean) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -np.inf
    return float(centers[int(np.argmax(sigma_b))])


def _midpoint_threshold(values: np.ndarray) -> float:
    """Midpoint between the two highest histogram modes of log10(eta_j)."""
    v = values[np.isfinite(values)]
    if v.size < 2:
        return float("nan")
    counts, edges = np.histogram(v, bins=60)
    centers = 0.5 * (edges[:-1] + edges[1:])
    order = np.argsort(counts)[::-1]
    # The two strongest, well-separated bins approximate the bimodal modes.
    top = centers[order[0]]
    for idx in order[1:]:
        if abs(centers[idx] - top) > (centers[1] - centers[0]) * 3:
            return float(0.5 * (top + centers[idx]))
    return float(np.median(v))


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def decluster_metrics(
    catalog: pd.DataFrame,
    n_target_events: int,
    threshold: float,
    threshold_mode: str,
    df: float = DF,
    b: float = GR_B,
) -> dict:
    """Run Zaliapin-Ben-Zion declustering and return cluster-share metrics.

    A nearest-neighbour link is "clustered" when ``log10(eta_j) < threshold``.
    Clustered links are merged via union-find into clusters; everything else is a
    background singleton. ``largest_cluster_fraction`` etc. are normalised by
    ``n_target_events`` (the inversion's target count) so the metric is directly
    comparable to the fitted ``n``.
    """
    n = len(catalog)
    eta_j, parent = nearest_neighbour_eta(catalog, df=df, b=b)
    log_eta = np.log10(eta_j)  # +inf -> +inf (earliest event), kept as background

    # Resolve the threshold actually used.
    finite = log_eta[np.isfinite(log_eta)]
    otsu = _otsu_threshold(finite)
    midpoint = _midpoint_threshold(finite)
    if threshold_mode == "fixed":
        used = float(threshold)
    elif threshold_mode == "otsu":
        used = float(otsu)
    elif threshold_mode == "midpoint":
        used = float(midpoint)
    else:
        raise ValueError(f"unknown threshold_mode: {threshold_mode}")

    clustered_link = np.isfinite(log_eta) & (log_eta < used)

    uf = _UnionFind(n)
    for j in range(n):
        if clustered_link[j] and parent[j] >= 0:
            uf.union(parent[j], j)

    roots = np.array([uf.find(i) for i in range(n)])
    _, sizes = np.unique(roots, return_counts=True)
    sizes_sorted = np.sort(sizes)[::-1]

    largest = int(sizes_sorted[0]) if sizes_sorted.size else 0
    top3 = int(sizes_sorted[:3].sum()) if sizes_sorted.size else 0
    clustered_events = int(clustered_link.sum())

    denom = float(max(n_target_events, 1))
    return {
        "n_events_reconstructed": n,
        "largest_cluster_size": largest,
        "largest_cluster_fraction": largest / denom,
        "top3_cluster_size": top3,
        "top3_cluster_fraction": top3 / denom,
        "clustered_fraction": clustered_events / denom,
        "n_clusters": int(sizes.size),
        "threshold_used": used,
        "threshold_mode": threshold_mode,
        "threshold_otsu": otsu,
        "threshold_midpoint": midpoint,
    }


# ---------------------------------------------------------------------------
# Per-fit driver and regression
# ---------------------------------------------------------------------------
def process_fit(
    full_catalog: pd.DataFrame,
    window: str,
    origin_date: str,
    n_value: float,
    n_target_events: int,
    mc: float,
    threshold: float,
    threshold_mode: str,
) -> dict:
    window_start_date = WINDOW_START_DATES[window]
    train = training_catalog(full_catalog, window_start_date, origin_date, mc)
    metrics = decluster_metrics(
        train, n_target_events, threshold=threshold, threshold_mode=threshold_mode
    )
    row = {
        "window": window,
        "origin": origin_date,
        "origin_year": pd.Timestamp(origin_date).year,
        "window_start_date": window_start_date,
        "n": float(n_value),
        "n_target_events": int(n_target_events),
    }
    row.update(metrics)
    return row


def run_regression(frame: pd.DataFrame) -> dict:
    """OLS of n on largest_cluster_fraction plus correlation diagnostics."""
    from scipy import stats

    x = frame["largest_cluster_fraction"].to_numpy(dtype=float)
    y = frame["n"].to_numpy(dtype=float)

    ls = stats.linregress(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    fitted = ls.intercept + ls.slope * x
    residuals = y - fitted

    result = {
        "slope": float(ls.slope),
        "intercept": float(ls.intercept),
        "r_squared": float(ls.rvalue ** 2),
        "slope_p_value": float(ls.pvalue),
        "slope_stderr": float(ls.stderr),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }

    # Locate the (2000, 2017) key point and report its residual.
    key_mask = (frame["window"] == "2000") & (
        frame["origin"].astype(str).str.startswith("2017")
    )
    if key_mask.any():
        idx = int(np.flatnonzero(key_mask.to_numpy())[0])
        result["key_point_window"] = "2000"
        result["key_point_origin"] = str(frame.iloc[idx]["origin"])
        result["key_point_n"] = float(y[idx])
        result["key_point_largest_cluster_fraction"] = float(x[idx])
        result["key_point_fitted_n"] = float(fitted[idx])
        result["key_point_residual"] = float(residuals[idx])
        # Standardised residual relative to the residual spread.
        resid_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
        result["key_point_residual_std_units"] = (
            float(residuals[idx] / resid_std) if resid_std > 0 else float("nan")
        )
    return result


def load_fits(table_path: Path) -> pd.DataFrame:
    """Read multi_origin_branching.csv and normalise the fields we need."""
    df = pd.read_csv(table_path)
    df = df.rename(columns={"branching_ratio": "n"})
    df["window"] = df["window_start"].astype(str)
    df["origin"] = pd.to_datetime(df["origin"]).dt.strftime("%Y-%m-%d")
    return df[["window", "origin", "n", "n_target_events"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H3 sequence-dominance law: regress fitted ETAS n on the "
        "largest declustered-cluster fraction of each training catalog."
    )
    parser.add_argument(
        "--multi-origin-table",
        type=Path,
        default=MULTI_ORIGIN_TABLE,
        help="CSV with the 15 fitted branching ratios (default: "
        "tables/multi_origin_branching.csv).",
    )
    parser.add_argument(
        "--catalog", type=Path, default=CATALOG_PATH,
        help="Target catalog CSV (default: input_data/nzcat.csv).",
    )
    parser.add_argument(
        "--polygon", type=Path, default=POLYGON_PATH,
        help="Source-domain polygon .npy (default: input_data/nz_polygon.npy).",
    )
    parser.add_argument("--mc", type=float, default=MC, help="Completeness magnitude.")
    parser.add_argument("--df", type=float, default=DF, help="Fractal dimension df.")
    parser.add_argument("--b-value", type=float, default=GR_B, help="GR b-value.")
    parser.add_argument(
        "--threshold-mode", choices=["fixed", "otsu", "midpoint"], default="fixed",
        help="How to split background vs clustered nearest-neighbour links.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="log10(eta) split used when --threshold-mode fixed (default -4.5).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Process a single window/origin (see --smoke-window / --smoke-origin) "
        "and write to a *_smoke.csv file without running the regression.",
    )
    parser.add_argument("--smoke-window", default="2000", choices=list(WINDOW_START_DATES))
    parser.add_argument("--smoke-origin", default="2021-01-01")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override output CSV path.",
    )
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading fitted branching ratios from {args.multi_origin_table}", flush=True)
    fits = load_fits(args.multi_origin_table)

    print(f"Loading + polygon-filtering catalog {args.catalog}", flush=True)
    full_catalog = load_polygon_filtered_catalog(args.catalog, args.polygon)
    print(f"  {len(full_catalog)} events inside polygon", flush=True)

    if args.smoke:
        origin = pd.to_datetime(args.smoke_origin).strftime("%Y-%m-%d")
        sel = fits[(fits["window"] == args.smoke_window) & (fits["origin"] == origin)]
        if sel.empty:
            raise SystemExit(
                f"No fitted row for window={args.smoke_window} origin={origin} "
                f"in {args.multi_origin_table}"
            )
        fits = sel
        print(f"SMOKE: only window={args.smoke_window} origin={origin}", flush=True)

    rows = []
    for _, fit in fits.iterrows():
        row = process_fit(
            full_catalog,
            window=fit["window"],
            origin_date=fit["origin"],
            n_value=fit["n"],
            n_target_events=int(fit["n_target_events"]),
            mc=args.mc,
            threshold=args.threshold,
            threshold_mode=args.threshold_mode,
        )
        rows.append(row)
        print(
            f"  window={row['window']} origin={row['origin']}  "
            f"n={row['n']:.4f}  N={row['n_target_events']}  "
            f"largest_cluster={row['largest_cluster_size']} "
            f"(frac={row['largest_cluster_fraction']:.4f})  "
            f"top3_frac={row['top3_cluster_fraction']:.4f}  "
            f"clustered_frac={row['clustered_fraction']:.4f}  "
            f"thr={row['threshold_used']:.2f}",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values(["window", "origin"]).reset_index(drop=True)

    if args.output is not None:
        out = args.output
    elif args.smoke:
        out = TABLE_DIR / "sequence_dominance_smoke.csv"
    else:
        out = TABLE_DIR / "sequence_dominance.csv"
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}", flush=True)
    print(frame.to_string(index=False), flush=True)

    if not args.smoke and len(frame) >= 3:
        reg = run_regression(frame)
        print("\n=== Regression: n ~ largest_cluster_fraction (OLS, 15 fits) ===")
        print(
            f"  slope     = {reg['slope']:.4f}  (p={reg['slope_p_value']:.3g}, "
            f"se={reg['slope_stderr']:.4f})"
        )
        print(f"  intercept = {reg['intercept']:.4f}")
        print(f"  R^2       = {reg['r_squared']:.4f}")
        print(f"  Pearson r = {reg['pearson_r']:.4f}  (p={reg['pearson_p']:.3g})")
        print(f"  Spearman r= {reg['spearman_r']:.4f}  (p={reg['spearman_p']:.3g})")
        if "key_point_residual" in reg:
            print(
                f"\n  KEY POINT (2000, {reg['key_point_origin']}): "
                f"n={reg['key_point_n']:.4f}, "
                f"largest_cluster_fraction={reg['key_point_largest_cluster_fraction']:.4f}"
            )
            print(
                f"    fitted n on line = {reg['key_point_fitted_n']:.4f}  "
                f"residual = {reg['key_point_residual']:+.4f}  "
                f"({reg['key_point_residual_std_units']:+.2f} resid-std units)"
            )
            on_line = abs(reg["key_point_residual_std_units"]) <= 2.0
            print(
                "    -> The 2017 spike "
                + ("LIES ON" if on_line else "DEVIATES FROM")
                + " the regression line (|standardised residual| "
                + ("<=" if on_line else ">")
                + " 2)."
            )
        # Persist the regression diagnostics alongside the per-fit table.
        import json

        reg_path = out.with_name(out.stem + "_regression.json")
        with open(reg_path, "w") as fh:
            json.dump(reg, fh, indent=2)
        print(f"\nWrote {reg_path}")


if __name__ == "__main__":
    main()
