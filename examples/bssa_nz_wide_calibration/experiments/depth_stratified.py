#!/usr/bin/env python3
"""Experiment H4: a depth-stratified Simpson's paradox for ETAS criticality.

Hypothesis. A single ETAS triggering vector (k0, a, d, gamma, ...) is fitted to a
catalog that mixes two physically distinct tectonic regimes -- shallow crustal
seismicity (<=40 km) and the deep slab / subduction-interface population
(>40 km). If those regimes have different productivity and spatial-decay
structure, the one pooled vector must compromise between them and can
*misallocate* cross-regime triggering. The hypothesis is that this misallocation
inflates the pooled branching ratio above unity (supercritical) even when each
depth stratum, fitted on its own, is subcritical -- a Simpson's-paradox-for-
criticality signal.

Design. The published target catalog (nzcat.csv) carries no depth column, so the
stratification is impossible there. We instead use the buffered, typed catalog
(nzcat_buffered_typed.csv, raw reported magnitudes) which carries a 'depth'
column in km, and we pass the *published* 165--180 deg E polygon (nz_polygon.npy)
so the inversion clips to the same spatial domain as the published 1960 fit and
the pooled control fit stays comparable to it.

We build three catalogs that differ ONLY in depth membership:
  pooled  : all events (control)
  crustal : depth <= 40 km
  deep    : depth >  40 km
Events with NaN depth are kept in the pooled set and excluded from both strata;
the count of such dropped events is reported.

We refit ETAS on each of the three catalogs for two training windows:
  1960 window : timewindow_start=1960-01-01, auxiliary_start=1950-01-01
  2000 window : timewindow_start=2000-01-01, auxiliary_start=1990-01-01
giving six inversions in total (3 strata x 2 windows). Each fit uses mc=4.1,
shape_coords=nz_polygon.npy, forecast_start=2021-01-01, the canonical ETAS
inversion recipe, and the asymptotic-Hessian + delta-method 95% interval on the
branching ratio reused verbatim from branching_uncertainty.py.

The six inversions run in a spawn ProcessPoolExecutor exactly like
run_multi_origin.py / run_homogenized.py: each worker sets its numba thread count
and imports etas *inside* the worker process.

Outputs tables/depth_stratified.csv with one row per (window, stratum):
  window, stratum, n_target_events, n, ci_low, ci_high, excludes_unity (+ beta,
  se, hessian_pos_def for provenance). The script then prints the key 1960-window
  comparison: is n_pooled > n_crustal and n_pooled > n_deep, and is either
  stratum subcritical?
"""

from __future__ import annotations

import sys
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

TABLE_DIR = BSSA / "tables"
INPUT_DIR = ROOT / "input_data"

# Source catalog: HAS a depth column (km) and raw reported magnitudes. The
# published nzcat.csv has no depth, so it cannot be stratified.
SOURCE_CATALOG = INPUT_DIR / "nzcat_buffered_typed.csv"
# Published 165--180 deg E domain: keeps the pooled fit comparable to the
# published 1960 fit (the inversion clips events to this polygon).
POLYGON = str(INPUT_DIR / "nz_polygon.npy")

# Depth boundary (km) between the shallow crustal population and the deep
# slab / subduction population.
DEPTH_SPLIT_KM = 40.0

# Temp catalog files written to input_data/, one per stratum. index=True so the
# 'id' index is preserved; columns include time,latitude,longitude,magnitude
# (plus depth, carried through harmlessly).
STRATA_FILES = {
    "pooled": INPUT_DIR / "nzcat_typed_pooled.csv",
    "crustal": INPUT_DIR / "nzcat_typed_crustal.csv",
    "deep": INPUT_DIR / "nzcat_typed_deep.csv",
}

WINDOWS = {
    "1960": ("1960-01-01 00:00:00", "1950-01-01 00:00:00"),
    "2000": ("2000-01-01 00:00:00", "1990-01-01 00:00:00"),
}

MC = 4.1
FORECAST_START = dt.datetime(2021, 1, 1)

# Reuse the exact CI machinery used by every other experiment.
from branching_uncertainty import _numerical_hessian  # noqa: E402


def build_strata_catalogs() -> dict:
    """Write the pooled/crustal/deep depth-stratified catalogs to input_data/.

    Returns a dict of provenance counts (total, kept-with-depth, NaN-depth
    dropped from strata, crustal, deep). Strata differ ONLY in depth membership;
    the pooled catalog is the full event set (NaN-depth events included).
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(SOURCE_CATALOG, index_col=0, parse_dates=["time"])

    required = ["time", "latitude", "longitude", "magnitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"source catalog missing required columns: {missing}")
    if "depth" not in df.columns:
        raise ValueError("source catalog has no 'depth' column; cannot stratify")

    # Carry depth through so the written catalogs remain self-describing; the
    # ETAS reader uses only time/lat/lon/magnitude and ignores extra columns.
    keep_cols = required + ["depth"]
    base = df[keep_cols].copy()

    depth = base["depth"]
    nan_depth_mask = depth.isna()
    n_nan_depth = int(nan_depth_mask.sum())

    pooled = base  # control: everything, NaN depth included
    crustal = base[(~nan_depth_mask) & (depth <= DEPTH_SPLIT_KM)]
    deep = base[(~nan_depth_mask) & (depth > DEPTH_SPLIT_KM)]

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(STRATA_FILES["pooled"], index=True)
    crustal.to_csv(STRATA_FILES["crustal"], index=True)
    deep.to_csv(STRATA_FILES["deep"], index=True)

    counts = {
        "n_total": int(len(base)),
        "n_nan_depth_dropped_from_strata": n_nan_depth,
        "n_pooled": int(len(pooled)),
        "n_crustal": int(len(crustal)),
        "n_deep": int(len(deep)),
    }
    print(
        "Built depth-stratified catalogs (whole catalog, pre-clip to polygon):\n"
        f"  total events            : {counts['n_total']}\n"
        f"  NaN depth dropped strata: {counts['n_nan_depth_dropped_from_strata']}\n"
        f"  pooled  (control)       : {counts['n_pooled']}\n"
        f"  crustal (<= {DEPTH_SPLIT_KM:g} km)      : {counts['n_crustal']}\n"
        f"  deep    (>  {DEPTH_SPLIT_KM:g} km)      : {counts['n_deep']}",
        flush=True,
    )
    # Sanity: pooled = crustal + deep + NaN-depth.
    assert counts["n_pooled"] == (
        counts["n_crustal"] + counts["n_deep"] + counts["n_nan_depth_dropped_from_strata"]
    ), "stratum membership does not partition the pooled catalog"
    return counts


def _fit(task: dict) -> dict:
    """Run one ETAS inversion on one (window, stratum) catalog in a worker.

    Sets numba threads and imports etas inside the worker so the spawn start
    method re-imports cleanly. Returns the branching ratio with its asymptotic
    Hessian + delta-method 95% interval (machinery copied from
    branching_uncertainty.py).
    """
    import numba
    numba.set_num_threads(task["numba_threads"])

    import numpy as np
    import run_nz_wide_forecast as fc
    from etas.inversion import (
        ETASParameterCalculation,
        branching_ratio,
        neg_log_likelihood,
    )

    tw_start, aux_start = WINDOWS[task["window"]]
    forecast_start = FORECAST_START
    paths = fc.build_run_paths(task["label"], forecast_start)
    config = fc.build_inversion_config(
        paths["run_label"],
        task["fn_catalog"],
        forecast_start,
        aux_start,
        tw_start,
        MC,
        fc.build_initial_theta(),
        bg_term=None,
        shape_coords=task["shape_coords"],
    )
    calc = ETASParameterCalculation(config)
    calc.prepare()
    calc.invert()

    beta = float(calc.beta)
    mc_min = calc.m_ref - calc.delta_m / 2
    th = calc.theta
    # Fitted 8-vector in the order neg_log_likelihood expects.
    theta8 = np.array(
        [
            th["log10_k0"], th["a"], th["log10_c"], th["omega"],
            th["log10_tau"], th["log10_d"], th["gamma"], th["rho"],
        ],
        dtype=float,
    )
    pij, src = calc.pij, calc.source_events

    def nll(v):
        # neg_log_likelihood mutates source_events["G"]; pass a copy.
        return float(neg_log_likelihood(v, pij, src.copy(), mc_min))

    H = _numerical_hessian(nll, theta8)
    try:
        cov = np.linalg.inv(H)
        pos_def = bool(np.all(np.linalg.eigvalsh(H) > 0))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
        pos_def = False
    if not np.all(np.linalg.eigvalsh(H) > 0):
        cov = np.linalg.pinv(H)
        pos_def = False

    def n_of(v, b):
        # mu, iota slots are unused by branching_ratio.
        return float(branching_ratio(np.array([0.0, 0.0, *v]), b))

    n_point = n_of(theta8, beta)

    # Gradient of n w.r.t. the 8 triggering params (central difference).
    grad = np.zeros(8)
    step = np.maximum(np.abs(theta8) * 1e-5, 1e-7)
    for i in range(8):
        tp = theta8.copy(); tp[i] += step[i]
        tm = theta8.copy(); tm[i] -= step[i]
        grad[i] = (n_of(tp, beta) - n_of(tm, beta)) / (2 * step[i])
    var_params = float(grad @ cov @ grad)

    # Independent beta term: exponential-MLE SE = beta / sqrt(N).
    n_target = int(len(calc.target_events))
    se_beta = beta / np.sqrt(max(n_target, 1))
    dn_db = (n_of(theta8, beta + 1e-4) - n_of(theta8, beta - 1e-4)) / 2e-4
    se_n = float(np.sqrt(max(var_params, 0.0) + (dn_db * se_beta) ** 2))

    return {
        "window": task["window"],
        "stratum": task["stratum"],
        "n_target_events": n_target,
        "n": n_point,
        "ci_low": n_point - 1.96 * se_n,
        "ci_high": n_point + 1.96 * se_n,
        "excludes_unity": (n_point + 1.96 * se_n < 1.0) or (n_point - 1.96 * se_n > 1.0),
        "beta": beta,
        "se_n": se_n,
        "hessian_pos_def": pos_def,
    }


def _print_paradox_summary(frame) -> None:
    """Print the key Simpson's-paradox-for-criticality comparison per window."""
    print("\n=== Simpson's-paradox-for-criticality check ===", flush=True)
    for window in WINDOWS:
        sub = frame[frame["window"] == window]
        by = {r["stratum"]: r for _, r in sub.iterrows()}
        if not {"pooled", "crustal", "deep"} <= set(by):
            print(f"  [{window}] incomplete strata; skipping comparison", flush=True)
            continue
        n_p = by["pooled"]["n"]
        n_c = by["crustal"]["n"]
        n_d = by["deep"]["n"]
        pooled_above_both = (n_p > n_c) and (n_p > n_d)
        crustal_sub = n_c < 1.0
        deep_sub = n_d < 1.0
        pooled_super = n_p >= 1.0
        print(
            f"  [{window}] n_pooled={n_p:.4f}  n_crustal={n_c:.4f}  n_deep={n_d:.4f}",
            flush=True,
        )
        print(
            f"          n_pooled > both strata: {pooled_above_both} | "
            f"pooled supercritical: {pooled_super} | "
            f"crustal subcritical: {crustal_sub} | deep subcritical: {deep_sub}",
            flush=True,
        )
        paradox = pooled_super and crustal_sub and deep_sub
        if window == "1960":
            print(
                f"          --> 1960-window Simpson's-paradox-for-criticality "
                f"signal present: {paradox}",
                flush=True,
            )


def main() -> None:
    import argparse
    import multiprocessing as mp
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=6,
        help="ProcessPoolExecutor workers (default 6 -> all 6 inversions in parallel).",
    )
    parser.add_argument(
        "--numba-threads", type=int, default=10,
        help="numba threads set inside each worker before importing etas.",
    )
    parser.add_argument(
        "--windows", nargs="*", default=list(WINDOWS),
        help="subset of training windows to run (default: all).",
    )
    parser.add_argument(
        "--strata", nargs="*", default=list(STRATA_FILES),
        help="subset of depth strata to run (default: pooled crustal deep).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="output CSV name under tables/ (default depth_stratified.csv). "
             "Use a *_smoke.csv name for smoke tests so the real table is not clobbered.",
    )
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2. Build the depth-stratified catalogs (always, so requested strata exist).
    counts = build_strata_catalogs()

    # 3-4. Assemble the (window x stratum) inversion tasks.
    tasks = []
    for window in args.windows:
        if window not in WINDOWS:
            raise SystemExit(f"unknown window {window!r}; choose from {list(WINDOWS)}")
        for stratum in args.strata:
            if stratum not in STRATA_FILES:
                raise SystemExit(
                    f"unknown stratum {stratum!r}; choose from {list(STRATA_FILES)}"
                )
            tasks.append({
                "window": window,
                "stratum": stratum,
                "fn_catalog": str(STRATA_FILES[stratum]),
                "shape_coords": POLYGON,
                "label": f"nzdepth_{window}_{stratum}",
                "numba_threads": args.numba_threads,
            })

    print(
        f"\nRunning {len(tasks)} depth-stratified inversions "
        f"({args.workers} workers x {args.numba_threads} numba threads)",
        flush=True,
    )

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_fit, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(
                    f"  {row['window']} {row['stratum']:7s}: n={row['n']:.4f} "
                    f"CI[{row['ci_low']:.4f},{row['ci_high']:.4f}] "
                    f"({'super' if row['n'] >= 1 else 'sub'}critical, "
                    f"{row['n_target_events']} target ev) "
                    f"excludes 1: {row['excludes_unity']}",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover
                print(f"  FAILED {t['window']} {t['stratum']}: {exc}", flush=True)

    if not rows:
        raise SystemExit("no inversions succeeded; nothing to write")

    # 5. Output table: ordered by window then stratum (pooled, crustal, deep).
    stratum_order = {"pooled": 0, "crustal": 1, "deep": 2}
    window_order = {"1960": 0, "2000": 1}
    frame = pd.DataFrame(rows)
    frame["__w"] = frame["window"].map(window_order)
    frame["__s"] = frame["stratum"].map(stratum_order)
    frame = frame.sort_values(["__w", "__s"]).drop(columns=["__w", "__s"]).reset_index(drop=True)

    # Column order: the required schema first, provenance after.
    col_order = [
        "window", "stratum", "n_target_events", "n", "ci_low", "ci_high",
        "excludes_unity", "beta", "se_n", "hessian_pos_def",
    ]
    frame = frame[[c for c in col_order if c in frame.columns]]

    out_name = args.out if args.out else "depth_stratified.csv"
    out = TABLE_DIR / out_name
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(frame.to_string(index=False))

    _print_paradox_summary(frame)


if __name__ == "__main__":
    main()
