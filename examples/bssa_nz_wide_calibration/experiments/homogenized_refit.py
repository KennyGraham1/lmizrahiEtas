#!/usr/bin/env python3
"""Magnitude-homogenized refit: isolate the magnitude-scale-drift contribution
to long-window ETAS supercriticality.

Motivation
----------
The published NZ-wide sweep fits the unhomogenized target catalog
(``input_data/nzcat.csv``) and finds that long training windows are
*supercritical* (branching ratio eta = 1.02--1.03 for 1960/1980 starts) while
only the short 2000-window is sub-critical (eta = 0.969). A leading suspect for
that long-window supercriticality is uncorrected *magnitude-scale drift*: early-era
(pre-1987) deep events carried inflated reported magnitudes, which inflates the
apparent productivity and pushes eta above unity.

``experiments/catalog_homogeneity.py`` already produced
``input_data/nzcat_buffered_homogeneous.csv`` by applying the Zuniga et al. (2005)
historical magnitude correction to those deep pre-1987 events, but NO ETAS fit has
yet consumed it. If a large part of the long-window supercriticality is an artefact
of that drift, refitting on the homogenized catalog should LOWER the long-window eta
toward (or below) unity.

What this script does
---------------------
1. Refits ETAS on three training windows -- 1960 (aux 1950), 1980 (aux 1970),
   2000 (aux 1990) -- all M_c = 4.1, forecast origin 2021-01-01, on the homogenized
   buffered catalog (corrected ``magnitude`` column), clipped to the published
   domain ``input_data/nz_polygon.npy`` via the inversion's own polygon filter
   (passed as ``shape_coords``). Each fit uses the canonical inversion recipe plus
   the asymptotic-Hessian + delta-method 95% interval on eta copied verbatim from
   ``branching_uncertainty.py``.

2. Reads the published (uncorrected ``nzcat.csv``) eta for the same three windows
   from ``tables/branching_uncertainty.csv`` (keys: baseline=1960, window_1980,
   window_2000) and ``tables/multi_origin_branching.csv`` (the 2021 origin rows),
   and tabulates the SHIFT ``delta_n = n_homogenized - n_published`` per window.

3. MATCHED CONTROL. The homogenized catalog is not only magnitude-corrected, it is
   also a *different catalog version and event set* than ``nzcat.csv`` (buffered
   source events clipped to the same polygon, vs. the published target-only
   catalog). So ``delta_n`` above conflates the magnitude correction with the
   catalog-version change. To isolate the magnitude correction, this script ALSO
   fits the UNcorrected buffered catalog (``input_data/nzcat_buffered_typed.csv``)
   on the identical three windows / domain. Because typed and homogeneous share the
   exact same events and differ ONLY in the ``magnitude`` column, the contrast
       delta_n_isolated = n(homogeneous) - n(typed)
   isolates the magnitude correction's effect on a matched event set, free of the
   catalog-version confound.

Output
------
``tables/homogenized_refit.csv`` with, per window, both the homogenized refit
(``catalog="homogeneous"``) and the matched typed control (``catalog="typed"``):
window, catalog, n_target_events, branching_ratio (eta), ci_low, ci_high,
n_published, delta_n (vs published), delta_n_isolated (homogeneous - typed on the
matched set, attached to the homogeneous rows), and excludes_unity.

A spawn ProcessPoolExecutor (default 3 workers) runs the inversions in parallel,
mirroring ``run_multi_origin.py`` / ``run_homogenized.py``.
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

HOMOG_CATALOG = str(ROOT / "input_data" / "nzcat_buffered_homogeneous.csv")
TYPED_CATALOG = str(ROOT / "input_data" / "nzcat_buffered_typed.csv")
PUBLISHED_POLYGON = str(ROOT / "input_data" / "nz_polygon.npy")  # published domain, lon 165-180

# (timewindow_start, auxiliary_start), all M_c = 4.1, forecast origin 2021.
WINDOWS = {
    "1960": ("1960-01-01 00:00:00", "1950-01-01 00:00:00"),
    "1980": ("1980-01-01 00:00:00", "1970-01-01 00:00:00"),
    "2000": ("2000-01-01 00:00:00", "1990-01-01 00:00:00"),
}
MC = 4.1
FORECAST_ORIGIN = dt.datetime(2021, 1, 1)

# Two catalog versions sharing the identical event set / domain so their contrast
# isolates the magnitude correction. "homogeneous" carries the Zuniga-2005
# corrected magnitudes; "typed" carries the raw reported magnitudes.
CATALOGS = {
    "homogeneous": HOMOG_CATALOG,
    "typed": TYPED_CATALOG,
}

# Map each window to its published (uncorrected nzcat.csv) scenario key in
# branching_uncertainty.csv. The 1960 start is the "baseline" scenario.
PUBLISHED_KEYS = {
    "1960": "baseline",
    "1980": "window_1980",
    "2000": "window_2000",
}

# Reuse the verbatim Hessian/delta-method machinery rather than reinventing it.
from branching_uncertainty import _numerical_hessian  # noqa: E402


def _fit(task: dict) -> dict:
    """Run one inversion in a worker process and return eta with its 95% CI.

    Uses the canonical recipe (build_inversion_config + ETASParameterCalculation)
    and the same asymptotic-Hessian + delta-method interval as
    branching_uncertainty.py / run_homogenized.py.
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
    paths = fc.build_run_paths(task["label"], FORECAST_ORIGIN)
    config = fc.build_inversion_config(
        paths["run_label"],
        task["fn_catalog"],
        FORECAST_ORIGIN,
        aux_start,
        tw_start,
        MC,
        fc.build_initial_theta(),
        bg_term=None,
        shape_coords=task["shape_coords"],  # published-domain polygon filter
    )
    calc = ETASParameterCalculation(config)
    calc.prepare()
    calc.invert()

    beta = float(calc.beta)
    mc_min = calc.m_ref - calc.delta_m / 2
    th = calc.theta
    # Fitted 8-vector in the order neg_log_likelihood / branching_ratio expect.
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
        # log10_mu, log10_iota slots are unused by branching_ratio.
        return float(branching_ratio(np.array([0.0, 0.0, *v]), b))

    n_point = n_of(theta8, beta)

    grad = np.zeros(8)
    step = np.maximum(np.abs(theta8) * 1e-5, 1e-7)
    for i in range(8):
        tp = theta8.copy(); tp[i] += step[i]
        tm = theta8.copy(); tm[i] -= step[i]
        grad[i] = (n_of(tp, beta) - n_of(tm, beta)) / (2 * step[i])
    var_params = float(grad @ cov @ grad)

    # Independent beta term (exponential MLE SE = beta / sqrt(N)).
    n_target = int(len(calc.target_events))
    se_beta = beta / np.sqrt(max(n_target, 1))
    dn_db = (n_of(theta8, beta + 1e-4) - n_of(theta8, beta - 1e-4)) / 2e-4
    se_n = float(np.sqrt(max(var_params, 0.0) + (dn_db * se_beta) ** 2))

    return {
        "window": task["window"],
        "catalog": task["catalog"],
        "n_target_events": n_target,
        "beta": beta,
        "branching_ratio": n_point,
        "se_branching_ratio": se_n,
        "ci_low": n_point - 1.96 * se_n,
        "ci_high": n_point + 1.96 * se_n,
        "excludes_unity": (n_point + 1.96 * se_n < 1.0) or (n_point - 1.96 * se_n > 1.0),
        "hessian_pos_def": pos_def,
    }


def _published_n(window: str) -> float:
    """Published (uncorrected nzcat.csv, 2021 origin) eta for a window.

    Primary source is branching_uncertainty.csv; multi_origin_branching.csv (the
    2021 origin row) is used as a cross-check / fallback.
    """
    import pandas as pd

    key = PUBLISHED_KEYS[window]
    bu_path = TABLE_DIR / "branching_uncertainty.csv"
    mo_path = TABLE_DIR / "multi_origin_branching.csv"

    n_pub = float("nan")
    if bu_path.exists():
        bu = pd.read_csv(bu_path)
        match = bu.loc[bu["scenario"] == key]
        if len(match):
            n_pub = float(match["branching_ratio"].iloc[0])

    if pd.isna(n_pub) and mo_path.exists():
        mo = pd.read_csv(mo_path)
        match = mo.loc[
            (mo["window_start"].astype(str) == window)
            & (mo["origin_year"].astype(str) == "2021")
        ]
        if len(match):
            n_pub = float(match["branching_ratio"].iloc[0])
    return n_pub


def main() -> None:
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3,
                        help="Spawn worker processes for parallel inversions. Default: 3")
    parser.add_argument("--numba-threads", type=int, default=10,
                        help="numba threads per worker. Default: 10")
    parser.add_argument(
        "--windows", nargs="*", default=list(WINDOWS),
        help="Subset of windows to fit (e.g. 2000). Default: all of 1960 1980 2000.",
    )
    parser.add_argument(
        "--catalogs", nargs="*", default=list(CATALOGS),
        choices=list(CATALOGS),
        help="Which catalog versions to fit. Default: homogeneous typed (both).",
    )
    parser.add_argument(
        "--out", default=str(TABLE_DIR / "homogenized_refit.csv"),
        help="Output CSV path. Use a *_smoke.csv name for smoke tests.",
    )
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for window in args.windows:
        for catalog in args.catalogs:
            tasks.append({
                "window": window,
                "catalog": catalog,
                "fn_catalog": CATALOGS[catalog],
                "shape_coords": PUBLISHED_POLYGON,
                "label": f"nzhomrefit_{catalog}_{window}",
                "numba_threads": args.numba_threads,
            })

    print(
        f"Refitting {len(tasks)} inversions "
        f"(windows={args.windows}, catalogs={args.catalogs}) "
        f"with {args.workers} spawn workers x {args.numba_threads} numba threads",
        flush=True,
    )

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futs = {pool.submit(_fit, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
                rows.append(r)
                print(
                    f"  {r['catalog']:11s} {r['window']}: "
                    f"n={r['branching_ratio']:.4f} "
                    f"CI[{r['ci_low']:.4f},{r['ci_high']:.4f}] "
                    f"({'super' if r['branching_ratio'] >= 1 else 'sub'}critical, "
                    f"{r['n_target_events']} ev, excl-unity={r['excludes_unity']})",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover
                print(f"  FAILED {t['catalog']} {t['window']}: {exc}", flush=True)

    if not rows:
        print("No successful inversions; nothing written.")
        return

    frame = pd.DataFrame(rows)

    # Attach the published (uncorrected nzcat.csv) eta and the shift vs. published.
    frame["n_published"] = frame["window"].map(_published_n)
    frame["delta_n"] = frame["branching_ratio"] - frame["n_published"]

    # Matched-control isolation: delta_n_isolated = n(homogeneous) - n(typed) on the
    # identical event set, per window. Defined only when both catalogs were fit;
    # attached to the homogeneous rows (NaN on typed rows).
    typed_by_window = (
        frame.loc[frame["catalog"] == "typed"]
        .set_index("window")["branching_ratio"].to_dict()
    )

    def _isolated(row):
        if row["catalog"] != "homogeneous":
            return float("nan")
        n_typed = typed_by_window.get(row["window"], float("nan"))
        return row["branching_ratio"] - n_typed

    frame["delta_n_isolated"] = frame.apply(_isolated, axis=1)

    # Stable, readable ordering: window then catalog (homogeneous before typed).
    cat_order = {"homogeneous": 0, "typed": 1}
    frame["__c"] = frame["catalog"].map(cat_order).fillna(9)
    frame = (
        frame.sort_values(["window", "__c"])
        .drop(columns="__c")
        .reset_index(drop=True)
    )

    cols = [
        "window", "catalog", "n_target_events", "branching_ratio",
        "ci_low", "ci_high", "n_published", "delta_n", "delta_n_isolated",
        "excludes_unity", "beta", "se_branching_ratio", "hessian_pos_def",
    ]
    frame = frame[[c for c in cols if c in frame.columns]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(frame.to_string(index=False))

    # Narrative: did the correction move the 1960 eta toward / below unity?
    homog = frame.loc[frame["catalog"] == "homogeneous"].set_index("window")
    typed = frame.loc[frame["catalog"] == "typed"].set_index("window")
    if "1960" in homog.index:
        h = homog.loc["1960"]
        n_homog = float(h["branching_ratio"])
        n_pub = float(h["n_published"])
        print("\n--- 1960-window magnitude-homogenization effect ---")
        if pd.notna(n_pub):
            shift_pub = n_homog - n_pub
            direction = "toward" if abs(n_homog - 1.0) < abs(n_pub - 1.0) else "away from"
            print(
                f"  vs published (nzcat.csv) n={n_pub:.4f}: homogenized n={n_homog:.4f} "
                f"(delta_n={shift_pub:+.4f}, moves {direction} unity; "
                f"{'now SUB-critical' if n_homog < 1.0 else 'still supercritical'})"
            )
        if "1960" in typed.index:
            n_typed = float(typed.loc["1960", "branching_ratio"])
            iso = n_homog - n_typed
            direction = "toward" if abs(n_homog - 1.0) < abs(n_typed - 1.0) else "away from"
            print(
                f"  matched control (typed, same events) n={n_typed:.4f}: "
                f"delta_n_isolated={iso:+.4f} -> the magnitude correction alone moves "
                f"eta {direction} unity by {abs(iso):.4f} "
                f"({'crosses below 1' if (n_typed >= 1.0 > n_homog) else 'no unity crossing'})."
            )
        else:
            print("  (typed control not fit this run; delta_n_isolated unavailable.)")


if __name__ == "__main__":
    main()
