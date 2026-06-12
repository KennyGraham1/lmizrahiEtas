#!/usr/bin/env python3
"""Hypothesis 2: the background--triggering degeneracy ladder.

Question
--------
Is the ETAS branching ratio an identifiable property of the catalog, or is it a
monotone function of how many degrees of freedom the *background* model is given?
The manuscript already holds the two endpoints: a homogeneous background gives the
supercritical 1960-window fit (n ~ 1.03), while a sharp spatial covariate
background (the earlier hft_background_rate run) collapsed triggering entirely
(n -> 0, background-only Poisson). This experiment fills in the ladder between
them and asks where an out-of-sample criterion places the "right" flexibility.

Design
------
The ETAS inversion supports a background covariate ``bg_term``: each target's
background intensity becomes  mu + iota * bg_term_j , with both mu and iota
fitted (etas/inversion.py expectation_step). We build a smoothed-seismicity
covariate from the training epicentres at a ladder of Gaussian bandwidths sigma.

  - large sigma  -> nearly flat covariate -> behaves like homogeneous mu -> high n
  - small sigma  -> spiky covariate that sits on each training epicentre ->
                    background over-explains the catalogue -> triggering and n collapse

So sigma is an inverse-degrees-of-freedom knob. PART A fits the ETAS ladder and
records n(sigma). PART B selects the background flexibility purely out-of-sample:
a smoothed-seismicity density built on 1960--2015 epicentres is scored by the
held-out spatial Poisson log-likelihood of 2015--2021 epicentres (the standard
smoothed-seismicity cross-validation, independent of ETAS). The sigma that
maximises the held-out score is the data-supported background flexibility; we
then report the ETAS n at that sigma.

Claim under test: n declines monotonically as sigma shrinks (DoF grows), crossing
unity; the homogeneous fit's supercriticality is an artefact of an
under-flexible background, the sharp fit's collapse is over-fitting, and the
CV-selected flexibility sits near the stationarity boundary -- i.e. the catalogue
does not pin down criticality.

Outputs
-------
tables/background_ladder.csv  (one row per sigma: n, iota, induced fraction, CV score)
"""

from __future__ import annotations

import argparse
import json
import sys
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

TABLE_DIR = BSSA / "tables"
TMP_DIR = HERE / "_ladder_tmp"
CATALOG_PATH = ROOT / "input_data" / "nzcat.csv"
POLYGON_PATH = ROOT / "input_data" / "nz_polygon.npy"

TW_START = dt.datetime(1960, 1, 1)
AUX_START = "1950-01-01 00:00:00"
ORIGIN = dt.datetime(2021, 1, 1)
CV_SPLIT = dt.datetime(2015, 1, 1)
MC = 4.1
GRID = 0.1
SIGMAS = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1]  # finite-bandwidth covariates; inf = homogeneous
FLOOR_FRAC = 0.02


def _poly_path():
    from matplotlib.path import Path as MplPath
    coords = np.load(POLYGON_PATH)
    return MplPath(np.column_stack([coords[:, 1], coords[:, 0]])), coords


def build_density(epi_lon, epi_lat, sigma_deg, coords):
    """Grid-based smoothed-seismicity density (normalised over in-polygon cells)."""
    from scipy.ndimage import gaussian_filter
    from matplotlib.path import Path as MplPath
    poly_path = MplPath(np.column_stack([coords[:, 1], coords[:, 0]]))
    lats, lons = coords[:, 0], coords[:, 1]
    lon_edges = np.arange(lons.min(), lons.max() + GRID, GRID)
    lat_edges = np.arange(lats.min(), lats.max() + GRID, GRID)
    counts, _, _ = np.histogram2d(epi_lon, epi_lat, bins=[lon_edges, lat_edges])
    smoothed = gaussian_filter(counts, sigma=sigma_deg / GRID, mode="constant")
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c, indexing="ij")
    in_poly = poly_path.contains_points(
        np.column_stack([LON.ravel(), LAT.ravel()])).reshape(LON.shape)
    dens = np.where(in_poly, smoothed, 0.0)
    if dens.sum() <= 0:
        raise RuntimeError("empty density")
    dens = dens / dens.sum()
    uni = in_poly.astype(float); uni = uni / uni.sum()
    prob = (1 - FLOOR_FRAC) * dens + FLOOR_FRAC * uni
    prob = prob / prob.sum()
    return {"lon_edges": lon_edges, "lat_edges": lat_edges, "prob": prob,
            "in_poly": in_poly, "n_cells": int(in_poly.sum())}


def lookup_density(dens, lon, lat):
    li = np.clip(np.digitize(lon, dens["lon_edges"]) - 1, 0, dens["prob"].shape[0] - 1)
    lj = np.clip(np.digitize(lat, dens["lat_edges"]) - 1, 0, dens["prob"].shape[1] - 1)
    return dens["prob"][li, lj]


def load_catalog_in_polygon():
    cat = pd.read_csv(CATALOG_PATH, index_col=0, parse_dates=["time"])
    poly_path, coords = _poly_path()
    inside = poly_path.contains_points(
        np.column_stack([cat["longitude"], cat["latitude"]]))
    return cat[inside].copy(), coords


def _fit_sigma(task: dict) -> dict:
    import numba
    numba.set_num_threads(task["numba_threads"])
    import run_nz_wide_forecast as fc
    from etas.inversion import (ETASParameterCalculation, branching_ratio,
                                parameter_dict2array)

    sigma = task["sigma"]
    cat = pd.read_csv(task["aug_catalog"], index_col=0, parse_dates=["time"])

    if sigma is None:  # homogeneous background, no covariate
        config = fc.build_inversion_config(
            f"ladder_homog", task["aug_catalog"], ORIGIN, AUX_START,
            TW_START.strftime("%Y-%m-%d %H:%M:%S"), MC, fc.build_initial_theta(),
            bg_term=None, shape_coords=str(POLYGON_PATH))
    else:
        config = fc.build_inversion_config(
            f"ladder_s{sigma}", task["aug_catalog"], ORIGIN, AUX_START,
            TW_START.strftime("%Y-%m-%d %H:%M:%S"), MC, fc.build_initial_theta(),
            bg_term="bg_smooth", shape_coords=str(POLYGON_PATH))

    calc = ETASParameterCalculation(config)
    calc.prepare()
    calc.invert()
    n = float(branching_ratio(parameter_dict2array(calc.theta), calc.beta))
    th = calc.theta
    te = calc.target_events
    p_bg = float(te["P_background"].mean()) if "P_background" in te else np.nan
    p_ind = float(te["P_induced"].mean()) if "P_induced" in te else 0.0
    return {
        "sigma_deg": (np.inf if sigma is None else sigma),
        "background": ("homogeneous" if sigma is None else f"smoothed_{sigma}deg"),
        "n_branching": n,
        "log10_mu": float(th["log10_mu"]),
        "log10_iota": (float(th["log10_iota"]) if th.get("log10_iota") is not None else np.nan),
        "mean_P_background": p_bg,
        "mean_P_induced": p_ind,
        "n_target_events": int(len(te)),
        "n_iterations": int(calc.__dict__.get("n_iterations", -1)) if hasattr(calc, "__dict__") else -1,
    }


def cross_validate(cat, coords):
    """Held-out spatial Poisson log-lik per event of 2015--2021 epicentres under a
    1960--2015 smoothed density, for each bandwidth. Returns dict sigma->score."""
    train = cat[(cat["time"] >= TW_START) & (cat["time"] < CV_SPLIT)
                & (cat["magnitude"] >= MC)]
    holdout = cat[(cat["time"] >= CV_SPLIT) & (cat["time"] < ORIGIN)
                  & (cat["magnitude"] >= MC)]
    cell_area_deg2 = GRID * GRID
    scores = {}
    for sigma in SIGMAS:
        dens = build_density(train["longitude"].to_numpy(), train["latitude"].to_numpy(),
                             sigma, coords)
        # probability density per unit area; per-event log-likelihood of holdout.
        p = lookup_density(dens, holdout["longitude"].to_numpy(),
                           holdout["latitude"].to_numpy())
        pdf = p / cell_area_deg2
        scores[sigma] = float(np.mean(np.log(np.clip(pdf, 1e-12, None))))
    return scores, len(train), len(holdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--numba-threads", type=int, default=8)
    parser.add_argument("--sigmas", type=float, nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    global SIGMAS
    if args.sigmas:
        SIGMAS = list(args.sigmas)
    if args.smoke:
        SIGMAS = [1.0]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    cat, coords = load_catalog_in_polygon()

    # Build the covariate from the full 1960--2021 training epicentres for the ETAS
    # ladder (the in-sample background field), one augmented catalog per sigma.
    train_epi = cat[(cat["time"] >= TW_START) & (cat["time"] < ORIGIN)
                    & (cat["magnitude"] >= MC)]
    aug_paths = {}
    for sigma in SIGMAS:
        dens = build_density(train_epi["longitude"].to_numpy(),
                             train_epi["latitude"].to_numpy(), sigma, coords)
        aug = cat.copy()
        aug["bg_smooth"] = lookup_density(dens, aug["longitude"].to_numpy(),
                                          aug["latitude"].to_numpy())
        aug["bg_smooth"] = aug["bg_smooth"].clip(lower=1e-12)
        path = TMP_DIR / f"aug_s{sigma}.csv"
        aug.to_csv(path, index=True)
        aug_paths[sigma] = str(path)

    tasks = [{"sigma": s, "aug_catalog": aug_paths[s],
              "numba_threads": args.numba_threads} for s in SIGMAS]
    if not args.smoke:
        # homogeneous reference (no covariate); reuse the smallest-sigma aug file (column ignored).
        tasks.insert(0, {"sigma": None, "aug_catalog": aug_paths[SIGMAS[0]],
                         "numba_threads": args.numba_threads})

    print(f"Fitting background ladder: homogeneous + sigmas {SIGMAS}", flush=True)
    rows = []
    if args.smoke or args.workers <= 1:
        for t in tasks:
            rows.append(_fit_sigma(t))
            r = rows[-1]
            print(f"  {r['background']:18s}: n={r['n_branching']:.4f} "
                  f"P_ind={r['mean_P_induced']:.3f} P_bg={r['mean_P_background']:.3f}",
                  flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(_fit_sigma, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    rows.append(r)
                    print(f"  {r['background']:18s}: n={r['n_branching']:.4f} "
                          f"P_ind={r['mean_P_induced']:.3f} P_bg={r['mean_P_background']:.3f}",
                          flush=True)
                except Exception as exc:
                    print(f"  FAILED {futures[fut]['sigma']}: {exc}", flush=True)

    frame = pd.DataFrame(rows).sort_values("sigma_deg", ascending=False).reset_index(drop=True)

    # Cross-validation selection of background flexibility.
    if not args.smoke:
        cv_scores, n_tr, n_ho = cross_validate(cat, coords)
        frame["cv_holdout_loglik_per_event"] = frame["sigma_deg"].map(
            lambda s: cv_scores.get(s, np.nan))
        best_sigma = max(cv_scores, key=cv_scores.get)
        print(f"\nCV (train {n_tr} -> holdout {n_ho} events): best sigma = "
              f"{best_sigma} deg (held-out logL/event {cv_scores[best_sigma]:.3f})")
        sel = frame[np.isclose(frame["sigma_deg"], best_sigma)]
        if len(sel):
            print(f"ETAS n at CV-selected sigma={best_sigma}: "
                  f"{sel.iloc[0]['n_branching']:.4f}")

    out = TABLE_DIR / ("background_ladder_smoke.csv" if args.smoke
                       else "background_ladder.csv")
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(frame.to_string(index=False))

    import shutil
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
