#!/usr/bin/env python3
"""Hypothesis 1: the observation-artifact mechanism for apparent supercriticality.

Question
--------
Can a genuinely SUBCRITICAL earthquake process, observed through New Zealand's
*time-varying* detection threshold and magnitude-scale practice, be fitted as
SUPERCRITICAL (n >= 1) when a single fixed completeness magnitude and a single
homogeneous magnitude scale are assumed -- exactly the assumptions of the
published 1960-start sweep? If so, the long-window supercriticality reported in
the manuscript is (at least partly) manufactured by the observation process, not
a property of the seismicity.

Design (controlled synthetic injection)
---------------------------------------
1. GROUND TRUTH. Take the fitted 2000-window kernel (the one admissible fit) and
   rescale k0 so the generating branching ratio is a chosen, clearly subcritical
   value (default n_true = 0.95). Because n is exactly linear in k0
   (branching_ratio has k_factor = k0 * pi/rho * d^-rho), the rescaling is exact.
   Simulate a full 1950--2021 catalog from this model over the published NZ
   polygon, with background epicentres drawn from the observed 2000--2021
   epicentre cloud (so the synthetic is spatially NZ-like). Magnitudes follow the
   fitted exponential Gutenberg--Richter law (capped at M8.5 to avoid the
   nonphysical tail). The generating n is known and < 1 by construction.

2. DEGRADE the synthetic with the real observation process, era by era:
     (a) magnitude-scale drift: add a per-era offset to the *reported* magnitude
         (early eras reported high relative to the modern scale), sign-consistent
         with NZ magnitude-homogenisation studies (Zuniga-style corrections).
     (b) era-dependent incompleteness: keep an event only if its reported
         magnitude exceeds that era's estimated completeness (from
         tables/catalog_completeness.csv, KS estimates, floored at the synthetic
         mc). Pre-1968 ~4.8, 1968--86 ~4.6, 1987--99 ~4.2, 2000+ ~4.1.

3. REFIT ETAS exactly as the published sweep did: 1960 training start, auxiliary
   1950, a single fixed mc = 4.1, homogeneous magnitudes. Compare the fitted n to
   the known generating n.

Four variants isolate the mechanism:
   control   : no drift, no era incompleteness  -> must recover n_true (validation)
   detection : era incompleteness only          -> Seif-2017 truncation bias
   magnitude : scale drift only                 -> scale-heterogeneity bias
   both      : drift + era incompleteness        -> the realistic observation process

Many independent realisations give a DISTRIBUTION of fitted n per variant. The
hypothesis is confirmed if the realistic ("both", and/or one of the single
mechanisms) variant centres at or above unity while the control recovers n_true.

Outputs
-------
tables/synthetic_injection.csv         (one row per realisation x variant)
tables/synthetic_injection_summary.csv (per-variant mean/median n, P(n>=1), etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

TABLE_DIR = BSSA / "tables"
TMP_DIR = HERE / "_synth_tmp"
CATALOG_PATH = ROOT / "input_data" / "nzcat.csv"
POLYGON_PATH = ROOT / "input_data" / "nz_polygon.npy"
ETAS_2000_PARAMS = (EXAMPLES / "output_nz_wide" /
                    "nz_wide_calibration_window_2000_20210101_000000" /
                    "parameters_nz_wide_calibration_window_2000_20210101_000000.json")
COMPLETENESS = TABLE_DIR / "catalog_completeness.csv"

GEN_START = dt.datetime(1950, 1, 1)
GEN_END = dt.datetime(2021, 1, 1)
REFIT_TW_START = "1960-01-01 00:00:00"
REFIT_AUX_START = "1950-01-01 00:00:00"
REFIT_ORIGIN = dt.datetime(2021, 1, 1)
MC = 4.1
DELTA_M = 0.1
M_MAX_GEN = 8.5

# Era boundaries (left-closed) for both degradation steps.
ERAS = [
    ("1950-01-01", "1968-01-01"),
    ("1968-01-01", "1987-01-01"),
    ("1987-01-01", "2000-01-01"),
    ("2000-01-01", "2011-01-01"),
    ("2011-01-01", "2100-01-01"),
]
# Stylised per-era magnitude-scale drift added to the reported magnitude
# (early eras read high relative to the modern scale). Sign-consistent with the
# Zuniga et al. (2005) corrections used in catalog_homogeneity.py.
MAG_DRIFT = {0: 0.30, 1: 0.20, 2: 0.10, 3: 0.0, 4: 0.0}


def load_generating_model(n_true: float, mu_boost: float):
    """Return (params_dict, beta, polygon_coords, bg_lats, bg_lons, n_generating)."""
    import numpy as np
    from etas.inversion import branching_ratio, parameter_dict2array

    with open(ETAS_2000_PARAMS) as fh:
        payload = json.load(fh)
    params = dict(payload["final_parameters"])
    beta = float(payload["beta"])

    # Rescale k0 so the generating branching ratio equals n_true (n is linear in k0).
    n_current = float(branching_ratio(parameter_dict2array(params), beta))
    params["log10_k0"] = float(params["log10_k0"] + np.log10(n_true / n_current))
    # Optionally boost the background rate so the synthetic is comparable in size
    # to the real 1960 catalog (mu does not affect the branching ratio).
    params["log10_mu"] = float(params["log10_mu"] + np.log10(mu_boost))
    n_generating = float(branching_ratio(parameter_dict2array(params), beta))

    coords = np.load(POLYGON_PATH)  # [lat, lon] vertex pairs

    # Background epicentre template: observed 2000--2021, M>=mc, inside polygon.
    from matplotlib.path import Path as MplPath
    cat = pd.read_csv(CATALOG_PATH, index_col=0, parse_dates=["time"])
    train = cat[(cat["time"] >= dt.datetime(2000, 1, 1)) &
                (cat["time"] < dt.datetime(2021, 1, 1)) &
                (cat["magnitude"] >= MC)]
    poly_path = MplPath(np.column_stack([coords[:, 1], coords[:, 0]]))
    inside = poly_path.contains_points(
        np.column_stack([train["longitude"], train["latitude"]]))
    train = train[inside]
    return (params, beta, coords,
            train["latitude"].to_numpy(), train["longitude"].to_numpy(),
            n_generating)


def era_completeness():
    """Per-era detection thresholds = max(KS completeness, mc), floored at mc."""
    comp = pd.read_csv(COMPLETENESS)
    comp = comp[comp["region"] == "all_target"].reset_index(drop=True)
    # comp rows are ordered by era matching ERAS.
    thresholds = {}
    for i, (_, _) in enumerate(ERAS):
        if i < len(comp):
            thresholds[i] = max(float(comp.iloc[i]["mc_ks"]), MC)
        else:
            thresholds[i] = MC
    return thresholds


def _era_index(times):
    import numpy as np
    idx = np.zeros(len(times), dtype=int)
    t = pd.to_datetime(times)
    for i, (a, b) in enumerate(ERAS):
        mask = (t >= pd.Timestamp(a)) & (t < pd.Timestamp(b))
        idx[mask.to_numpy()] = i
    return idx


def degrade(df, mode: str, thresholds: dict):
    """Apply scale drift and/or era incompleteness; return a degraded copy."""
    import numpy as np
    out = df.copy()
    eidx = _era_index(out["time"])
    reported = out["magnitude"].to_numpy().astype(float)

    if mode in ("magnitude", "both"):
        reported = reported + np.array([MAG_DRIFT[e] for e in eidx])
    out["magnitude"] = reported

    if mode in ("detection", "both"):
        thr = np.array([thresholds[e] for e in eidx])
    else:
        thr = np.full(len(out), MC)  # control / magnitude: flat mc floor
    keep = out["magnitude"].to_numpy() >= (thr - DELTA_M / 2)
    return out[keep].copy()


def _refit_branching(catalog_path: str, run_label: str, numba_threads: int):
    """Invert on a catalog CSV and return (n, ci_low, ci_high, n_target)."""
    import numpy as np
    import run_nz_wide_forecast as fc
    from etas.inversion import (ETASParameterCalculation, branching_ratio,
                                parameter_dict2array, neg_log_likelihood)
    from branching_uncertainty import _numerical_hessian

    config = fc.build_inversion_config(
        run_label, catalog_path, REFIT_ORIGIN,
        REFIT_AUX_START, REFIT_TW_START, MC,
        fc.build_initial_theta(), bg_term=None, shape_coords=str(POLYGON_PATH))
    calc = ETASParameterCalculation(config)
    calc.prepare()
    calc.invert()

    beta = float(calc.beta)
    mc_min = calc.m_ref - calc.delta_m / 2
    th = calc.theta
    theta8 = np.array([th["log10_k0"], th["a"], th["log10_c"], th["omega"],
                       th["log10_tau"], th["log10_d"], th["gamma"], th["rho"]], float)
    n_point = float(branching_ratio(parameter_dict2array(th), beta))
    n_target = int(len(calc.target_events))

    # Asymptotic CI (delta method), mirroring branching_uncertainty.py.
    pij, src = calc.pij, calc.source_events

    def nll(v):
        return float(neg_log_likelihood(v, pij, src.copy(), mc_min))

    def n_of(v, b):
        return float(branching_ratio(np.array([0.0, 0.0, *v]), b))

    try:
        H = _numerical_hessian(nll, theta8)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
        grad = np.zeros(8)
        step = np.maximum(np.abs(theta8) * 1e-5, 1e-7)
        for i in range(8):
            tp = theta8.copy(); tp[i] += step[i]
            tm = theta8.copy(); tm[i] -= step[i]
            grad[i] = (n_of(tp, beta) - n_of(tm, beta)) / (2 * step[i])
        var_params = float(grad @ cov @ grad)
        se_beta = beta / np.sqrt(max(n_target, 1))
        dn_db = (n_of(theta8, beta + 1e-4) - n_of(theta8, beta - 1e-4)) / 2e-4
        se_n = float(np.sqrt(max(var_params, 0.0) + (dn_db * se_beta) ** 2))
    except Exception:
        se_n = float("nan")
    return n_point, n_point - 1.96 * se_n, n_point + 1.96 * se_n, n_target, beta


def _one_realization(task: dict) -> list:
    import numba
    numba.set_num_threads(task["numba_threads"])

    import numpy as np
    from shapely.geometry import Polygon
    from etas.simulation import generate_catalog

    seed = task["seed"]
    np.random.seed(seed)

    params = task["params"]
    beta = task["beta"]
    coords = np.asarray(task["coords"])
    thresholds = task["thresholds"]
    bg_mode = task.get("bg_mode", "uniform")
    approx_times = task.get("approx_times", False)

    polygon = Polygon([(float(a), float(b)) for a, b in coords])

    # Background spatial template. "uniform" (default) draws background epicentres
    # uniformly in the polygon, so the synthetic background carries NO spurious
    # spatial clustering that the EM inversion could misread as triggering -- this
    # is what lets the control refit recover the generating n. "observed" reuses
    # the real epicentre cloud (spatially clustered, like the manuscript's own
    # forecast simulator), which inflates the recovered n.
    if bg_mode == "observed":
        bg_lats = pd.Series(np.asarray(task["bg_lats"]))
        bg_lons = pd.Series(np.asarray(task["bg_lons"]))
        bg_probs = np.full(len(bg_lats), 0.5)
    else:
        bg_lats = bg_lons = bg_probs = None  # generate_catalog -> uniform in polygon

    # --- generate the ground-truth subcritical catalog (1950--2021) ---
    synth = generate_catalog(
        polygon=polygon,
        timewindow_start=GEN_START,
        timewindow_end=GEN_END,
        parameters=params,
        mc=MC - DELTA_M / 2,
        beta_main=beta,
        delta_m=DELTA_M,
        m_max=M_MAX_GEN,
        background_lats=bg_lats,
        background_lons=bg_lons,
        background_probs=bg_probs,
        gaussian_scale=0.1,
        approx_times=approx_times,
    )
    synth = synth[["time", "latitude", "longitude", "magnitude"]].copy()
    synth = synth.sort_values("time").reset_index(drop=True)
    synth.index.name = "id"
    n_generated = len(synth)

    real_dir = TMP_DIR / f"r{seed}"
    real_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in task.get("variants", ["control", "detection", "magnitude", "both"]):
        deg = degrade(synth, mode, thresholds)
        cat_path = real_dir / f"{mode}.csv"
        deg.to_csv(cat_path, index=True)
        try:
            n_fit, lo, hi, n_tar, beta_fit = _refit_branching(
                str(cat_path), f"synth_r{seed}_{mode}", task["numba_threads"])
            ok = True
        except Exception as exc:
            n_fit = lo = hi = beta_fit = float("nan")
            n_tar = len(deg)
            ok = False
            print(f"  [seed {seed}] refit FAILED ({mode}): {exc}", flush=True)
        rows.append({
            "seed": seed, "variant": mode,
            "n_true": task["n_generating"], "n_generated_events": n_generated,
            "n_refit": n_fit, "ci_low": lo, "ci_high": hi,
            "n_target_events": n_tar, "beta_fit": beta_fit,
            "supercritical": (n_fit >= 1.0) if ok else None,
            "ok": ok,
        })
        print(f"  [seed {seed}] {mode:9s}: n_fit={n_fit:.4f} "
              f"(n_true={task['n_generating']:.3f}, {n_tar} targets)", flush=True)

    shutil.rmtree(real_dir, ignore_errors=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-realizations", type=int, default=16)
    parser.add_argument("--n-true", type=float, default=0.95,
                        help="generating branching ratio (clearly subcritical)")
    parser.add_argument("--mu-boost", type=float, default=3.0,
                        help="multiply background rate so synthetic ~ real 1960 size")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--numba-threads", type=int, default=8)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--bg-mode", choices=["uniform", "observed"], default="uniform",
                        help="background spatial template (uniform avoids spurious "
                             "clustering so the control recovers n_true)")
    parser.add_argument("--approx-times", action="store_true",
                        help="use the approximate temporal sampler (default: exact, "
                             "matching the inversion kernel)")
    parser.add_argument("--variants", nargs="*",
                        default=["control", "detection", "magnitude", "both"])
    parser.add_argument("--smoke", action="store_true",
                        help="1 realisation, short, for a quick end-to-end check")
    parser.add_argument("--out", default=None,
                        help="base output name under tables/ (e.g. synthetic_injection_uniform); "
                             "writes <out>.csv and <out>_summary.csv. Default synthetic_injection.")
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    params, beta, coords, bg_lats, bg_lons, n_generating = load_generating_model(
        args.n_true, args.mu_boost)
    thresholds = era_completeness()
    print(f"Generating model: n_true(target)={args.n_true} -> n_generating="
          f"{n_generating:.4f}, beta={beta:.4f}, mu_boost={args.mu_boost}", flush=True)
    print(f"Era detection thresholds: {thresholds}", flush=True)

    n_real = 1 if args.smoke else args.n_realizations
    tasks = [{
        "seed": args.seed0 + i,
        "params": params, "beta": beta, "coords": coords.tolist(),
        "bg_lats": bg_lats.tolist(), "bg_lons": bg_lons.tolist(),
        "n_generating": n_generating, "thresholds": thresholds,
        "numba_threads": args.numba_threads,
        "bg_mode": args.bg_mode, "approx_times": args.approx_times,
        "variants": args.variants,
    } for i in range(n_real)]

    all_rows = []
    if args.smoke or args.workers <= 1:
        for t in tasks:
            all_rows.extend(_one_realization(t))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(_one_realization, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    all_rows.extend(fut.result())
                except Exception as exc:
                    print(f"  realization FAILED: {exc}", flush=True)

    import pandas as _pd
    frame = _pd.DataFrame(all_rows)
    base = args.out if args.out else ("synthetic_injection_smoke" if args.smoke
                                      else "synthetic_injection")
    out = TABLE_DIR / f"{base}.csv"
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # Per-variant summary.
    ok = frame[frame["ok"]]
    summ = ok.groupby("variant").agg(
        n_real=("n_refit", "size"),
        mean_n_refit=("n_refit", "mean"),
        median_n_refit=("n_refit", "median"),
        std_n_refit=("n_refit", "std"),
        frac_supercritical=("supercritical", "mean"),
        mean_n_target=("n_target_events", "mean"),
    ).reset_index()
    summ["n_true"] = n_generating
    order = {"control": 0, "detection": 1, "magnitude": 2, "both": 3}
    summ = summ.sort_values("variant", key=lambda s: s.map(order)).reset_index(drop=True)
    if not args.smoke:
        summ.to_csv(TABLE_DIR / f"{base}_summary.csv", index=False)
    print("\nPer-variant summary (n_generating = %.4f):" % n_generating)
    print(summ.to_string(index=False))
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
