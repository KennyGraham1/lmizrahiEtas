#!/usr/bin/env python3
"""Reviewer comment 1: refit the window-start models on the homogenized,
dateline-aware catalog and test whether the admissibility pattern survives.

The published sweep fits an unhomogenized catalog (nzcat.csv) on a domain
truncated at 180 deg E. This script refits the three training-window starts on
the team's homogenized, buffered catalog (nzcat_buffered_homogeneous.csv;
Zuniga-2005 deep-event corrections), under two source domains:
  - original  (165--180 deg E): isolates the homogenization effect
  - dateline  (165--184 deg E): homogenization + dateline-aware domain combined

For each fit it reports the branching ratio with the same asymptotic
Hessian/delta-method 95% interval used elsewhere, so the results are directly
comparable to Figure 6 of the manuscript. Note that the Zuniga corrections only
touch pre-1987 deep events, so the 2000-window training set is unchanged by
homogenization and moves only under the dateline domain.

Outputs tables/homogenized_refit.csv.
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
ORIG_POLYGON = str(ROOT / "input_data" / "nz_polygon.npy")          # 165-180
DATELINE_POLYGON = str(ROOT / "input_data" / "nz_polygon_dateline.npy")  # 165-184

WINDOWS = {
    "1960": ("1960-01-01 00:00:00", "1950-01-01 00:00:00"),
    "1980": ("1980-01-01 00:00:00", "1970-01-01 00:00:00"),
    "2000": ("2000-01-01 00:00:00", "1990-01-01 00:00:00"),
}

from branching_uncertainty import _numerical_hessian  # noqa: E402


def _fit(task: dict) -> dict:
    import numba
    numba.set_num_threads(task["numba_threads"])
    import numpy as np
    import run_nz_wide_forecast as fc
    from etas.inversion import (
        ETASParameterCalculation, branching_ratio, neg_log_likelihood,
    )

    tw_start, aux_start = WINDOWS[task["window"]]
    forecast_start = dt.datetime(2021, 1, 1)
    paths = fc.build_run_paths(task["label"], forecast_start)
    config = fc.build_inversion_config(
        paths["run_label"], task["fn_catalog"], forecast_start,
        aux_start, tw_start, 4.1, fc.build_initial_theta(), bg_term=None,
    )
    config["shape_coords"] = task["shape_coords"]
    calc = ETASParameterCalculation(config)
    calc.prepare()
    calc.invert()

    beta = float(calc.beta)
    mc_min = calc.m_ref - calc.delta_m / 2
    th = calc.theta
    theta8 = np.array([th["log10_k0"], th["a"], th["log10_c"], th["omega"],
                       th["log10_tau"], th["log10_d"], th["gamma"], th["rho"]], float)
    pij, src = calc.pij, calc.source_events

    def nll(v):
        return float(neg_log_likelihood(v, pij, src.copy(), mc_min))

    H = _numerical_hessian(nll, theta8)
    try:
        cov = np.linalg.inv(H); pos_def = bool(np.all(np.linalg.eigvalsh(H) > 0))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H); pos_def = False
    if not np.all(np.linalg.eigvalsh(H) > 0):
        cov = np.linalg.pinv(H); pos_def = False

    def n_of(v, b):
        return float(branching_ratio(np.array([0.0, 0.0, *v]), b))

    n_point = n_of(theta8, beta)
    grad = np.zeros(8)
    step = np.maximum(np.abs(theta8) * 1e-5, 1e-7)
    for i in range(8):
        tp = theta8.copy(); tp[i] += step[i]
        tm = theta8.copy(); tm[i] -= step[i]
        grad[i] = (n_of(tp, beta) - n_of(tm, beta)) / (2 * step[i])
    var_params = float(grad @ cov @ grad)
    n_target = int(len(calc.target_events))
    se_beta = beta / np.sqrt(max(n_target, 1))
    dn_db = (n_of(theta8, beta + 1e-4) - n_of(theta8, beta - 1e-4)) / 2e-4
    se_n = float(np.sqrt(max(var_params, 0.0) + (dn_db * se_beta) ** 2))
    return {
        "window": task["window"], "domain": task["domain"],
        "n_target_events": n_target, "beta": beta, "branching_ratio": n_point,
        "se_branching_ratio": se_n, "ci95_low": n_point - 1.96 * se_n,
        "ci95_high": n_point + 1.96 * se_n,
        "excludes_unity": (n_point + 1.96 * se_n < 1) or (n_point - 1.96 * se_n > 1),
        "hessian_pos_def": pos_def,
    }


def main():
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--numba-threads", type=int, default=10)
    args = parser.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for window in WINDOWS:
        for domain, poly in (("original", ORIG_POLYGON), ("dateline", DATELINE_POLYGON)):
            tasks.append({
                "window": window, "domain": domain,
                "fn_catalog": HOMOG_CATALOG, "shape_coords": poly,
                "label": f"nzhom_{window}_{domain}",
                "numba_threads": args.numba_threads,
            })
    print(f"Refitting {len(tasks)} homogenized fits ({args.workers} workers)", flush=True)

    import multiprocessing as mp
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(_fit, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result(); rows.append(r)
                print(f"  {r['window']} {r['domain']:8s}: n={r['branching_ratio']:.4f} "
                      f"CI[{r['ci95_low']:.4f},{r['ci95_high']:.4f}] "
                      f"({'super' if r['branching_ratio']>=1 else 'sub'}critical, "
                      f"{r['n_target_events']} ev)", flush=True)
            except Exception as exc:
                print(f"  FAILED {t['window']} {t['domain']}: {exc}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["window", "domain"]).reset_index(drop=True)
    out = TABLE_DIR / "homogenized_refit.csv"
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
