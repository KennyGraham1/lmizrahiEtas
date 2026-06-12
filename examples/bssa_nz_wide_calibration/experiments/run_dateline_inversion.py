#!/usr/bin/env python3
"""Recommendation 9 (partial): branching-ratio sensitivity to the dateline boundary.

The published domain is bounded at 180 deg E, which excludes an active offshore
source zone immediately east of the artificial boundary (GeoNet reports ~2,730
M>=4 events in 180--184 deg E within the latitude band, 1,156 of them in the
2000--2021 window at M>=4.1). This script re-fits the admissible 2000-window
model on a buffered, dateline-aware domain (165--184 deg E, longitudes unwrapped
to a continuous range so the great-circle distances remain correct) and reports
the branching ratio with the same asymptotic confidence interval used in
``branching_uncertainty.py``.

A full forecast/scoring re-run on the extended domain additionally requires
antimeridian-aware pyCSEP regions and is left as a scoped follow-up; this script
isolates the first-order question of whether including the excluded sources
changes admissibility.

Outputs ``tables/dateline_inversion.csv``.
"""

from __future__ import annotations

import json
import sys
import datetime as dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

TABLE_DIR = BSSA / "tables"
DATELINE_CATALOG = str(ROOT / "input_data" / "nzcat_dateline.csv")
DATELINE_POLYGON = str(ROOT / "input_data" / "nz_polygon_dateline.npy")

# Reuse the CI machinery from the branching-uncertainty experiment.
from branching_uncertainty import _numerical_hessian  # noqa: E402


def fit_and_ci(fn_catalog, shape_coords, label, numba_threads=12):
    import numba
    numba.set_num_threads(numba_threads)
    import numpy as np
    import run_nz_wide_forecast as fc
    from etas.inversion import (
        ETASParameterCalculation, branching_ratio, neg_log_likelihood,
    )

    forecast_start = dt.datetime(2021, 1, 1)
    paths = fc.build_run_paths(label, forecast_start)
    config = fc.build_inversion_config(
        paths["run_label"], fn_catalog, forecast_start,
        "1990-01-01 00:00:00", "2000-01-01 00:00:00", 4.1,
        fc.build_initial_theta(), bg_term=None,
    )
    config["shape_coords"] = shape_coords
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
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

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
        "label": label, "n_target_events": n_target, "beta": beta,
        "branching_ratio": n_point, "se_branching_ratio": se_n,
        "ci95_low": n_point - 1.96 * se_n, "ci95_high": n_point + 1.96 * se_n,
        "excludes_unity": (n_point + 1.96 * se_n < 1) or (n_point - 1.96 * se_n > 1),
    }


def main():
    import pandas as pd
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    row = fit_and_ci(DATELINE_CATALOG, DATELINE_POLYGON, "nzdl_window_2000")
    print("Dateline-extended 2000-window fit:")
    print(f"  n_target={row['n_target_events']}  n={row['branching_ratio']:.4f} "
          f"+/- {row['se_branching_ratio']:.4f}  CI [{row['ci95_low']:.4f},"
          f" {row['ci95_high']:.4f}]  excludes 1: {row['excludes_unity']}")
    # Attach the published bounded-domain values for comparison.
    bounded = {"label": "bounded_window_2000_published", "n_target_events": 5747,
               "branching_ratio": 0.969049, "se_branching_ratio": 0.015660,
               "ci95_low": 0.938355, "ci95_high": 0.999742, "excludes_unity": True}
    frame = pd.DataFrame([bounded, row])
    out = TABLE_DIR / "dateline_inversion.csv"
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
