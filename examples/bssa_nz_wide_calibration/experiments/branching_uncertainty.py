#!/usr/bin/env python3
"""Recommendation 2: parameter uncertainty on the ETAS branching ratio.

The published sweep reports point estimates only, so it cannot say whether the
2000-window fit (n=0.969) is statistically below unity or whether the long-window
fits (n=1.02--1.03) are statistically above it. Here we attach a confidence
interval to n for every scenario.

Method. After the EM inversion converges, we form the observed information of the
M-step triggering log-likelihood by numerically differentiating
``neg_log_likelihood`` twice with respect to the eight triggering parameters
(log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho) at the fitted
solution. Its inverse is the asymptotic covariance of those parameters,
conditional on the EM-converged background probabilities (a standard profile /
conditional approximation). The branching ratio is a closed-form function
n(theta, beta); we propagate the covariance to n with the delta method and add an
independent term for beta, whose exponential-MLE standard error is beta/sqrt(N)
with N the number of target events. The 95% interval is n +/- 1.96 SE(n).

Outputs ``tables/branching_uncertainty.csv``.
"""

from __future__ import annotations

import json
import os
import sys
import copy
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

SCENARIOS = {
    "baseline": ("1960-01-01 00:00:00", "1950-01-01 00:00:00", 4.1),
    "low_mu_k0": ("1960-01-01 00:00:00", "1950-01-01 00:00:00", 4.1),
    "high_mu_k0": ("1960-01-01 00:00:00", "1950-01-01 00:00:00", 4.1),
    "mc_4p3": ("1960-01-01 00:00:00", "1950-01-01 00:00:00", 4.3),
    "mc_4p5": ("1960-01-01 00:00:00", "1950-01-01 00:00:00", 4.5),
    "window_1980": ("1980-01-01 00:00:00", "1970-01-01 00:00:00", 4.1),
    "window_2000": ("2000-01-01 00:00:00", "1990-01-01 00:00:00", 4.1),
}
DELTAS = {  # (log10_mu_delta, log10_k0_delta) matching the published scenarios
    "low_mu_k0": (-0.5, -0.25),
    "high_mu_k0": (0.5, 0.25),
}
ORIGIN = "2021-01-01 00:00:00"


def _numerical_hessian(func, x, rel_step=1e-4, abs_floor=1e-6):
    """Central-difference Hessian of a scalar function at x (1-D array)."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.maximum(np.abs(x) * rel_step, abs_floor)
    f0 = func(x)
    H = np.zeros((n, n))
    # Diagonal
    for i in range(n):
        xp = x.copy(); xp[i] += h[i]
        xm = x.copy(); xm[i] -= h[i]
        H[i, i] = (func(xp) - 2 * f0 + func(xm)) / (h[i] ** 2)
    # Off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = (func(xpp) - func(xpm) - func(xmp) + func(xmm)) / (4 * h[i] * h[j])
            H[j, i] = H[i, j]
    return H


def _branching_uncertainty(task: dict) -> dict:
    import numba
    numba.set_num_threads(task["numba_threads"])

    import numpy as np
    import run_nz_wide_forecast as fc
    from etas.inversion import (
        ETASParameterCalculation,
        branching_ratio,
        neg_log_likelihood,
    )

    scenario = task["scenario"]
    tw_start, aux_start, mc = task["tw_start"], task["aux_start"], task["mc"]
    mu_delta, k0_delta = DELTAS.get(scenario, (0.0, 0.0))

    import datetime as dt
    forecast_start = dt.datetime.strptime(ORIGIN, "%Y-%m-%d %H:%M:%S")
    paths = fc.build_run_paths(f"nzbu_{scenario}", forecast_start)
    config = fc.build_inversion_config(
        paths["run_label"], fc.CATALOG_PATH, forecast_start,
        aux_start, tw_start, mc,
        fc.build_initial_theta(mu_delta, k0_delta), bg_term=None,
    )
    calc = ETASParameterCalculation(config)
    calc.prepare()
    calc.invert()

    beta = float(calc.beta)
    mc_min = calc.m_ref - calc.delta_m / 2
    # Fitted 8-vector in the order neg_log_likelihood expects.
    theta = calc.theta
    theta8 = np.array([
        theta["log10_k0"], theta["a"], theta["log10_c"], theta["omega"],
        theta["log10_tau"], theta["log10_d"], theta["gamma"], theta["rho"],
    ], dtype=float)

    pij = calc.pij
    src = calc.source_events

    def nll(theta8_vec):
        # neg_log_likelihood mutates source_events["G"]; pass a copy.
        return float(neg_log_likelihood(theta8_vec, pij, src.copy(), mc_min))

    H = _numerical_hessian(nll, theta8)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    # n as a function of the 8 triggering params (mu, iota slots are unused by n).
    def n_of_theta8(theta8_vec, beta_val):
        theta10 = np.array([0.0, 0.0, *theta8_vec], dtype=float)
        return float(branching_ratio(theta10, beta_val))

    n_point = n_of_theta8(theta8, beta)

    # Gradient of n w.r.t. the 8 params (central difference).
    grad = np.zeros(8)
    step = np.maximum(np.abs(theta8) * 1e-5, 1e-7)
    for i in range(8):
        tp = theta8.copy(); tp[i] += step[i]
        tm = theta8.copy(); tm[i] -= step[i]
        grad[i] = (n_of_theta8(tp, beta) - n_of_theta8(tm, beta)) / (2 * step[i])

    var_params = float(grad @ cov @ grad)

    # Independent beta contribution (exponential MLE SE = beta / sqrt(N)),
    # with N the number of target events used to estimate beta.
    n_target = int(len(calc.target_events))
    se_beta = beta / np.sqrt(max(n_target, 1))
    dn_dbeta = (n_of_theta8(theta8, beta + 1e-4) - n_of_theta8(theta8, beta - 1e-4)) / 2e-4
    var_beta = float((dn_dbeta * se_beta) ** 2)

    var_n = max(var_params, 0.0) + var_beta
    se_n = float(np.sqrt(var_n))

    return {
        "scenario": scenario,
        "n_target_events": n_target,
        "beta": beta,
        "branching_ratio": n_point,
        "se_branching_ratio": se_n,
        "se_from_params": float(np.sqrt(max(var_params, 0.0))),
        "se_from_beta": float(np.sqrt(var_beta)),
        "ci95_low": n_point - 1.96 * se_n,
        "ci95_high": n_point + 1.96 * se_n,
        "excludes_unity": (n_point + 1.96 * se_n < 1.0) or (n_point - 1.96 * se_n > 1.0),
        "hessian_pos_def": bool(np.all(np.linalg.eigvalsh(H) > 0)),
    }


def main() -> None:
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--numba-threads", type=int, default=12)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "scenario": s,
            "tw_start": SCENARIOS[s][0],
            "aux_start": SCENARIOS[s][1],
            "mc": SCENARIOS[s][2],
            "numba_threads": args.numba_threads,
        }
        for s in args.scenarios
    ]
    print(f"Computing branching-ratio CIs for {len(tasks)} scenarios "
          f"({args.workers} workers x {args.numba_threads} numba threads)", flush=True)

    import multiprocessing as mp
    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_branching_uncertainty, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"  {row['scenario']:12s} n={row['branching_ratio']:.4f} "
                      f"+/- {row['se_branching_ratio']:.4f}  "
                      f"95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]  "
                      f"excludes 1: {row['excludes_unity']}", flush=True)
            except Exception as exc:  # pragma: no cover
                print(f"  FAILED {t['scenario']}: {exc}", flush=True)

    order = list(SCENARIOS)
    frame = pd.DataFrame(rows)
    frame["__o"] = frame["scenario"].map({s: i for i, s in enumerate(order)})
    frame = frame.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    out = TABLE_DIR / "branching_uncertainty.csv"
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
