#!/usr/bin/env python3
"""Conditional profile-likelihood intervals for the ETAS branching ratio.

For each selected fit, the EM algorithm is run to convergence. The final
triggering responsibilities are then held fixed while the eight triggering
parameters are re-optimized subject to n(theta, beta) = n_grid. This is a
conditional M-step profile likelihood, not a full catalog bootstrap: it
accounts for covariance among triggering parameters but conditions on the
final E-step responsibilities, beta, catalog, and model specification.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

SCENARIOS = {
    "baseline": ("1960-01-01 00:00:00", "1950-01-01 00:00:00"),
    "window_1980": ("1980-01-01 00:00:00", "1970-01-01 00:00:00"),
    "window_2000": ("2000-01-01 00:00:00", "1990-01-01 00:00:00"),
}
ORIGIN = dt.datetime(2021, 1, 1)
MC = 4.1


def interpolate_crossing(x1: float, y1: float, x2: float, y2: float) -> float:
    return x1 + (3.841459 - y1) * (x2 - x1) / (y2 - y1)


def confidence_limits(profile: pd.DataFrame) -> tuple[float, float, bool, bool]:
    valid = profile[profile["success"]].sort_values("n_grid")
    point = valid.loc[valid["lr_stat"].idxmin(), "n_grid"]
    lower = valid[valid["n_grid"] <= point]
    upper = valid[valid["n_grid"] >= point]

    low, low_censored = valid["n_grid"].min(), True
    for left, right in zip(lower.iloc[:-1].itertuples(), lower.iloc[1:].itertuples()):
        if left.lr_stat >= 3.841459 > right.lr_stat:
            low = interpolate_crossing(left.n_grid, left.lr_stat, right.n_grid, right.lr_stat)
            low_censored = False

    high, high_censored = valid["n_grid"].max(), True
    for left, right in zip(upper.iloc[:-1].itertuples(), upper.iloc[1:].itertuples()):
        if left.lr_stat < 3.841459 <= right.lr_stat:
            high = interpolate_crossing(left.n_grid, left.lr_stat, right.n_grid, right.lr_stat)
            high_censored = False
            break
    return low, high, low_censored, high_censored


def fit_profile(scenario: str, grid: np.ndarray) -> tuple[pd.DataFrame, dict]:
    import numba
    numba.set_num_threads(12)

    import run_nz_wide_forecast as fc
    from etas.inversion import RANGES, ETASParameterCalculation, branching_ratio
    from etas.inversion import neg_log_likelihood

    tw_start, aux_start = SCENARIOS[scenario]
    paths = fc.build_run_paths(f"nzprof_{scenario}", ORIGIN)
    config = fc.build_inversion_config(
        paths["run_label"], fc.CATALOG_PATH, ORIGIN, aux_start, tw_start, MC,
        fc.build_initial_theta(),
    )
    calc = ETASParameterCalculation(config)
    calc.prepare()
    calc.invert()

    theta = calc.theta
    fitted = np.array([
        theta["log10_k0"], theta["a"], theta["log10_c"], theta["omega"],
        theta["log10_tau"], theta["log10_d"], theta["gamma"], theta["rho"],
    ])
    beta = float(calc.beta)
    mc_min = calc.m_ref - calc.delta_m / 2
    pij = calc.pij
    sources = calc.source_events

    def objective(x: np.ndarray) -> float:
        return float(neg_log_likelihood(x, pij, sources.copy(), mc_min))

    def n_value(x: np.ndarray) -> float:
        # branching_ratio() asserts alpha - beta < 0 for unbounded magnitudes
        # (the magnitude integral diverges otherwise). SLSQP transiently probes
        # candidates with alpha = a - rho*gamma >= beta while searching for a
        # target n above the near-critical point; guard that forbidden region
        # with a large, downward-sloping penalty so the optimizer steers back
        # into alpha < beta instead of crashing.
        alpha = x[1] - x[7] * x[6]  # a - rho * gamma
        if alpha >= beta - 1e-6:
            return 1.0e8 * (1.0 + (alpha - beta + 1e-6))
        return float(branching_ratio(np.array([0.0, 0.0, *x]), beta))

    n_point = n_value(fitted)
    grid = np.unique(np.r_[grid, n_point])
    ordered = sorted(grid, key=lambda value: abs(value - n_point))
    bounds = list(RANGES[2:])
    starts = {n_point: fitted}
    rows = []
    for n_grid in ordered:
        nearest = min(starts, key=lambda value: abs(value - n_grid))
        result = minimize(
            objective,
            starts[nearest],
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda x, target=n_grid: n_value(x) - target},
            options={"maxiter": 400, "ftol": 1e-9, "disp": False},
        )
        constraint_error = abs(n_value(result.x) - n_grid)
        success = bool(result.success and constraint_error < 1e-5)
        if success:
            starts[n_grid] = result.x
        rows.append({
            "scenario": scenario,
            "n_grid": n_grid,
            "neg_log_likelihood": result.fun,
            "success": success,
            "constraint_error": constraint_error,
            "iterations": result.nit,
            "optimizer_message": result.message,
        })
        print(
            f"{scenario:11s} n={n_grid:.4f} nll={result.fun:.3f} "
            f"success={success} error={constraint_error:.2g}",
            flush=True,
        )

    profile = pd.DataFrame(rows).sort_values("n_grid")
    min_nll = profile.loc[profile["success"], "neg_log_likelihood"].min()
    profile["lr_stat"] = 2 * (profile["neg_log_likelihood"] - min_nll)
    low, high, low_censored, high_censored = confidence_limits(profile)
    summary = {
        "scenario": scenario,
        "n_point": n_point,
        "ci95_low": low,
        "ci95_high": high,
        "ci95_low_grid_censored": low_censored,
        "ci95_high_grid_censored": high_censored,
        "includes_unity": low <= 1.0 <= high,
        "n_target_events": len(calc.target_events),
        "beta_fixed": beta,
        "method": "conditional M-step profile likelihood",
    }
    return profile, summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--n-min", type=float, default=0.88)
    parser.add_argument("--n-max", type=float, default=1.12)
    parser.add_argument("--n-step", type=float, default=0.01)
    args = parser.parse_args()

    grid = np.arange(args.n_min, args.n_max + args.n_step / 2, args.n_step)
    profiles, summaries = [], []
    for scenario in args.scenarios:
        profile, summary = fit_profile(scenario, grid)
        profiles.append(profile)
        summaries.append(summary)

    table_dir = BSSA / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(profiles, ignore_index=True).to_csv(
        table_dir / "branching_profile.csv", index=False
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(table_dir / "branching_profile_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
