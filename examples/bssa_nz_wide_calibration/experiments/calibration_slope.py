#!/usr/bin/env python3
"""Experiment H6: calibration-slope diagnostic for ETAS criticality miscalibration.

Motivation
----------
The published count-calibration table reports, for every sweep scenario and five
forecast horizons T (365..1826 days), the observed target count, the mean
simulated count, and their ratio r_T = N_obs / N_mean_sim. The standard nested
N-test asks only whether the observed count falls inside the simulated count
quantiles at each horizon -- it is a per-horizon, pass/fail check that is blind
to the *trend* of the ratio across horizons. A near-critical overestimate of the
branching ratio (eta_fit slightly above 1) makes the simulated count grow faster
than reality as T lengthens, so r_T declines steadily even while individual
horizons may still "pass" the N-test.

Hypothesis
----------
The slope of ln(r_T) against horizon T estimates (eta_true - eta_fit): a
well-calibrated fit has a near-zero slope, whereas a supercritical fit
(eta_fit > 1) produces a steadily declining ratio and hence a negative slope.
A self-exciting (branching) cascade grows its expected count roughly like a power
of T, so the slope of ln(r_T) against ln(T) is a cleaner near-(eta-1) proxy than
the slope against T itself. We report BOTH the linear-in-T slope and the
log-log slope and do not overclaim that either is exactly (eta_true - eta_fit).

What this script does
---------------------
1. Reads ``tables/count_calibration.csv`` (5 horizons per scenario).
2. For each scenario, fits two ordinary-least-squares lines:
      ln(r_T) ~ T        -> slope_per_day, slope_per_1000d, intercept, R^2, SE
      ln(r_T) ~ ln(T)    -> loglog_slope, loglog_intercept, loglog_r2, loglog_se
2a. Also fits the *simulated-cascade growth rate* d ln(N_mean_sim) / d ln(T).
    Because ln(r_T) = ln(N_obs) - ln(N_sim), the cross-scenario VARIATION in the
    obs/sim slope is, for scenarios that share the same target catalog, driven
    almost entirely by how fast the simulated cascade grows with T. That growth
    rate is the cleanest near-(eta-1) proxy and it is not contaminated by the
    observed-count accumulation, which differs between the M_c scenarios (which
    use a higher-M_c, hence different, observed target catalog).
3. Merges the fitted branching ratio n and its 95% CI from
   ``tables/branching_uncertainty.csv`` (keyed on scenario name).
4. Computes the cross-scenario Pearson and Spearman correlation between each
   slope definition and the fitted n, with p-values, over (a) all 7 scenarios
   and (b) the 5 scenarios that share the common M_c=4.1 target catalog.
5. Writes ``tables/calibration_slope.csv`` (one row per scenario) and prints the
   correlation coefficients.

Important caveat (confirmed empirically on this data). The raw obs/sim slope
ranks the COMMON-M_c=4.1 scenarios cleanly with n (window_2000 least negative,
the supercritical 1960 fits most negative; Pearson |r|~0.99), but across all 7
scenarios the two M_c scenarios break the monotone relation because they change
the observed-count term. The clean, confound-free statement is: the
simulated-cascade growth rate increases monotonically with fitted n across all 7
scenarios (Pearson r~+0.88). We report all three so no view is overclaimed.

This is an analysis-only experiment: it runs NO inversions or simulations and is
essentially instantaneous. It depends only on pandas / numpy / scipy.stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- sys.path bootstrap (copied from the sibling experiment scripts) ----------
HERE = Path(__file__).resolve().parent          # experiments dir
BSSA = HERE.parent                              # bssa_nz_wide_calibration dir
EXAMPLES = BSSA.parent                          # examples dir
ROOT = EXAMPLES.parent                          # repo root
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

TABLE_DIR = BSSA / "tables"

# Canonical scenario order (matches both source tables).
SCENARIO_ORDER = [
    "baseline",
    "low_mu_k0",
    "high_mu_k0",
    "mc_4p3",
    "mc_4p5",
    "window_1980",
    "window_2000",
]


def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Ordinary least squares of y on x via scipy.stats.linregress.

    Returns slope, intercept, r^2, and the standard error of the slope. Requires
    at least three points for a meaningful R^2 and slope SE.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.allclose(x, x[0]):
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "se": np.nan}
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue) ** 2,
        "se": float(res.stderr),
    }


def compute_slopes(count_csv: Path, uncertainty_csv: Path) -> pd.DataFrame:
    """Fit the per-scenario calibration slopes and merge in the fitted n + CI."""
    counts = pd.read_csv(count_csv)
    unc = pd.read_csv(uncertainty_csv)

    # Map scenario -> (n, ci_low, ci_high, n_target, beta) from the CI table.
    unc_by_scen = unc.set_index("scenario")

    rows = []
    for scenario, grp in counts.groupby("scenario_name"):
        grp = grp.sort_values("duration_days")
        T = grp["duration_days"].to_numpy(dtype=float)
        ratio = grp["observed_to_sim_mean_ratio"].to_numpy(dtype=float)
        n_sim = grp["mean_simulated_filtered_count"].to_numpy(dtype=float)
        n_obs = grp["observed_filtered_count"].to_numpy(dtype=float)

        # Guard against non-positive ratios before taking logs.
        valid = ratio > 0
        T_v = T[valid]
        ln_r = np.log(ratio[valid])

        # Linear-in-T fit: slope has units 1/day.
        lin = _ols(T_v, ln_r)
        # Log-log fit: ln(r) on ln(T); the slope is the power-law exponent of
        # the ratio's decay and is the cleaner near-(eta-1) proxy.
        loglog = _ols(np.log(T_v), ln_r)
        # Decompose the log-log ratio slope into its growth-rate components.
        # ln(r) = ln(N_obs) - ln(N_sim), so the ratio slope is the difference of
        # the two count-growth exponents. The simulated growth exponent is the
        # confound-free near-(eta-1) proxy.
        sim_growth = _ols(np.log(T), np.log(n_sim))
        obs_growth = _ols(np.log(T), np.log(n_obs))

        row = {
            "scenario": scenario,
            "n_horizons": int(valid.sum()),
            "ratio_365": float(ratio[0]) if ratio.size else np.nan,
            "ratio_1826": float(ratio[-1]) if ratio.size else np.nan,
            # Linear-in-T fit.
            "slope_per_day": lin["slope"],
            "slope_per_1000d": lin["slope"] * 1000.0,
            "intercept": lin["intercept"],
            "r2": lin["r2"],
            "slope_se_per_day": lin["se"],
            "slope_se_per_1000d": lin["se"] * 1000.0,
            # Log-log fit.
            "loglog_slope": loglog["slope"],
            "loglog_intercept": loglog["intercept"],
            "loglog_r2": loglog["r2"],
            "loglog_slope_se": loglog["se"],
            # Confound-free decomposition: count-growth exponents.
            "sim_growth_exponent": sim_growth["slope"],
            "obs_growth_exponent": obs_growth["slope"],
        }

        # Merge in the fitted branching ratio and CI, if available.
        if scenario in unc_by_scen.index:
            u = unc_by_scen.loc[scenario]
            row["n"] = float(u["branching_ratio"])
            row["ci_low"] = float(u["ci95_low"])
            row["ci_high"] = float(u["ci95_high"])
            row["n_target_events"] = int(u["n_target_events"])
            row["beta"] = float(u["beta"])
        else:
            row["n"] = np.nan
            row["ci_low"] = np.nan
            row["ci_high"] = np.nan
            row["n_target_events"] = pd.NA
            row["beta"] = np.nan

        rows.append(row)

    frame = pd.DataFrame(rows)
    # Order rows canonically.
    frame["__o"] = frame["scenario"].map(
        {s: i for i, s in enumerate(SCENARIO_ORDER)}
    )
    frame = (
        frame.sort_values("__o", na_position="last")
        .drop(columns="__o")
        .reset_index(drop=True)
    )
    return frame


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson and Spearman correlation with p-values over paired samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out = {"n_pairs": int(x.size)}
    if x.size < 3:
        out.update(
            pearson_r=np.nan, pearson_p=np.nan,
            spearman_r=np.nan, spearman_p=np.nan,
        )
        return out
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    out.update(
        pearson_r=float(pr[0]), pearson_p=float(pr[1]),
        spearman_r=float(sr[0]), spearman_p=float(sr[1]),
    )
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="H6 calibration-slope diagnostic (analysis only)."
    )
    parser.add_argument(
        "--count-csv",
        type=Path,
        default=TABLE_DIR / "count_calibration.csv",
        help="Source count-calibration table.",
    )
    parser.add_argument(
        "--uncertainty-csv",
        type=Path,
        default=TABLE_DIR / "branching_uncertainty.csv",
        help="Source branching-ratio + CI table.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=TABLE_DIR / "calibration_slope.csv",
        help="Output table path. Use a *_smoke.csv name for smoke tests.",
    )
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    frame = compute_slopes(args.count_csv, args.uncertainty_csv)

    # The required output schema (one row per scenario). We keep the richer
    # columns in the same file for downstream use.
    out_cols = [
        "scenario",
        "slope_per_1000d",
        "slope_se_per_1000d",
        "loglog_slope",
        "loglog_slope_se",
        "sim_growth_exponent",
        "obs_growth_exponent",
        "r2",
        "loglog_r2",
        "n",
        "ci_low",
        "ci_high",
        "n_target_events",
        "beta",
        "ratio_365",
        "ratio_1826",
        "n_horizons",
    ]
    out_frame = frame[out_cols].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(args.out, index=False)

    # --- Cross-scenario correlations between each slope and fitted n ----------
    # The 5 scenarios sharing the common M_c=4.1 observed target catalog. The two
    # M_c scenarios use a different (higher-M_c) observed catalog, so their
    # observed-count growth term differs and confounds the raw obs/sim slope.
    common_mc = frame[~frame["scenario"].isin(["mc_4p3", "mc_4p5"])]

    corr_specs = [
        ("linear-in-T slope (per 1000 d) vs n", "slope_per_1000d", frame),
        ("obs/sim log-log slope        vs n", "loglog_slope", frame),
        ("sim-cascade growth exponent  vs n", "sim_growth_exponent", frame),
        ("obs/sim log-log slope (Mc=4.1 only) vs n", "loglog_slope", common_mc),
    ]

    # --- Report ---------------------------------------------------------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(f"Wrote {args.out}\n")
    print("Per-scenario calibration slopes (sorted by fitted n):")
    show = frame.sort_values("n").reset_index(drop=True)
    print(
        show[
            ["scenario", "n", "slope_per_1000d", "loglog_slope",
             "sim_growth_exponent", "r2", "ratio_365", "ratio_1826"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    print("\nCross-scenario correlation vs fitted n:")
    for label, col, df in corr_specs:
        cc = _corr(df[col].to_numpy(), df["n"].to_numpy())
        print(
            f"  {label:42s}: "
            f"Pearson r={cc['pearson_r']:+.4f} (p={cc['pearson_p']:.4g}), "
            f"Spearman rho={cc['spearman_r']:+.4f} (p={cc['spearman_p']:.4g}) "
            f"[n_pairs={cc['n_pairs']}]"
        )

    # --- Sanity highlights ----------------------------------------------------
    least_neg = frame.loc[frame["slope_per_1000d"].idxmax()]
    most_neg = frame.loc[frame["slope_per_1000d"].idxmin()]
    fastest = frame.loc[frame["sim_growth_exponent"].idxmax()]
    slowest = frame.loc[frame["sim_growth_exponent"].idxmin()]
    print(
        f"\nLeast-negative obs/sim slope: {least_neg['scenario']} "
        f"(slope/1000d={least_neg['slope_per_1000d']:+.4f}, n={least_neg['n']:.4f})"
    )
    print(
        f"Most-negative obs/sim slope:  {most_neg['scenario']} "
        f"(slope/1000d={most_neg['slope_per_1000d']:+.4f}, n={most_neg['n']:.4f})"
    )
    print(
        f"Slowest sim-cascade growth (most subcritical): {slowest['scenario']} "
        f"(exponent={slowest['sim_growth_exponent']:.4f}, n={slowest['n']:.4f})"
    )
    print(
        f"Fastest sim-cascade growth (most supercritical): {fastest['scenario']} "
        f"(exponent={fastest['sim_growth_exponent']:.4f}, n={fastest['n']:.4f})"
    )


if __name__ == "__main__":
    main()
