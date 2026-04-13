"""
Advanced ETAS Forecast Analysis Script

Improved experiment analysis:
1. Uses a primary 7-day dense-grid comparison plot.
2. Builds a formal non-overlapping evaluation grid for each horizon.
3. Scores adaptive and fixed-regional arms on the same paired metrics.
4. Writes detailed CSV/JSON summaries for reproducible comparison.
"""

import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, '.')
from date_grids import SEQUENCE_DATE_GRID_METADATA
from visualize_results import *


DEFAULT_EVALUATION_HORIZONS_DAYS = [1.0, 3.0, 7.0, 30.0]
DEFAULT_PRIMARY_HORIZON_DAYS = 7.0
DEFAULT_SIMULATION_TAG = f"{int(max(DEFAULT_EVALUATION_HORIZONS_DAYS))}d"
DEFAULT_EXPERIMENT_METADATA_PATH = os.path.join(
    "output_nz", f"experiment_config_{DEFAULT_SIMULATION_TAG}.json"
)
PRIMARY_METRIC_KEY = "delta_information_gain"
PRIMARY_METRIC_LABEL = "adaptive minus fixed information gain"
MAG_THRESHOLDS = [4.1, 4.5, 5.0, 5.5]


def load_experiment_configuration():
    """Load experiment metadata written by run_parallel_simulations.py."""
    defaults = {
        "evaluation_horizons_days": DEFAULT_EVALUATION_HORIZONS_DAYS,
        "primary_evaluation_horizon_days": DEFAULT_PRIMARY_HORIZON_DAYS,
        "simulation_horizon_days": max(DEFAULT_EVALUATION_HORIZONS_DAYS),
        "adaptive_simulation_dir": f"simulations_nz_{DEFAULT_SIMULATION_TAG}",
        "fixed_simulation_dir": f"simulations_nz_fixed_{DEFAULT_SIMULATION_TAG}",
        "sequence_date_grids": SEQUENCE_DATE_GRID_METADATA,
    }
    if not os.path.exists(DEFAULT_EXPERIMENT_METADATA_PATH):
        return defaults

    with open(DEFAULT_EXPERIMENT_METADATA_PATH, "r") as f:
        metadata = json.load(f)

    defaults.update({
        "evaluation_horizons_days": metadata.get(
            "evaluation_horizons_days", defaults["evaluation_horizons_days"]
        ),
        "primary_evaluation_horizon_days": metadata.get(
            "primary_evaluation_horizon_days",
            defaults["primary_evaluation_horizon_days"],
        ),
        "simulation_horizon_days": metadata.get(
            "simulation_horizon_days", defaults["simulation_horizon_days"]
        ),
        "adaptive_simulation_dir": metadata.get(
            "adaptive_simulation_dir", defaults["adaptive_simulation_dir"]
        ),
        "fixed_simulation_dir": metadata.get(
            "fixed_simulation_dir", defaults["fixed_simulation_dir"]
        ),
        "sequence_date_grids": metadata.get(
            "sequence_date_grids", defaults["sequence_date_grids"]
        ),
    })
    return defaults


EXPERIMENT_CONFIG = load_experiment_configuration()
EVALUATION_HORIZONS_DAYS = [
    float(h) for h in EXPERIMENT_CONFIG["evaluation_horizons_days"]
]
PRIMARY_HORIZON_DAYS = float(EXPERIMENT_CONFIG["primary_evaluation_horizon_days"])
SIMULATION_HORIZON_DAYS = float(EXPERIMENT_CONFIG["simulation_horizon_days"])
ADAPTIVE_SIM_DIR = EXPERIMENT_CONFIG["adaptive_simulation_dir"]
FIXED_SIM_DIR = EXPERIMENT_CONFIG["fixed_simulation_dir"]
SEQUENCE_DATE_GRID_INFO = EXPERIMENT_CONFIG["sequence_date_grids"]


def select_non_overlapping_windows(window_df: pd.DataFrame,
                                   horizon_days: float) -> pd.DataFrame:
    """Greedily select windows whose evaluation periods do not overlap."""
    selected = []
    next_allowed_start = None
    ordered = window_df.sort_values("forecast_start")
    for row in ordered.itertuples(index=False):
        if next_allowed_start is None or row.forecast_start >= next_allowed_start:
            selected.append({
                "model_idx": int(row.model_idx),
                "forecast_start": row.forecast_start,
                "forecast_end": row.forecast_start + timedelta(days=horizon_days),
            })
            next_allowed_start = row.forecast_start + timedelta(days=horizon_days)
    return pd.DataFrame(selected)


def build_simulation_count_distribution(simulations: pd.DataFrame,
                                        catalog_ids=None) -> pd.Series:
    """Build a count distribution including zero-event catalogs."""
    if catalog_ids is None:
        if len(simulations) == 0 or "catalog_id" not in simulations.columns:
            return pd.Series(dtype=float)
        catalog_ids = np.sort(simulations["catalog_id"].unique())

    catalog_index = pd.Index(np.asarray(catalog_ids, dtype=int), name="catalog_id")
    sim_counts = pd.Series(0.0, index=catalog_index)
    if len(simulations) > 0 and "catalog_id" in simulations.columns:
        observed_counts = simulations.groupby("catalog_id").size().astype(float)
        sim_counts.loc[observed_counts.index] = observed_counts.values
    return sim_counts


def safe_n_test(simulations: pd.DataFrame, observed_count: int,
                catalog_ids=None) -> dict:
    """Return N-test metrics even when simulations are missing."""
    sim_counts = build_simulation_count_distribution(simulations, catalog_ids)
    if len(sim_counts) == 0:
        return {
            "observed": observed_count,
            "simulated_mean": np.nan,
            "simulated_std": np.nan,
            "simulated_median": np.nan,
            "quantile": np.nan,
            "p5": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "consistent": np.nan,
            "distribution": np.array([]),
        }

    quantile = float((sim_counts < observed_count).mean())
    return {
        "observed": observed_count,
        "simulated_mean": float(sim_counts.mean()),
        "simulated_std": float(sim_counts.std()),
        "simulated_median": float(sim_counts.median()),
        "quantile": quantile,
        "p5": float(sim_counts.quantile(0.05)),
        "p25": float(sim_counts.quantile(0.25)),
        "p75": float(sim_counts.quantile(0.75)),
        "p95": float(sim_counts.quantile(0.95)),
        "consistent": bool(0.025 <= quantile <= 0.975),
        "distribution": sim_counts.values,
    }


def safe_forecast_skill(simulations: pd.DataFrame, observed: pd.DataFrame,
                        catalog_ids=None) -> dict:
    """Return forecast skill metrics even when simulations are missing."""
    sim_counts = build_simulation_count_distribution(simulations, catalog_ids)
    if len(sim_counts) == 0:
        return {
            "information_gain": np.nan,
            "brier_score": np.nan,
            "mean_rate": np.nan,
            "etas_likelihood": np.nan,
            "poisson_likelihood": np.nan,
        }

    n_obs = len(observed)
    mean_rate = float(sim_counts.mean())
    etas_likelihood = float((sim_counts == n_obs).mean())
    if etas_likelihood == 0:
        etas_likelihood = 1 / len(sim_counts)

    from scipy.stats import poisson

    poisson_likelihood = float(poisson.pmf(n_obs, mean_rate))
    information_gain = float(np.log(etas_likelihood / max(poisson_likelihood, 1e-10)))

    upper_bin = max(int(sim_counts.max()), n_obs) + 20
    bins = np.arange(0, upper_bin + 20, 20)
    if len(bins) < 2:
        bins = np.array([0, 1])
    bin_counts, _ = np.histogram(sim_counts, bins=bins, density=False)
    bin_probs = bin_counts / max(bin_counts.sum(), 1)

    obs_bin_idx = np.digitize(n_obs, bins) - 1
    obs_bin_idx = min(max(obs_bin_idx, 0), len(bin_probs) - 1)
    outcomes = np.zeros(len(bin_probs))
    outcomes[obs_bin_idx] = 1
    brier_score = float(np.mean((bin_probs - outcomes) ** 2))

    return {
        "information_gain": information_gain,
        "brier_score": brier_score,
        "mean_rate": mean_rate,
        "etas_likelihood": etas_likelihood,
        "poisson_likelihood": poisson_likelihood,
    }


def compute_metric_bundle(simulations: pd.DataFrame,
                          observed: pd.DataFrame,
                          catalog_ids=None) -> dict:
    """Compute scalar metrics for a single arm/window."""
    n_result = safe_n_test(simulations, len(observed), catalog_ids=catalog_ids).copy()
    n_result.pop("distribution", None)
    skill_result = safe_forecast_skill(
        simulations, observed, catalog_ids=catalog_ids
    )
    ltest_result = calculate_spatial_ltest(simulations, observed)
    return {
        **n_result,
        **skill_result,
        **ltest_result,
        "n_catalogs": int(len(catalog_ids)) if catalog_ids is not None else int(
            simulations["catalog_id"].nunique()
        ) if len(simulations) > 0 and "catalog_id" in simulations.columns else 0,
        "simulated_event_total": int(len(simulations)),
        "observed_event_total": int(len(observed)),
    }


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    """Prefix metric keys for adaptive/fixed arm comparison rows."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def build_paired_row(sequence: str, model_idx: int, forecast_start,
                     horizon_days: float, grid_type: str,
                     observed: pd.DataFrame, adaptive_metrics: dict,
                     fixed_metrics: dict) -> dict:
    """Combine both arms into a single paired comparison record."""
    row = {
        "sequence": sequence,
        "model_idx": int(model_idx),
        "forecast_start": forecast_start,
        "forecast_end": forecast_start + timedelta(days=horizon_days),
        "horizon_days": float(horizon_days),
        "grid_type": grid_type,
        "observed_count": int(len(observed)),
    }
    row.update(prefix_metrics(adaptive_metrics, "adaptive"))
    row.update(prefix_metrics(fixed_metrics, "fixed"))

    row["delta_information_gain"] = (
        row["adaptive_information_gain"] - row["fixed_information_gain"]
    )
    row["delta_brier_skill"] = (
        row["fixed_brier_score"] - row["adaptive_brier_score"]
    )
    row["delta_l_test_stat"] = (
        row["adaptive_l_test_stat"] - row["fixed_l_test_stat"]
    )
    row["delta_spatial_ll"] = (
        row["adaptive_spatial_ll"] - row["fixed_spatial_ll"]
    )
    row["delta_abs_count_error_skill"] = (
        abs(row["observed_count"] - row["fixed_simulated_median"])
        - abs(row["observed_count"] - row["adaptive_simulated_median"])
    )
    row["delta_quantile_error_skill"] = (
        abs(0.5 - row["fixed_quantile"])
        - abs(0.5 - row["adaptive_quantile"])
    )
    return row


def summarize_delta_series(series: pd.Series) -> dict:
    """Summarize paired deltas with sign test and bootstrap CI."""
    values = np.asarray(series, dtype=float)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return {
            "n_valid": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "mean_delta": None,
            "median_delta": None,
            "bootstrap_ci_95": [None, None],
            "sign_test_pvalue": None,
        }

    wins = int((valid > 0).sum())
    losses = int((valid < 0).sum())
    ties = int((valid == 0).sum())
    non_ties = wins + losses
    p_value = (
        float(binomtest(wins, non_ties, 0.5).pvalue)
        if non_ties > 0
        else None
    )

    rng = np.random.default_rng(42)
    bootstrap_means = np.array([
        rng.choice(valid, size=len(valid), replace=True).mean()
        for _ in range(2000)
    ])

    return {
        "n_valid": int(len(valid)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_delta": float(valid.mean()),
        "median_delta": float(np.median(valid)),
        "bootstrap_ci_95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "sign_test_pvalue": p_value,
    }


def build_formal_summary(paired_df: pd.DataFrame, sequence: str) -> dict:
    """Summarize the formal paired evaluation by horizon and metric."""
    metric_map = {
        "delta_information_gain": "Information gain",
        "delta_brier_skill": "Brier skill",
        "delta_l_test_stat": "Spatial L-test statistic",
        "delta_spatial_ll": "Spatial log-likelihood",
        "delta_abs_count_error_skill": "Absolute count error skill",
        "delta_quantile_error_skill": "N-test quantile error skill",
    }

    summary = {
        "sequence": sequence,
        "primary_metric_key": PRIMARY_METRIC_KEY,
        "primary_metric_label": PRIMARY_METRIC_LABEL,
        "primary_horizon_days": PRIMARY_HORIZON_DAYS,
        "simulation_horizon_days": SIMULATION_HORIZON_DAYS,
        "horizons": {},
    }

    for horizon_days in sorted(paired_df["horizon_days"].unique()):
        horizon_subset = paired_df[paired_df["horizon_days"] == horizon_days]
        summary["horizons"][str(horizon_days)] = {
            "n_windows": int(len(horizon_subset)),
            "metrics": {
                metric_key: {
                    "label": metric_label,
                    **summarize_delta_series(horizon_subset[metric_key]),
                }
                for metric_key, metric_label in metric_map.items()
            },
        }
    return summary


def run_advanced_analyses(sequence="Kaikoura"):
    """Execute the upgraded experiment analysis for one sequence."""
    print(f"\n{'='*70}")
    print(f"Advanced Statistical Analyses: {sequence} Sequence")
    print(f"{'='*70}\n")

    catalog = load_catalog()
    params_df = load_parameters(sequence)
    window_df = (
        params_df[["index", "date"]]
        .dropna()
        .rename(columns={"index": "model_idx", "date": "forecast_start"})
        .sort_values("forecast_start")
        .reset_index(drop=True)
    )

    print(f"Loaded {len(params_df)} adaptive parameter sets")
    print(
        f"Simulation inputs: adaptive={ADAPTIVE_SIM_DIR}, "
        f"fixed={FIXED_SIM_DIR}, simulated_horizon={SIMULATION_HORIZON_DAYS:g} days\n"
    )
    grid_info = SEQUENCE_DATE_GRID_INFO.get(sequence, {})
    if grid_info:
        print(
            f"{sequence} forecast grid: {grid_info.get('n_forecast_origins')} origins "
            f"from {grid_info.get('first_forecast_origin')} "
            f"to {grid_info.get('last_forecast_origin')}"
        )
        for phase in grid_info.get("design", []):
            print(f"   - {phase}")
        print()

    formal_windows_by_horizon = {
        float(h): select_non_overlapping_windows(window_df, float(h))
        for h in EVALUATION_HORIZONS_DAYS
    }
    formal_model_ids_by_horizon = {
        horizon: set(formal_windows["model_idx"].tolist())
        for horizon, formal_windows in formal_windows_by_horizon.items()
    }

    print("Formal non-overlapping windows by horizon:")
    for horizon in EVALUATION_HORIZONS_DAYS:
        print(
            f"   {horizon:g}d: "
            f"{len(formal_windows_by_horizon[horizon])} windows"
        )

    n_test_results_by_mag = {mag_threshold: [] for mag_threshold in MAG_THRESHOLDS}
    adaptive_primary_results = []
    fixed_primary_results = []
    adaptive_ig_results = []
    adaptive_ltest_results = []
    paired_records = []

    for row in window_df.itertuples(index=False):
        model_idx = int(row.model_idx)
        forecast_start = row.forecast_start

        adaptive_all = load_simulations(
            sequence, model_idx, sim_dir=ADAPTIVE_SIM_DIR
        )
        fixed_all = load_fixed_simulations(
            sequence, model_idx, sim_dir=FIXED_SIM_DIR
        )
        adaptive_catalog_ids = (
            np.sort(adaptive_all["catalog_id"].unique())
            if len(adaptive_all) > 0 and "catalog_id" in adaptive_all.columns
            else None
        )
        fixed_catalog_ids = (
            np.sort(fixed_all["catalog_id"].unique())
            if len(fixed_all) > 0 and "catalog_id" in fixed_all.columns
            else None
        )

        primary_end = forecast_start + timedelta(days=PRIMARY_HORIZON_DAYS)
        observed_primary = get_observed_in_window(
            catalog, forecast_start, primary_end
        )
        adaptive_primary = filter_simulations_to_window(
            adaptive_all, forecast_start, primary_end
        )
        fixed_primary = filter_simulations_to_window(
            fixed_all, forecast_start, primary_end
        )

        adaptive_primary_metrics = compute_metric_bundle(
            adaptive_primary, observed_primary, catalog_ids=adaptive_catalog_ids
        )
        adaptive_primary_metrics["date"] = forecast_start
        adaptive_primary_metrics["model_idx"] = model_idx
        adaptive_primary_results.append(adaptive_primary_metrics)
        adaptive_ig_results.append({
            "information_gain": adaptive_primary_metrics["information_gain"],
            "brier_score": adaptive_primary_metrics["brier_score"],
            "date": forecast_start,
            "model_idx": model_idx,
        })
        adaptive_ltest_results.append({
            "l_test_stat": adaptive_primary_metrics["l_test_stat"],
            "spatial_ll": adaptive_primary_metrics["spatial_ll"],
            "date": forecast_start,
            "model_idx": model_idx,
        })

        fixed_primary_metrics = compute_metric_bundle(
            fixed_primary, observed_primary, catalog_ids=fixed_catalog_ids
        )
        fixed_primary_metrics["date"] = forecast_start
        fixed_primary_metrics["model_idx"] = model_idx
        fixed_primary_results.append(fixed_primary_metrics)

        for mag_threshold in MAG_THRESHOLDS:
            sims_mag = adaptive_primary[adaptive_primary["magnitude"] >= mag_threshold]
            observed_mag = observed_primary[
                observed_primary["magnitude"] >= mag_threshold
            ]
            mag_result = safe_n_test(
                sims_mag,
                len(observed_mag),
                catalog_ids=adaptive_catalog_ids,
            )
            mag_result["date"] = forecast_start
            mag_result["model_idx"] = model_idx
            n_test_results_by_mag[mag_threshold].append(mag_result)

        for horizon_days in EVALUATION_HORIZONS_DAYS:
            if model_idx not in formal_model_ids_by_horizon[horizon_days]:
                continue

            forecast_end = forecast_start + timedelta(days=horizon_days)
            observed = get_observed_in_window(catalog, forecast_start, forecast_end)
            adaptive_window = filter_simulations_to_window(
                adaptive_all, forecast_start, forecast_end
            )
            fixed_window = filter_simulations_to_window(
                fixed_all, forecast_start, forecast_end
            )

            if adaptive_catalog_ids is None or fixed_catalog_ids is None:
                continue

            adaptive_metrics = compute_metric_bundle(
                adaptive_window, observed, catalog_ids=adaptive_catalog_ids
            )
            fixed_metrics = compute_metric_bundle(
                fixed_window, observed, catalog_ids=fixed_catalog_ids
            )
            paired_records.append(
                build_paired_row(
                    sequence,
                    model_idx,
                    forecast_start,
                    horizon_days,
                    grid_type="non_overlapping",
                    observed=observed,
                    adaptive_metrics=adaptive_metrics,
                    fixed_metrics=fixed_metrics,
                )
            )

        del adaptive_all, fixed_all, adaptive_primary, fixed_primary

    print("\n1. Running magnitude-dependent N-tests (adaptive arm, primary horizon)...")
    for mag_threshold in MAG_THRESHOLDS:
        results = n_test_results_by_mag[mag_threshold]
        valid_results = [r for r in results if pd.notna(r["consistent"])]
        consistent = sum(1 for r in valid_results if r["consistent"])
        total = len(valid_results)
        pct = 100 * consistent / total if total > 0 else 0
        print(f"   M >= {mag_threshold}: {consistent}/{total} ({pct:.0f}%) consistent")

    primary_dates = [result["date"] for result in adaptive_primary_results]
    if primary_dates:
        plot_magnitude_dependent_ntests(
            n_test_results_by_mag,
            sequence,
            primary_dates,
            os.path.join(
                OUTPUT_DIR,
                f"mag_dependent_ntest_{sequence.lower()}_{PRIMARY_HORIZON_DAYS:g}d.png",
            ),
        )

    print("\n2. Primary-horizon dense-grid summaries...")
    adaptive_ig_values = [
        result["information_gain"]
        for result in adaptive_primary_results
        if pd.notna(result["information_gain"])
    ]
    fixed_ig_values = [
        result["information_gain"]
        for result in fixed_primary_results
        if pd.notna(result["information_gain"])
    ]
    adaptive_l_values = [
        result["l_test_stat"]
        for result in adaptive_primary_results
        if pd.notna(result["l_test_stat"])
    ]
    fixed_l_values = [
        result["l_test_stat"]
        for result in fixed_primary_results
        if pd.notna(result["l_test_stat"])
    ]
    if adaptive_ig_values:
        print(f"   Adaptive mean IG: {np.mean(adaptive_ig_values):.3f}")
    if fixed_ig_values:
        print(f"   Fixed mean IG:    {np.mean(fixed_ig_values):.3f}")
    if adaptive_l_values:
        print(f"   Adaptive mean L-test: {np.mean(adaptive_l_values):.3f}")
    if fixed_l_values:
        print(f"   Fixed mean L-test:    {np.mean(fixed_l_values):.3f}")

    if primary_dates:
        plot_information_gain_timeline(
            adaptive_ig_results,
            sequence,
            primary_dates,
            os.path.join(
                OUTPUT_DIR,
                f"information_gain_{sequence.lower()}_{PRIMARY_HORIZON_DAYS:g}d.png",
            ),
        )
        plot_spatial_ltest_results(
            adaptive_ltest_results,
            sequence,
            primary_dates,
            os.path.join(
                OUTPUT_DIR,
                f"spatial_ltest_{sequence.lower()}_{PRIMARY_HORIZON_DAYS:g}d.png",
            ),
        )

    print("\n3. Comparing adaptive vs fixed parameters on the dense primary grid...")
    adaptive_primary_by_model = {
        result["model_idx"]: result for result in adaptive_primary_results
    }
    fixed_primary_by_model = {
        result["model_idx"]: result for result in fixed_primary_results
    }
    shared_dense_model_ids = [
        int(model_idx)
        for model_idx in window_df["model_idx"]
        if (
            model_idx in adaptive_primary_by_model
            and model_idx in fixed_primary_by_model
            and adaptive_primary_by_model[model_idx]["n_catalogs"] > 0
            and fixed_primary_by_model[model_idx]["n_catalogs"] > 0
        )
    ]

    if not shared_dense_model_ids:
        print("   No shared dense-grid windows found. Skipping comparison plot.")
        adaptive_dense_comparison = []
        fixed_dense_comparison = []
    else:
        adaptive_dense_comparison = [
            adaptive_primary_by_model[model_idx] for model_idx in shared_dense_model_ids
        ]
        fixed_dense_comparison = [
            fixed_primary_by_model[model_idx] for model_idx in shared_dense_model_ids
        ]
        dense_dates = [
            adaptive_primary_by_model[model_idx]["date"]
            for model_idx in shared_dense_model_ids
        ]
        adaptive_consistent = sum(
            1 for r in adaptive_dense_comparison if r["consistent"] is True
        )
        fixed_consistent = sum(
            1 for r in fixed_dense_comparison if r["consistent"] is True
        )
        print(f"   Shared dense windows: {len(shared_dense_model_ids)}")
        print(
            f"   Adaptive: {adaptive_consistent}/{len(adaptive_dense_comparison)} "
            f"consistent"
        )
        print(
            f"   Fixed:    {fixed_consistent}/{len(fixed_dense_comparison)} "
            f"consistent"
        )

        plot_adaptive_vs_fixed_comparison(
            adaptive_dense_comparison,
            fixed_dense_comparison,
            sequence,
            dense_dates,
            os.path.join(
                OUTPUT_DIR,
                f"adaptive_vs_fixed_{sequence.lower()}_{PRIMARY_HORIZON_DAYS:g}d_dense.png",
            ),
            horizon_days=PRIMARY_HORIZON_DAYS,
        )

    print("\n4. Formal paired non-overlapping evaluation...")
    paired_df = pd.DataFrame(paired_records)
    if paired_df.empty:
        print("   No paired non-overlapping windows were available.")
        formal_summary = {
            "sequence": sequence,
            "primary_metric_key": PRIMARY_METRIC_KEY,
            "primary_metric_label": PRIMARY_METRIC_LABEL,
            "primary_horizon_days": PRIMARY_HORIZON_DAYS,
            "simulation_horizon_days": SIMULATION_HORIZON_DAYS,
            "horizons": {},
        }
    else:
        paired_df.sort_values(
            by=["horizon_days", "forecast_start", "model_idx"], inplace=True
        )
        detailed_path = os.path.join(
            OUTPUT_DIR,
            f"paired_nonoverlap_metrics_{sequence.lower()}.csv",
        )
        paired_df.to_csv(detailed_path, index=False)
        print(f"   Saved paired metrics: {detailed_path}")

        formal_summary = build_formal_summary(paired_df, sequence)
        summary_path = os.path.join(
            OUTPUT_DIR,
            f"paired_nonoverlap_summary_{sequence.lower()}.json",
        )
        with open(summary_path, "w") as f:
            json.dump(formal_summary, f, indent=2)
        print(f"   Saved paired summary: {summary_path}")

        for horizon_key in sorted(formal_summary["horizons"], key=float):
            primary_summary = formal_summary["horizons"][horizon_key]["metrics"][
                PRIMARY_METRIC_KEY
            ]
            print(
                f"   {float(horizon_key):g}d: "
                f"n={primary_summary['n_valid']}, "
                f"mean {PRIMARY_METRIC_LABEL}="
                f"{primary_summary['mean_delta']:+.3f}, "
                f"wins/losses={primary_summary['wins']}/{primary_summary['losses']}, "
                f"sign-test p={primary_summary['sign_test_pvalue']}"
            )

    print(f"\n{'='*70}")
    print("Advanced analyses complete")
    print(f"{'='*70}\n")

    return {
        "mag_dependent": n_test_results_by_mag,
        "adaptive_vs_fixed_dense": (
            adaptive_dense_comparison,
            fixed_dense_comparison,
        ),
        "paired_nonoverlap": paired_df,
        "paired_summary": formal_summary,
    }


if __name__ == "__main__":
    kaikoura_advanced = run_advanced_analyses("Kaikoura")
    canterbury_advanced = run_advanced_analyses("Canterbury")

    print("\n" + "=" * 70)
    print("ADVANCED ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Primary horizon: {PRIMARY_HORIZON_DAYS:g} days")
    print(f"Evaluated horizons: {EVALUATION_HORIZONS_DAYS}")
    print(f"Simulated horizon: {SIMULATION_HORIZON_DAYS:g} days")
    print(f"Adaptive simulation dir: {ADAPTIVE_SIM_DIR}")
    print(f"Fixed simulation dir: {FIXED_SIM_DIR}")
    print(f"Forecast grid metadata source: {DEFAULT_EXPERIMENT_METADATA_PATH}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("=" * 70 + "\n")
