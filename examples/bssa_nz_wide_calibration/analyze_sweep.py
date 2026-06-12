#!/usr/bin/env python3
"""Reproduce manuscript tables and figures from the NZ-wide calibration sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, SymLogNorm

from etas.inversion import branching_ratio


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BATCH = (
    ROOT
    / "examples"
    / "output_nz_wide_calibration"
    / "nz_wide_calibration_20210101_000000"
)
OUTPUT_ROOT = ROOT / "examples" / "output_nz_wide"
SIM_ROOT = ROOT / "examples" / "simulations_nz_wide"
TABLE_DIR = HERE / "tables"
FIGURE_DIR = HERE / "figures"

SCENARIO_ORDER = [
    "baseline",
    "low_mu_k0",
    "high_mu_k0",
    "mc_4p3",
    "mc_4p5",
    "window_1980",
    "window_2000",
]
import sys as _sys
if str(HERE) not in _sys.path:
    _sys.path.insert(0, str(HERE))
import figstyle

DISPLAY_NAMES = figstyle.DISPLAY_NAMES
COLORS = figstyle.COLORS


def setup() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    figstyle.apply()


def run_label(scenario: str) -> str:
    return f"nz_wide_calibration_{scenario}_20210101_000000"


def parameter_path(scenario: str) -> Path:
    label = run_label(scenario)
    return OUTPUT_ROOT / label / f"parameters_{label}.json"


def simulation_path(scenario: str, duration: int = 1826) -> Path:
    return SIM_ROOT / run_label(scenario) / f"forecasts_{duration}days.csv"


def observed_path(scenario: str, duration: int = 1826) -> Path:
    return OUTPUT_ROOT / run_label(scenario) / f"observed_{duration}days.csv"


def load_parameter_table() -> pd.DataFrame:
    rows = []
    for scenario in SCENARIO_ORDER:
        with parameter_path(scenario).open() as handle:
            payload = json.load(handle)
        theta = payload["final_parameters"]
        alpha = theta["a"] - theta["rho"] * theta["gamma"]
        theta_array = [
            theta["log10_mu"],
            np.nan if theta["log10_iota"] is None else theta["log10_iota"],
            theta["log10_k0"],
            theta["a"],
            theta["log10_c"],
            theta["omega"],
            theta["log10_tau"],
            theta["log10_d"],
            theta["gamma"],
            theta["rho"],
        ]
        rows.append(
            {
                "scenario": scenario,
                "display_name": DISPLAY_NAMES[scenario].replace("$", ""),
                "mc": payload["mc"],
                "training_start": payload["timewindow_start"][:10],
                "n_training": payload["n_target_events"],
                "beta": payload["beta"],
                "b_value": payload["beta"] / np.log(10),
                "branching_ratio": payload["branching_ratio"],
                "branching_ratio_mmax_7p5": branching_ratio(
                    theta_array, payload["beta"], dm_max=7.5 - payload["mc"]
                ),
                "branching_ratio_mmax_8p0": branching_ratio(
                    theta_array, payload["beta"], dm_max=8.0 - payload["mc"]
                ),
                "branching_ratio_mmax_8p5": branching_ratio(
                    theta_array, payload["beta"], dm_max=8.5 - payload["mc"]
                ),
                "branching_ratio_mmax_9p0": branching_ratio(
                    theta_array, payload["beta"], dm_max=9.0 - payload["mc"]
                ),
                "degenerate": payload["inversion_degenerate"],
                "log10_mu": theta["log10_mu"],
                "log10_k0": theta["log10_k0"],
                "alpha": alpha,
                "c_days": 10 ** theta["log10_c"],
                "omega": theta["omega"],
                "tau_days": 10 ** theta["log10_tau"],
                "d_km2": 10 ** theta["log10_d"],
                "gamma": theta["gamma"],
                "rho": theta["rho"],
                "iterations": payload["n_iterations"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "parameter_estimates.csv", index=False)
    return frame


def load_horizon_table() -> pd.DataFrame:
    frame = pd.read_csv(BATCH / "scenario_horizon_metrics.csv")
    # Recompute consistency from the empirical tails. Earlier sweep output used
    # an incorrect two-sided rule that mislabeled (0, 1) as consistent.
    for prefix in ("number", "magnitude"):
        frame[f"{prefix}_consistent"] = (
            (frame[f"{prefix}_q_lower"] >= 0.025)
            & (frame[f"{prefix}_q_upper"] >= 0.025)
        )
    for prefix in ("spatial", "pseudolikelihood"):
        frame[f"{prefix}_consistent"] = frame[f"{prefix}_q_upper"] >= 0.025

    resampled_path = HERE / "resampled_magnitude_window2000.csv"
    if resampled_path.exists():
        resampled = pd.read_csv(resampled_path).set_index("duration_days")
        mask = frame["scenario_name"].eq("window_2000")
        for index in frame.index[mask]:
            duration = int(frame.at[index, "duration_days"])
            row = resampled.loc[duration]
            frame.at[index, "magnitude_status"] = row["status"]
            frame.at[index, "magnitude_q_lower"] = row["q_lower"]
            frame.at[index, "magnitude_q_upper"] = row["q_upper"]
            frame.at[index, "magnitude_consistent"] = (
                row["q_lower"] >= 0.025 and row["q_upper"] >= 0.025
            )
    frame["scenario_name"] = pd.Categorical(
        frame["scenario_name"], categories=SCENARIO_ORDER, ordered=True
    )
    frame = frame.sort_values(["scenario_name", "duration_days"])
    frame.to_csv(TABLE_DIR / "horizon_metrics.csv", index=False)
    return frame


def load_comparison_table(horizons: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(BATCH / "scenario_comparison.csv")
    corrected = horizons.groupby("scenario_name", observed=True).agg(
        n_consistency_fraction=("number_consistent", "mean"),
        m_consistency_fraction=("magnitude_consistent", "mean"),
        s_consistency_fraction=("spatial_consistent", "mean"),
        pl_consistency_fraction=("pseudolikelihood_consistent", "mean"),
    )
    frame = frame.drop(
        columns=[
            "n_consistency_fraction",
            "m_consistency_fraction",
            "s_consistency_fraction",
            "pl_consistency_fraction",
        ]
    ).merge(corrected, on="scenario_name", how="left")
    keep = [
        "scenario_name",
        "mc",
        "timewindow_start",
        "mean_obs_to_sim_ratio",
        "mean_abs_log_count_ratio",
        "mean_count_bias",
        "mean_empty_cell_fraction",
        "n_consistency_fraction",
        "m_consistency_fraction",
        "s_consistency_fraction",
        "pl_consistency_fraction",
    ]
    frame[keep].to_csv(TABLE_DIR / "scenario_summary.csv", index=False)
    return frame


def summarize_magnitude_tails() -> pd.DataFrame:
    rows = []
    thresholds = [7.5, 8.0, 9.0]
    for scenario in SCENARIO_ORDER:
        maxima = (
            pd.read_csv(
                simulation_path(scenario),
                usecols=["catalog_id", "magnitude"],
                dtype={"catalog_id": "int32", "magnitude": "float32"},
            )
            .groupby("catalog_id", sort=False)["magnitude"]
            .max()
            .reindex(range(2000), fill_value=np.nan)
        )
        row = {
            "scenario": scenario,
            "median_catalog_max": maxima.median(),
            "p95_catalog_max": maxima.quantile(0.95),
            "maximum_simulated_magnitude": maxima.max(),
        }
        for threshold in thresholds:
            row[f"fraction_catalogs_ge_{threshold:g}"] = float((maxima >= threshold).mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "magnitude_tail_diagnostics.csv", index=False)
    return frame


def summarize_count_calibration(horizons: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario_name",
        "duration_days",
        "observed_filtered_count",
        "mean_simulated_filtered_count",
        "p05_simulated_filtered_count",
        "p95_simulated_filtered_count",
        "observed_to_sim_mean_ratio",
        "number_q_lower",
        "number_q_upper",
        "number_consistent",
    ]
    result = horizons[columns].copy()
    result.to_csv(TABLE_DIR / "count_calibration.csv", index=False)
    return result


def plot_parameters(parameters: pd.DataFrame) -> None:
    merged = parameters.copy()
    merged["order"] = merged["scenario"].map({v: i for i, v in enumerate(SCENARIO_ORDER)})
    merged = merged.sort_values("order")
    x = np.arange(len(merged))

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.7))
    colors = [COLORS[s] for s in merged["scenario"]]
    axes[0].bar(x, merged["branching_ratio"], color=colors, edgecolor="white",
                linewidth=0.7, zorder=3)
    axes[0].axhline(1.0, color=figstyle.REFERENCE_LINE, linestyle="--", linewidth=1.1,
                    zorder=2)
    axes[0].text(len(x) - 0.4, 1.001, "stationarity boundary", ha="right", va="bottom",
                 fontsize=7.5, color=figstyle.SUBTLE, style="italic")
    axes[0].set_ylabel("Branching ratio $n$")
    figstyle.panel_label(axes[0], "(a) Point estimates")
    axes[0].set_ylim(0.94, 1.045)
    for i, value in enumerate(merged["branching_ratio"]):
        axes[0].text(i, value + 0.0013, f"{value:.3f}", ha="center", va="bottom",
                     fontsize=7.5, color=figstyle.TEXT)

    mmax_values = [7.5, 8.0, 8.5, 9.0]
    for scenario in SCENARIO_ORDER:
        row = merged[merged["scenario"] == scenario].iloc[0]
        values = [row[f"branching_ratio_mmax_{str(value).replace('.', 'p')}"] for value in mmax_values]
        axes[1].plot(
            mmax_values, values, marker="o", color=COLORS[scenario],
            linewidth=1.7, markersize=4.5, markeredgecolor="white",
            markeredgewidth=0.5, label=DISPLAY_NAMES[scenario],
        )
    axes[1].axhline(1.0, color=figstyle.REFERENCE_LINE, linestyle="--", linewidth=1.1)
    axes[1].set_xlabel(r"Assumed maximum magnitude $M_{\max}$")
    axes[1].set_ylabel("Finite-range branching ratio $n$")
    figstyle.panel_label(axes[1], "(b) Upper-magnitude sensitivity")

    labels = [DISPLAY_NAMES[s] for s in merged["scenario"]]
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    figstyle.scenario_legend(fig, ncol=7, y=0.0)
    fig.savefig(FIGURE_DIR / "figure1_branching_sensitivity.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure1_branching_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_count_calibration(horizons: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.7))
    for scenario in SCENARIO_ORDER:
        subset = horizons[horizons["scenario_name"].astype(str) == scenario]
        lw = 2.4 if scenario == "window_2000" else 1.5
        z = 5 if scenario == "window_2000" else 3
        axes[0].plot(
            subset["duration_days"] / 365.25,
            subset["observed_to_sim_mean_ratio"],
            marker="o", color=COLORS[scenario], label=DISPLAY_NAMES[scenario],
            linewidth=lw, markersize=4.2, markeredgecolor="white",
            markeredgewidth=0.5, zorder=z,
        )
    axes[0].axhline(1.0, color=figstyle.REFERENCE_LINE, linestyle="--", linewidth=1.1,
                    zorder=2)
    axes[0].set_xlabel("Forecast horizon (yr)")
    axes[0].set_ylabel("Observed / simulated mean count")
    figstyle.panel_label(axes[0], "(a) Count-ratio evolution")
    axes[0].annotate("overprediction", xy=(4.6, 0.9), fontsize=7.5, color=figstyle.SUBTLE,
                     style="italic", ha="right")
    axes[0].annotate("underprediction", xy=(1.1, 1.62), fontsize=7.5, color=figstyle.SUBTLE,
                     style="italic")

    best = horizons[horizons["scenario_name"].astype(str) == "window_2000"]
    x = best["duration_days"] / 365.25
    axes[1].fill_between(
        x, best["p05_simulated_filtered_count"], best["p95_simulated_filtered_count"],
        color=COLORS["window_2000"], alpha=0.18, linewidth=0,
        label="5th–95th percentile",
    )
    axes[1].plot(x, best["mean_simulated_filtered_count"], "o-",
                 color=COLORS["window_2000"], linewidth=2.0, markersize=4.5,
                 markeredgecolor="white", markeredgewidth=0.5, label="Simulated mean")
    axes[1].plot(x, best["observed_filtered_count"], "s-", color=figstyle.OBSERVED,
                 linewidth=1.8, markersize=4.5, label="Observed")
    axes[1].set_xlabel("Forecast horizon (yr)")
    axes[1].set_ylabel(r"Number of $M\geq4.1$ earthquakes")
    figstyle.panel_label(axes[1], "(b) 2000-window forecast")
    axes[1].legend(loc="upper left")

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    figstyle.scenario_legend(fig, ncol=7, y=0.0)
    fig.savefig(FIGURE_DIR / "figure2_count_calibration.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure2_count_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_residuals() -> None:
    simulations = pd.read_csv(
        simulation_path("window_2000"),
        usecols=["longitude", "latitude"],
        dtype={"longitude": "float32", "latitude": "float32"},
    )
    observed = pd.read_csv(
        observed_path("window_2000"), usecols=["longitude", "latitude"]
    )
    lon_edges = np.arange(165.0, 180.0001, 0.1)
    lat_edges = np.arange(-48.0, -33.9999, 0.1)
    sim_counts, _, _ = np.histogram2d(
        simulations["longitude"], simulations["latitude"], bins=[lon_edges, lat_edges]
    )
    obs_counts, _, _ = np.histogram2d(
        observed["longitude"], observed["latitude"], bins=[lon_edges, lat_edges]
    )
    expected = sim_counts / 2000.0
    residual = obs_counts - expected

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker

    proj = ccrs.PlateCarree()
    extent = [165.0, 180.0, -48.0, -34.0]
    fig, axes = plt.subplots(
        1, 3, figsize=(11.5, 4.6), subplot_kw={"projection": proj}
    )

    def basemap(ax):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.OCEAN, facecolor="#EAF2F8", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#F1EFE9", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color="#52606D", zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#B8C4D0",
                          alpha=0.7, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(166, 181, 3))
        gl.ylocator = mticker.FixedLocator(np.arange(-48, -33, 3))
        gl.xlabel_style = {"size": 8.5, "color": figstyle.TEXT}
        gl.ylabel_style = {"size": 8.5, "color": figstyle.TEXT}

    def cbar(im, ax, label):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025, shrink=0.92)
        cb.set_label(label, fontsize=9.5, color=figstyle.TEXT)
        cb.ax.tick_params(labelsize=8.5, colors=figstyle.TEXT)
        cb.outline.set_edgecolor("#9FB3C8")
        cb.outline.set_linewidth(0.8)

    expected_positive = expected[expected > 0]
    vmax = max(float(np.percentile(expected_positive, 99.8)), 0.1)
    basemap(axes[0])
    im0 = axes[0].pcolormesh(
        lon_edges, lat_edges, expected.T, transform=proj, cmap="viridis",
        norm=LogNorm(vmin=max(float(expected_positive.min()), 1e-4), vmax=vmax),
        zorder=2, rasterized=True,
    )
    basemap(axes[1])
    im1 = axes[1].pcolormesh(
        lon_edges, lat_edges, obs_counts.T, transform=proj, cmap="magma",
        norm=LogNorm(vmin=1, vmax=max(float(obs_counts.max()), 1)), zorder=2,
        rasterized=True,
    )
    limit = float(np.max(np.abs(residual)))
    basemap(axes[2])
    im2 = axes[2].pcolormesh(
        lon_edges, lat_edges, residual.T, transform=proj, cmap="RdBu_r",
        norm=SymLogNorm(linthresh=0.25, vmin=-limit, vmax=limit), zorder=2,
        rasterized=True,
    )
    for ax, title in zip(axes, ["(a) Forecast mean", "(b) Observed",
                                "(c) Observed $-$ forecast"]):
        figstyle.panel_label(ax, title)
    axes[2].annotate(
        "March 2021\nEast Cape sequence",
        xy=(179.7, -37.45), xytext=(169.8, -34.9), transform=proj,
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": figstyle.NAVY},
        fontsize=8.0, color=figstyle.NAVY, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#9FB3C8",
                  alpha=0.9),
    )
    cbar(im0, axes[0], "Expected count per cell")
    cbar(im1, axes[1], "Observed count per cell")
    cbar(im2, axes[2], "Count residual")
    fig.subplots_adjust(wspace=0.12, left=0.04, right=0.97, top=0.93, bottom=0.06)
    fig.savefig(FIGURE_DIR / "figure3_spatial_residuals.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure3_spatial_residuals.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_magnitude_tails(tails: pd.DataFrame) -> None:
    tails = tails.set_index("scenario").loc[SCENARIO_ORDER].reset_index()
    x = np.arange(len(tails))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.7))
    axes[0].bar(
        x, 100 * tails["fraction_catalogs_ge_8"],
        color=[COLORS[s] for s in tails["scenario"]],
        edgecolor="white", linewidth=0.7, zorder=3,
    )
    axes[0].set_ylabel(r"Five-year catalogs with $M\geq8$ (\%)".replace("\\%", "%"))
    figstyle.panel_label(axes[0], "(a) Upper-tail probability")
    axes[0].set_xticks(
        x, [DISPLAY_NAMES[s] for s in tails["scenario"]], rotation=35, ha="right"
    )

    axes[1].bar(x - 0.19, tails["p95_catalog_max"], width=0.38, color="#6BAED6",
                edgecolor="white", linewidth=0.6, label="95th percentile maximum",
                zorder=3)
    axes[1].bar(x + 0.19, tails["maximum_simulated_magnitude"], width=0.38,
                color="#D1495B", edgecolor="white", linewidth=0.6,
                label="Ensemble maximum", zorder=3)
    axes[1].axhline(7.2, color=figstyle.OBSERVED, linestyle="--", linewidth=1.2,
                    label="Observed maximum (M7.2)", zorder=4)
    axes[1].set_ylabel("Magnitude")
    figstyle.panel_label(axes[1], "(b) Unbounded magnitude diagnostic")
    axes[1].set_xticks(
        x, [DISPLAY_NAMES[s] for s in tails["scenario"]], rotation=35, ha="right"
    )
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure4_magnitude_tails.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure4_magnitude_tails.pdf", bbox_inches="tight")
    plt.close(fig)


def write_key_results(
    parameters: pd.DataFrame, horizons: pd.DataFrame, tails: pd.DataFrame
) -> None:
    best = horizons[horizons["scenario_name"].astype(str) == "window_2000"]
    best_params = parameters[parameters["scenario"] == "window_2000"].iloc[0]
    best_tails = tails[tails["scenario"] == "window_2000"].iloc[0]
    payload = {
        "subcritical_scenarios": int((parameters["branching_ratio"] < 1).sum()),
        "window_2000_branching_ratio": float(best_params["branching_ratio"]),
        "window_2000_training_events": int(best_params["n_training"]),
        "window_2000_count_ratios": {
            str(int(row.duration_days)): float(row.observed_to_sim_mean_ratio)
            for row in best.itertuples()
        },
        "window_2000_n_test_passes": int(best["number_consistent"].astype(bool).sum()),
        "window_2000_m_test_passes": int(best["magnitude_consistent"].astype(bool).sum()),
        "window_2000_s_test_passes": int(best["spatial_consistent"].astype(bool).sum()),
        "window_2000_pl_test_passes": int(
            best["pseudolikelihood_consistent"].astype(bool).sum()
        ),
        "window_2000_empty_cell_fraction_mean": float(
            best["expected_count_in_empty_cells_fraction"].mean()
        ),
        "window_2000_fraction_catalogs_ge_8": float(
            best_tails["fraction_catalogs_ge_8"]
        ),
        "window_2000_ensemble_maximum": float(
            best_tails["maximum_simulated_magnitude"]
        ),
    }
    with (TABLE_DIR / "key_results.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    setup()
    parameters = load_parameter_table()
    horizons = load_horizon_table()
    comparison = load_comparison_table(horizons)
    tails = summarize_magnitude_tails()
    summarize_count_calibration(horizons)
    plot_parameters(parameters)
    plot_count_calibration(horizons)
    plot_spatial_residuals()
    plot_magnitude_tails(tails)
    write_key_results(parameters, horizons, tails)
    print(f"Wrote tables to {TABLE_DIR}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
