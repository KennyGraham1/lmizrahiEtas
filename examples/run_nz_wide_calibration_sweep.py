"""
Run a NZ-wide ETAS calibration sweep and compare scenarios with pyCSEP outputs.

This script orchestrates multiple runs of ``run_nz_wide_forecast.py`` with
different ETAS setup choices and then aggregates the resulting pyCSEP summaries
into a calibration scorecard.

Examples
--------
python run_nz_wide_calibration_sweep.py --forecast-start "2018-01-01 00:00:00"
python run_nz_wide_calibration_sweep.py --forecast-start "2018-01-01 00:00:00" \
    --scenario-set quick --n-simulations 150 --skip-plots
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT_DIR, ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FORECAST_SCRIPT = os.path.join(BASE_DIR, "run_nz_wide_forecast.py")
DEFAULT_FORECAST_START = "2018-01-01 00:00:00"
DEFAULT_DURATIONS = "30,90,365"
DEFAULT_BATCH_NAME = "nz_wide_calibration"
DEFAULT_N_SIMULATIONS = 250


@dataclass(frozen=True)
class Scenario:
    name: str
    mc: float
    timewindow_start: str
    auxiliary_start: str
    theta_log10_mu_delta: float = 0.0
    theta_log10_k0_delta: float = 0.0
    m_max: float | None = None
    notes: str = ""


def build_default_scenarios() -> dict[str, list[Scenario]]:
    baseline = Scenario(
        name="baseline",
        mc=4.1,
        timewindow_start="1960-01-01 00:00:00",
        auxiliary_start="1950-01-01 00:00:00",
        notes="Current NZ-wide baseline configuration.",
    )
    quick = [
        baseline,
        Scenario(
            name="mc_4p3",
            mc=4.3,
            timewindow_start=baseline.timewindow_start,
            auxiliary_start=baseline.auxiliary_start,
            notes="Higher completeness threshold.",
        ),
        Scenario(
            name="window_1980",
            mc=baseline.mc,
            timewindow_start="1980-01-01 00:00:00",
            auxiliary_start="1970-01-01 00:00:00",
            notes="Shorter modern training window.",
        ),
        Scenario(
            name="low_mu_k0",
            mc=baseline.mc,
            timewindow_start=baseline.timewindow_start,
            auxiliary_start=baseline.auxiliary_start,
            theta_log10_mu_delta=-0.5,
            theta_log10_k0_delta=-0.25,
            notes="Conservative initial background/productivity guess.",
        ),
    ]
    default = quick + [
        Scenario(
            name="mc_4p5",
            mc=4.5,
            timewindow_start=baseline.timewindow_start,
            auxiliary_start=baseline.auxiliary_start,
            notes="More conservative completeness threshold.",
        ),
        Scenario(
            name="window_2000",
            mc=baseline.mc,
            timewindow_start="2000-01-01 00:00:00",
            auxiliary_start="1990-01-01 00:00:00",
            notes="Very recent training window.",
        ),
        Scenario(
            name="high_mu_k0",
            mc=baseline.mc,
            timewindow_start=baseline.timewindow_start,
            auxiliary_start=baseline.auxiliary_start,
            theta_log10_mu_delta=0.5,
            theta_log10_k0_delta=0.25,
            notes="Aggressive initial background/productivity guess.",
        ),
    ]
    return {"quick": quick, "default": default}


SCENARIO_LIBRARY = build_default_scenarios()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep NZ-wide ETAS calibration settings and compare runs with "
            "pyCSEP and residual-based diagnostics."
        )
    )
    parser.add_argument(
        "--forecast-start",
        default=DEFAULT_FORECAST_START,
        help=f"Forecast origin. Default: {DEFAULT_FORECAST_START}",
    )
    parser.add_argument(
        "--durations",
        default=DEFAULT_DURATIONS,
        help=f"Comma-separated forecast durations in days. Default: {DEFAULT_DURATIONS}",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=DEFAULT_N_SIMULATIONS,
        help=f"Number of simulated catalogs per scenario. Default: {DEFAULT_N_SIMULATIONS}",
    )
    parser.add_argument(
        "--batch-name",
        default=DEFAULT_BATCH_NAME,
        help=f"Output prefix for the sweep artifacts. Default: {DEFAULT_BATCH_NAME}",
    )
    parser.add_argument(
        "--scenario-set",
        choices=sorted(SCENARIO_LIBRARY.keys()),
        default="default",
        help="Built-in scenario set to run. Default: default",
    )
    parser.add_argument(
        "--scenario-file",
        default=None,
        help="Optional JSON file with an explicit list of scenarios. Overrides --scenario-set.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Forward --skip-plots to the forecast runner.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun every scenario even if completed outputs already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the sweep if one scenario fails.",
    )
    parser.add_argument(
        "--pycsep-region-source",
        choices=["forecast_domain", "nz_csep_collection"],
        default="forecast_domain",
        help="Region source forwarded to the pyCSEP analysis step.",
    )
    parser.add_argument(
        "--pycsep-grid-spacing",
        type=float,
        default=0.1,
        help="Grid spacing forwarded to the pyCSEP analysis step. Default: 0.1",
    )
    parser.add_argument(
        "--pycsep-mag-bin",
        type=float,
        default=0.1,
        help="Magnitude bin width forwarded to the pyCSEP analysis step. Default: 0.1",
    )
    parser.add_argument(
        "--pycsep-max-mag",
        type=float,
        default=None,
        help="Maximum magnitude bin edge forwarded to the pyCSEP analysis step.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum():
            cleaned.append(char.lower())
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "run"


def parse_datetime(value: str) -> dt.datetime:
    return pd.Timestamp(value).to_pydatetime()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_scenarios(args: argparse.Namespace) -> list[Scenario]:
    if args.scenario_file is None:
        return SCENARIO_LIBRARY[args.scenario_set]

    with open(args.scenario_file, "r") as f:
        payload = json.load(f)

    if not isinstance(payload, list) or not payload:
        raise ValueError("Scenario file must contain a non-empty JSON list.")

    scenarios = []
    for idx, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Scenario at index {idx} is not a JSON object.")
        if "name" not in raw:
            raise ValueError(f"Scenario at index {idx} is missing 'name'.")
        scenarios.append(
            Scenario(
                name=str(raw["name"]),
                mc=float(raw.get("mc", 4.1)),
                timewindow_start=str(raw.get("timewindow_start", "1960-01-01 00:00:00")),
                auxiliary_start=str(raw.get("auxiliary_start", "1950-01-01 00:00:00")),
                theta_log10_mu_delta=float(raw.get("theta_log10_mu_delta", 0.0)),
                theta_log10_k0_delta=float(raw.get("theta_log10_k0_delta", 0.0)),
                m_max=None if raw.get("m_max") is None else float(raw["m_max"]),
                notes=str(raw.get("notes", "")),
            )
        )
    return scenarios


def scenario_run_label(experiment_name: str, forecast_start: dt.datetime) -> str:
    return f"{slugify(experiment_name)}_{forecast_start:%Y%m%d_%H%M%S}"


def scenario_metadata_path(experiment_name: str, forecast_start: dt.datetime) -> str:
    run_label = scenario_run_label(experiment_name, forecast_start)
    return os.path.join(BASE_DIR, "output_nz_wide", run_label, "experiment_config.json")


def scenario_complete(metadata_path: str) -> bool:
    if not os.path.exists(metadata_path):
        return False
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    pycsep = metadata.get("pycsep_analysis", {})
    summary_csv = pycsep.get("summary_csv")
    return (
        pycsep.get("status") == "completed"
        and isinstance(summary_csv, str)
        and os.path.exists(summary_csv)
    )


def metadata_matches_request(
    metadata_path: str,
    scenario: Scenario,
    args: argparse.Namespace,
) -> bool:
    if not os.path.exists(metadata_path):
        return False
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    requested_durations = sorted(
        int(float(piece.strip()))
        for piece in args.durations.split(",")
        if piece.strip()
    )
    metadata_durations = sorted(int(float(value)) for value in metadata.get("forecast_durations_days", []))
    if metadata.get("forecast_start") != args.forecast_start:
        return False
    if metadata_durations != requested_durations:
        return False
    if int(metadata.get("n_simulations", -1)) != int(args.n_simulations):
        return False
    if float(metadata.get("mc", np.nan)) != float(scenario.mc):
        return False
    if metadata.get("timewindow_start") != scenario.timewindow_start:
        return False
    if metadata.get("auxiliary_start") != scenario.auxiliary_start:
        return False
    if float(metadata.get("theta_log10_mu_delta", np.nan)) != float(scenario.theta_log10_mu_delta):
        return False
    if float(metadata.get("theta_log10_k0_delta", np.nan)) != float(scenario.theta_log10_k0_delta):
        return False

    pycsep = metadata.get("pycsep_analysis", {})
    if pycsep.get("status") != "completed":
        return False
    if pycsep.get("region_source") != args.pycsep_region_source:
        return False
    if float(pycsep.get("grid_spacing", np.nan)) != float(args.pycsep_grid_spacing):
        return False
    if float(pycsep.get("mag_bin", np.nan)) != float(args.pycsep_mag_bin):
        return False
    requested_max_mag = args.pycsep_max_mag
    stored_max_mag = pycsep.get("max_mag")
    if requested_max_mag is None:
        if stored_max_mag is None:
            return False
    elif float(stored_max_mag) != float(requested_max_mag):
        return False
    return True


def run_scenario(
    scenario: Scenario,
    args: argparse.Namespace,
    forecast_start: dt.datetime,
    logger: logging.Logger,
) -> tuple[str, str]:
    experiment_name = f"{args.batch_name}_{scenario.name}"
    metadata_path = scenario_metadata_path(experiment_name, forecast_start)
    if (
        scenario_complete(metadata_path)
        and metadata_matches_request(metadata_path, scenario, args)
        and not args.force_rerun
    ):
        logger.info("Reusing completed scenario '%s' from %s", scenario.name, metadata_path)
        return experiment_name, metadata_path

    needs_forced_reinvert = args.force_rerun or os.path.exists(metadata_path)
    command = [
        sys.executable,
        FORECAST_SCRIPT,
        "--forecast-start",
        args.forecast_start,
        "--durations",
        args.durations,
        "--n-simulations",
        str(args.n_simulations),
        "--experiment-name",
        experiment_name,
        "--timewindow-start",
        scenario.timewindow_start,
        "--auxiliary-start",
        scenario.auxiliary_start,
        "--mc",
        str(scenario.mc),
        "--theta-log10-mu-delta",
        str(scenario.theta_log10_mu_delta),
        "--theta-log10-k0-delta",
        str(scenario.theta_log10_k0_delta),
        "--pycsep-region-source",
        args.pycsep_region_source,
        "--pycsep-grid-spacing",
        str(args.pycsep_grid_spacing),
        "--pycsep-mag-bin",
        str(args.pycsep_mag_bin),
    ]
    if scenario.m_max is not None:
        command.extend(["--m-max", str(scenario.m_max)])
    if args.pycsep_max_mag is not None:
        command.extend(["--pycsep-max-mag", str(args.pycsep_max_mag)])
    if args.skip_plots:
        command.append("--skip-plots")
    if needs_forced_reinvert:
        command.append("--force-reinvert")

    logger.info("Running scenario '%s'", scenario.name)
    logger.info("Command: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=BASE_DIR)
    return experiment_name, metadata_path


def parse_bool_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    parsed = series.map(
        lambda value: mapping.get(str(value).strip().lower(), value)
    )
    numeric = pd.to_numeric(parsed, errors="coerce")
    return float(numeric.mean()) if len(numeric.dropna()) else float("nan")


def load_parameter_summary(parameter_path: str) -> dict[str, Any]:
    with open(parameter_path, "r") as f:
        payload = json.load(f)
    final_parameters = payload.get("final_parameters") or {}
    initial_values = payload.get("initial_values") or {}
    return {
        "beta": payload.get("beta", np.nan),
        "n_hat": payload.get("n_hat", np.nan),
        "final_log10_mu": final_parameters.get("log10_mu", np.nan),
        "final_log10_k0": final_parameters.get("log10_k0", np.nan),
        "final_a": final_parameters.get("a", np.nan),
        "final_gamma": final_parameters.get("gamma", np.nan),
        "final_rho": final_parameters.get("rho", np.nan),
        "initial_log10_mu": initial_values.get("log10_mu", np.nan),
        "initial_log10_k0": initial_values.get("log10_k0", np.nan),
        "initial_a": initial_values.get("a", np.nan),
    }


def safe_log_ratio(value: float) -> float:
    clipped = max(float(value), 1e-12)
    return abs(float(np.log(clipped)))


def ensure_completed_pycsep_analysis(
    metadata_path: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, Any]:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    pycsep = metadata.get("pycsep_analysis", {})
    summary_path = pycsep.get("summary_csv")
    if (
        pycsep.get("status") == "completed"
        and isinstance(summary_path, str)
        and os.path.exists(summary_path)
    ):
        return metadata

    logger.info(
        "Attempting standalone pyCSEP recovery for %s (status=%s)",
        metadata_path,
        pycsep.get("status", "missing"),
    )
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    try:
        from run_nz_wide_pycsep_analysis import run_analysis_from_metadata
    except Exception as exc:
        stored_error = pycsep.get("error")
        raise RuntimeError(
            "pyCSEP analysis is unavailable for this scenario. "
            f"stored status={pycsep.get('status', 'missing')}, "
            f"stored error={stored_error!r}, import error={exc!r}"
        ) from exc

    run_analysis_from_metadata(
        metadata_path=metadata_path,
        region_source=args.pycsep_region_source,
        grid_spacing=args.pycsep_grid_spacing,
        mag_bin=args.pycsep_mag_bin,
        max_mag=args.pycsep_max_mag,
        output_dir=None,
    )

    with open(metadata_path, "r") as f:
        refreshed_metadata = json.load(f)
    refreshed_pycsep = refreshed_metadata.get("pycsep_analysis", {})
    refreshed_summary = refreshed_pycsep.get("summary_csv")
    if (
        refreshed_pycsep.get("status") != "completed"
        or not isinstance(refreshed_summary, str)
        or not os.path.exists(refreshed_summary)
    ):
        raise RuntimeError(
            "Standalone pyCSEP recovery did not produce a completed summary. "
            f"status={refreshed_pycsep.get('status', 'missing')}, "
            f"error={refreshed_pycsep.get('error')!r}, "
            f"summary={refreshed_summary!r}"
        )
    return refreshed_metadata


def compute_calibration_score(record: dict[str, Any]) -> float:
    count_penalty = float(record["mean_abs_log_count_ratio"])
    consistency_penalty = (
        1.5 * (1.0 - float(record["n_consistency_fraction"]))
        + 0.25 * (1.0 - float(record["m_consistency_fraction"]))
        + 0.5 * (1.0 - float(record["s_consistency_fraction"]))
        + 0.5 * (1.0 - float(record["pl_consistency_fraction"]))
    )
    spatial_penalty = (
        float(record["mean_empty_cell_fraction"])
        + float(record["mean_zero_rate_observed_fraction"])
        + 10.0 * float(record["mean_spatial_abs_residual"])
    )
    return count_penalty + consistency_penalty + spatial_penalty


def collect_scenario_results(
    scenario: Scenario,
    experiment_name: str,
    metadata_path: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metadata = ensure_completed_pycsep_analysis(metadata_path, args, logger)
    pycsep = metadata.get("pycsep_analysis", {})
    summary_path = pycsep.get("summary_csv")
    if pycsep.get("status") != "completed" or not summary_path or not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Missing completed pyCSEP summary for {scenario.name}: "
            f"status={pycsep.get('status')}, summary={summary_path}, error={pycsep.get('error')!r}"
        )

    summary_df = pd.read_csv(summary_path)
    summary_df["scenario_name"] = scenario.name
    summary_df["experiment_name"] = experiment_name
    summary_df["notes"] = scenario.notes

    parameter_summary = load_parameter_summary(metadata["parameter_file"])
    record = {
        "scenario_name": scenario.name,
        "experiment_name": experiment_name,
        "run_label": metadata["run_label"],
        "metadata_path": metadata_path,
        "summary_csv": summary_path,
        "notes": scenario.notes,
        "mc": scenario.mc,
        "timewindow_start": scenario.timewindow_start,
        "auxiliary_start": scenario.auxiliary_start,
        "theta_log10_mu_delta": scenario.theta_log10_mu_delta,
        "theta_log10_k0_delta": scenario.theta_log10_k0_delta,
        "mean_obs_to_sim_ratio": float(summary_df["observed_to_sim_mean_ratio"].mean()),
        "mean_abs_log_count_ratio": float(
            summary_df["observed_to_sim_mean_ratio"].map(safe_log_ratio).mean()
        ),
        "mean_count_bias": float(summary_df["sim_minus_obs_mean_bias"].mean()),
        "mean_empty_cell_fraction": float(
            summary_df["expected_count_in_empty_cells_fraction"].mean()
        ),
        "mean_zero_rate_observed_fraction": float(
            summary_df["observed_count_in_zero_rate_cells_fraction"].mean()
        ),
        "mean_spatial_abs_residual": float(summary_df["spatial_mean_abs_residual"].mean()),
        "n_consistency_fraction": parse_bool_fraction(summary_df["number_consistent"]),
        "m_consistency_fraction": parse_bool_fraction(summary_df["magnitude_consistent"]),
        "s_consistency_fraction": parse_bool_fraction(summary_df["spatial_consistent"]),
        "pl_consistency_fraction": parse_bool_fraction(
            summary_df["pseudolikelihood_consistent"]
        ),
        "mean_observed_count": float(summary_df["observed_filtered_count"].mean()),
        "mean_simulated_count": float(summary_df["mean_simulated_filtered_count"].mean()),
        "max_positive_residual_value": float(summary_df["max_positive_residual_value"].max()),
        "max_negative_residual_value": float(summary_df["max_negative_residual_value"].min()),
        **parameter_summary,
    }
    record["calibration_score"] = compute_calibration_score(record)
    return record, summary_df


def write_markdown_report(
    comparison_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    output_path: str,
    args: argparse.Namespace,
) -> None:
    ordered = comparison_df.sort_values("calibration_score").reset_index(drop=True)
    best = ordered.iloc[0]
    lines = [
        "# NZ-Wide Calibration Sweep",
        "",
        f"- Forecast start: `{args.forecast_start}`",
        f"- Forecast horizons: `{args.durations}` days",
        f"- Simulations per scenario: `{args.n_simulations}`",
        f"- Scenario count: `{len(ordered)}`",
        "",
        "## Ranking Rule",
        "",
        (
            "- `calibration_score = mean_abs_log_count_ratio + consistency_penalty + "
            "spatial_penalty`, where consistency penalizes failed `N/M/S/PL` windows "
            "and spatial penalizes empty-cell rate allocation, unsupported observed "
            "events, and mean absolute spatial residuals."
        ),
        "",
        "## Best Scenario",
        "",
        f"- Best scenario: `{best['scenario_name']}`",
        f"- Score: `{best['calibration_score']:.3f}`",
        f"- Mean observed/simulated ratio: `{best['mean_obs_to_sim_ratio']:.3f}`",
        f"- Mean empty-cell forecast share: `{best['mean_empty_cell_fraction']:.3f}`",
        f"- Mean unsupported observed-event share: `{best['mean_zero_rate_observed_fraction']:.3f}`",
        "",
        "## Scenario Comparison",
        "",
        "| Scenario | Score | Obs/Sim ratio | Empty-cell share | Unsupported obs share | N frac | S frac | PL frac | final log10_mu | final log10_k0 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered.itertuples(index=False):
        lines.append(
            "| "
            f"{row.scenario_name} | "
            f"{row.calibration_score:.3f} | "
            f"{row.mean_obs_to_sim_ratio:.3f} | "
            f"{row.mean_empty_cell_fraction:.3f} | "
            f"{row.mean_zero_rate_observed_fraction:.3f} | "
            f"{row.n_consistency_fraction:.3f} | "
            f"{row.s_consistency_fraction:.3f} | "
            f"{row.pl_consistency_fraction:.3f} | "
            f"{row.final_log10_mu:.3f} | "
            f"{row.final_log10_k0:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Horizon Ratios",
            "",
            "| Scenario | Horizon (days) | Obs/Sim ratio | Empty-cell share | Unsupported obs share |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    horizon_ordered = horizon_df.merge(
        ordered[["scenario_name", "calibration_score"]],
        on="scenario_name",
        how="left",
    ).sort_values(["calibration_score", "duration_days"])
    for row in horizon_ordered.itertuples(index=False):
        lines.append(
            "| "
            f"{row.scenario_name} | "
            f"{row.duration_days:.0f} | "
            f"{row.observed_to_sim_mean_ratio:.3f} | "
            f"{row.expected_count_in_empty_cells_fraction:.3f} | "
            f"{row.observed_count_in_zero_rate_cells_fraction:.3f} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_scorecard(
    comparison_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    output_path: str,
) -> None:
    ordered = comparison_df.sort_values("calibration_score").reset_index(drop=True)
    scenario_names = ordered["scenario_name"].tolist()
    y = np.arange(len(ordered))

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    ax_score, ax_consistency, ax_ratio, ax_spatial, ax_heatmap, ax_params = axes.flatten()

    ax_score.barh(y, ordered["calibration_score"], color="#2C6EAA", alpha=0.9)
    ax_score.set_yticks(y, labels=scenario_names)
    ax_score.invert_yaxis()
    ax_score.set_title("Calibration Score")
    ax_score.set_xlabel("Lower Is Better")
    ax_score.grid(True, axis="x", alpha=0.25)

    width = 0.18
    x = np.arange(len(ordered))
    ax_consistency.bar(x - 1.5 * width, ordered["n_consistency_fraction"], width=width, label="N")
    ax_consistency.bar(x - 0.5 * width, ordered["m_consistency_fraction"], width=width, label="M")
    ax_consistency.bar(x + 0.5 * width, ordered["s_consistency_fraction"], width=width, label="S")
    ax_consistency.bar(x + 1.5 * width, ordered["pl_consistency_fraction"], width=width, label="PL")
    ax_consistency.set_xticks(x, labels=scenario_names, rotation=30, ha="right")
    ax_consistency.set_ylim(0, 1.05)
    ax_consistency.set_title("Consistency Fractions")
    ax_consistency.grid(True, axis="y", alpha=0.25)
    ax_consistency.legend(loc="upper right", frameon=True)

    ax_ratio.plot(
        ordered["mean_obs_to_sim_ratio"],
        y,
        "o",
        color="#F28E2B",
        markersize=8,
    )
    ax_ratio.axvline(1.0, color="#6B7280", linestyle="--", linewidth=1)
    ax_ratio.set_yticks(y, labels=scenario_names)
    ax_ratio.invert_yaxis()
    ax_ratio.set_title("Mean Observed / Simulated Ratio")
    ax_ratio.set_xlabel("Closer to 1 Is Better")
    ax_ratio.grid(True, axis="x", alpha=0.25)

    ax_spatial.bar(
        x - width / 2,
        100 * ordered["mean_empty_cell_fraction"],
        width=width,
        color="#F28E2B",
        label="Empty-cell forecast share",
    )
    ax_spatial.bar(
        x + width / 2,
        100 * ordered["mean_zero_rate_observed_fraction"],
        width=width,
        color="#D62728",
        label="Unsupported observed share",
    )
    ax_spatial.set_xticks(x, labels=scenario_names, rotation=30, ha="right")
    ax_spatial.set_title("Spatial Allocation Penalties")
    ax_spatial.set_ylabel("Percent")
    ax_spatial.grid(True, axis="y", alpha=0.25)
    ax_spatial.legend(loc="upper right", frameon=True)

    ratio_pivot = horizon_df.pivot_table(
        index="scenario_name",
        columns="duration_days",
        values="observed_to_sim_mean_ratio",
    )
    ratio_pivot = ratio_pivot.reindex(scenario_names)
    durations = ratio_pivot.columns.tolist()
    image = ax_heatmap.imshow(ratio_pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax_heatmap.set_title("Obs/Sim Ratio by Horizon")
    ax_heatmap.set_yticks(np.arange(len(scenario_names)), labels=scenario_names)
    ax_heatmap.set_xticks(np.arange(len(durations)), labels=[f"{int(d)}d" for d in durations])
    for i in range(ratio_pivot.shape[0]):
        for j in range(ratio_pivot.shape[1]):
            value = ratio_pivot.iat[i, j]
            if pd.notna(value):
                ax_heatmap.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.04)

    mask = np.isfinite(ordered["final_log10_mu"]) & np.isfinite(ordered["final_log10_k0"])
    if mask.any():
        ax_params.scatter(
            ordered.loc[mask, "final_log10_mu"],
            ordered.loc[mask, "final_log10_k0"],
            c=ordered.loc[mask, "calibration_score"],
            cmap="viridis_r",
            s=80,
            edgecolors="black",
            linewidths=0.5,
        )
        for row in ordered.loc[mask].itertuples(index=False):
            ax_params.annotate(
                row.scenario_name,
                (row.final_log10_mu, row.final_log10_k0),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
            )
        ax_params.set_title("Final log10_mu vs log10_k0")
        ax_params.set_xlabel("Final log10_mu")
        ax_params.set_ylabel("Final log10_k0")
        ax_params.grid(True, alpha=0.25)
    else:
        ax_params.axis("off")

    fig.suptitle("NZ-Wide Calibration Sweep Scorecard", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logger = logging.getLogger(__name__)
    forecast_start = parse_datetime(args.forecast_start)
    scenarios = load_scenarios(args)

    batch_label = scenario_run_label(args.batch_name, forecast_start)
    batch_output_dir = ensure_dir(os.path.join(BASE_DIR, "output_nz_wide_calibration", batch_label))

    scenario_records = []
    horizon_frames = []
    manifest = {
        "batch_name": args.batch_name,
        "batch_label": batch_label,
        "forecast_start": args.forecast_start,
        "durations": args.durations,
        "n_simulations": args.n_simulations,
        "scenarios": [],
        "failures": [],
    }

    for scenario in scenarios:
        try:
            experiment_name, metadata_path = run_scenario(scenario, args, forecast_start, logger)
            record, horizon_df = collect_scenario_results(
                scenario,
                experiment_name,
                metadata_path,
                args,
                logger,
            )
            scenario_records.append(record)
            horizon_frames.append(horizon_df)
            manifest["scenarios"].append(
                {
                    **asdict(scenario),
                    "experiment_name": experiment_name,
                    "metadata_path": metadata_path,
                }
            )
        except Exception as exc:  # pragma: no cover - best effort continuation
            logger.error("Scenario '%s' failed: %s", scenario.name, exc)
            manifest["failures"].append({"scenario": scenario.name, "error": str(exc)})
            if not args.continue_on_error:
                raise

    if not scenario_records:
        raise RuntimeError("No scenarios completed successfully.")

    comparison_df = pd.DataFrame.from_records(scenario_records).sort_values(
        "calibration_score"
    )
    comparison_df["rank"] = np.arange(1, len(comparison_df) + 1)
    horizon_df = pd.concat(horizon_frames, ignore_index=True).copy()

    comparison_csv = os.path.join(batch_output_dir, "scenario_comparison.csv")
    horizon_csv = os.path.join(batch_output_dir, "scenario_horizon_metrics.csv")
    manifest_path = os.path.join(batch_output_dir, "scenario_manifest.json")
    report_path = os.path.join(batch_output_dir, "scenario_report.md")
    figure_path = os.path.join(batch_output_dir, "scenario_scorecard.png")

    comparison_df.to_csv(comparison_csv, index=False)
    horizon_df.to_csv(horizon_csv, index=False)
    plot_scorecard(comparison_df, horizon_df, figure_path)
    write_markdown_report(comparison_df, horizon_df, report_path, args)

    manifest.update(
        {
            "comparison_csv": comparison_csv,
            "horizon_csv": horizon_csv,
            "report_md": report_path,
            "figure_png": figure_path,
        }
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Calibration sweep complete.")
    logger.info("Comparison CSV: %s", comparison_csv)
    logger.info("Horizon CSV: %s", horizon_csv)
    logger.info("Report: %s", report_path)
    logger.info("Figure: %s", figure_path)


if __name__ == "__main__":
    main()
