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
DEFAULT_FORECAST_START = "2021-01-01 00:00:00"
DEFAULT_DURATIONS = "365,730,1095,1461,1826"
DEFAULT_BATCH_NAME = "nz_wide_calibration"
DEFAULT_N_SIMULATIONS = 2000
DEFAULT_BACKGROUND_RATE_FILE = os.path.join(
    ROOT_DIR,
    "input_data",
    "hftlongtermmodel005.txt",
)
DEFAULT_BACKGROUND_RATE_MAG = 5.0


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
    parser.add_argument(
        "--background-rate-file",
        default=DEFAULT_BACKGROUND_RATE_FILE,
        help=(
            "Long-term background-rate grid forwarded to the forecast runner. "
            "Use an empty string to disable. "
            f"Default: {DEFAULT_BACKGROUND_RATE_FILE}"
        ),
    )
    parser.add_argument(
        "--background-rate-mag",
        type=float,
        default=DEFAULT_BACKGROUND_RATE_MAG,
        help=(
            "Magnitude slice from --background-rate-file used as the spatial "
            f"background covariate. Default: {DEFAULT_BACKGROUND_RATE_MAG}"
        ),
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


def normalize_optional_path(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.abspath(path)


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
    requested_background = normalize_optional_path(args.background_rate_file)
    stored_background = normalize_optional_path(metadata.get("background_rate_file"))
    if requested_background != stored_background:
        return False
    if requested_background is not None:
        if float(metadata.get("requested_background_rate_mag", np.nan)) != float(args.background_rate_mag):
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
    if args.background_rate_file:
        command.extend(
            [
                "--background-rate-file",
                args.background_rate_file,
                "--background-rate-mag",
                str(args.background_rate_mag),
            ]
        )
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

    # Derive catalog metadata from the best scenario row
    mc_val = best.get("mc", 4.1) if "mc" in ordered.columns else 4.1
    tw_start = best.get("timewindow_start", "1960-01-01") if "timewindow_start" in ordered.columns else "1960-01-01"
    aux_start = best.get("auxiliary_start", "1950-01-01") if "auxiliary_start" in ordered.columns else "1950-01-01"

    lines = [
        "# NZ-Wide ETAS Calibration Sweep",
        "",
        "## Data & Configuration",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| **Catalog** | GeoNet NZ FDSN (`nzcat.csv`) |",
        f"| **Mc** | {mc_val} |",
        f"| **Training window** | {str(tw_start)[:10]} → {args.forecast_start[:10]} |",
        f"| **Auxiliary start** | {str(aux_start)[:10]} |",
        f"| **Region** | 34°S–48°S, 164°E–180°E (rectangular) |",
        f"| **Background rate grid** | {args.background_rate_file or 'disabled'} |",
        f"| **Background rate magnitude slice** | {args.background_rate_mag if args.background_rate_file else 'n/a'} |",
        f"| **Forecast origin** | `{args.forecast_start}` |",
        f"| **Forecast horizons** | {args.durations} days |",
        f"| **Simulations / scenario** | {args.n_simulations} |",
        f"| **Scenario count** | {len(ordered)} |",
        "",
        "## Ranking Rule",
        "",
        (
            "`calibration_score = mean_abs_log_count_ratio + consistency_penalty + "
            "spatial_penalty`"
        ),
        "",
        "- **Consistency penalty** penalizes failed N/M/S/PL CSEP windows",
        "- **Spatial penalty** penalizes empty-cell rate allocation, unsupported observed events, and mean absolute spatial residuals",
        "",
        "## Best Scenario",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Best scenario | **{best['scenario_name']}** |",
        f"| Score | {best['calibration_score']:.3f} |",
        f"| Mean obs/sim ratio | {best['mean_obs_to_sim_ratio']:.3f} |",
        f"| Empty-cell share | {best['mean_empty_cell_fraction']:.1%} |",
        f"| Unsupported obs share | {best['mean_zero_rate_observed_fraction']:.1%} |",
        "",
        "## CSEP Consistency Tests",
        "",
        "| Test | Fraction Passed | Status |",
        "| --- | ---: | --- |",
    ]

    test_cols = [
        ("N-test", "n_consistency_fraction"),
        ("M-test", "m_consistency_fraction"),
        ("S-test", "s_consistency_fraction"),
        ("PL-test", "pl_consistency_fraction"),
    ]
    for label, col in test_cols:
        val = float(best.get(col, 0))
        status = "✅ Pass" if val >= 0.5 else "⚠️ Fail"
        lines.append(f"| {label} | {val:.3f} | {status} |")

    lines.extend([
        "",
        "> **Interpretation**: A test passes if ≥ 50% of forecast horizons produce",
        "> consistency at the 95% significance level (quantile between 0.025 and 0.975).",
        "",
        "## Scenario Comparison",
        "",
        "| Scenario | Score | Obs/Sim ratio | Empty-cell share | Unsupported obs share | N frac | S frac | PL frac | log₁₀(μ) | log₁₀(k₀) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
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

    # Diagnostic notes
    ratio = best['mean_obs_to_sim_ratio']
    lines.extend([
        "",
        "## Diagnostic Notes",
        "",
    ])
    if ratio < 0.5:
        lines.append(
            f"- ⚠️ **Over-prediction**: Obs/Sim ratio = {ratio:.3f}. "
            "The model forecasts ~3× more events than observed. Consider lowering "
            "the background rate (log₁₀μ) or narrowing the study region."
        )
    if best.get('mean_empty_cell_fraction', 0) > 0.9:
        lines.append(
            f"- ⚠️ **High empty-cell share** ({best['mean_empty_cell_fraction']:.1%}): "
            "The rectangular bounding box includes large oceanic areas with no real "
            "seismicity. A tighter NZ seismogenic-zone polygon would reduce this penalty "
            "and improve spatial diagnostics."
        )
    if best.get('n_consistency_fraction', 1) < 0.5:
        lines.append(
            "- ⚠️ **N-test failure**: The event-count forecast is inconsistent with "
            "observations across most horizons. This is the primary driver of the "
            "calibration score."
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_scorecard(
    comparison_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate a publication-quality calibration scorecard figure."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import Normalize
    import matplotlib.patheffects as pe

    # ── Design tokens ──────────────────────────────────────────────────
    _DEEP_NAVY = "#0D1B2A"
    _SLATE = "#1B2838"
    _TEAL = "#1B9AAA"
    _CORAL = "#E8505B"
    _AMBER = "#F2A65A"
    _EMERALD = "#2ECC71"
    _MUTED = "#8D99AE"
    _LIGHT_BG = "#F7F9FC"
    _PANEL_BG = "#FFFFFF"
    _GRID_CLR = "#E2E8F0"
    _TEXT_CLR = "#2D3748"
    _SUBTEXT = "#718096"

    _CONSISTENCY_PALETTE = {
        "N": "#3B82F6",  # Blue
        "M": "#F59E0B",  # Amber
        "S": "#10B981",  # Green
        "PL": "#EF4444",  # Red
    }
    _HORIZON_PALETTE = ["#3B82F6", "#06B6D4", "#8B5CF6"]

    ordered = comparison_df.sort_values("calibration_score").reset_index(drop=True)
    scenario_names = ordered["scenario_name"].tolist()
    n_scenarios = len(ordered)

    # ── Figure layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14), facecolor=_LIGHT_BG)
    gs = GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[0.12, 1, 1],
        hspace=0.35,
        wspace=0.32,
        left=0.06,
        right=0.96,
        top=0.95,
        bottom=0.06,
    )

    # ── Row 0: Context banner ──────────────────────────────────────────
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.set_xlim(0, 1)
    ax_banner.set_ylim(0, 1)
    ax_banner.axis("off")

    # Title
    ax_banner.text(
        0.5, 0.82,
        "NZ-Wide ETAS Calibration Scorecard",
        ha="center", va="top",
        fontsize=20, fontweight="bold", color=_DEEP_NAVY,
        fontfamily="sans-serif",
    )

    # Metadata line
    best = ordered.iloc[0]
    mc_val = best.get("mc", 4.1) if "mc" in ordered.columns else 4.1
    tw_start = best.get("timewindow_start", "1960") if "timewindow_start" in ordered.columns else "1960"
    tw_start_short = str(tw_start)[:4] if tw_start else "?"
    meta_parts = [
        f"Catalog: GeoNet NZ  (M$_c$ = {mc_val})",
        f"Training: {tw_start_short}–2018",
        f"Region: 34°S–48°S, 164°E–180°E",
        f"Scenarios: {n_scenarios}",
    ]
    ax_banner.text(
        0.5, 0.25,
        "  ·  ".join(meta_parts),
        ha="center", va="center",
        fontsize=11, color=_SUBTEXT,
        fontfamily="sans-serif",
    )

    # Thin separator
    ax_banner.axhline(0.0, color=_GRID_CLR, linewidth=1.5)

    # ── Helper: Style an axis ──────────────────────────────────────────
    def _style_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(_PANEL_BG)
        ax.set_title(
            title, fontsize=13, fontweight="bold", color=_DEEP_NAVY,
            pad=10, loc="left",
        )
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, color=_TEXT_CLR)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, color=_TEXT_CLR)
        for spine in ax.spines.values():
            spine.set_color(_GRID_CLR)
            spine.set_linewidth(0.8)
        ax.tick_params(colors=_TEXT_CLR, labelsize=9)
        ax.grid(True, alpha=0.35, color=_GRID_CLR, linewidth=0.6)

    # ── Panel 1: Calibration Score ─────────────────────────────────────
    ax_score = fig.add_subplot(gs[1, 0])
    _style_ax(ax_score, "Calibration Score", xlabel="Score  (lower is better)")

    y = np.arange(n_scenarios)
    scores = ordered["calibration_score"].values
    norm = Normalize(vmin=scores.min() - 0.1, vmax=scores.max() + 0.5)

    bars = ax_score.barh(
        y, scores,
        height=0.6,
        color=[plt.cm.RdYlGn_r(norm(s)) for s in scores],
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    # Value labels
    for i, (bar, s) in enumerate(zip(bars, scores)):
        ax_score.text(
            s + scores.max() * 0.02, i,
            f"{s:.2f}",
            va="center", fontsize=10, fontweight="bold", color=_TEXT_CLR,
        )
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(scenario_names, fontsize=10, fontweight="medium")
    ax_score.invert_yaxis()
    ax_score.set_xlim(0, scores.max() * 1.25)

    # Best-scenario badge
    ax_score.annotate(
        "★ BEST",
        xy=(scores[0], 0),
        xytext=(-8, 0),
        textcoords="offset points",
        fontsize=8, fontweight="bold", color=_EMERALD,
        ha="right", va="center",
    )

    # ── Panel 2: CSEP Consistency ──────────────────────────────────────
    ax_consistency = fig.add_subplot(gs[1, 1])
    _style_ax(ax_consistency, "CSEP Consistency Tests", ylabel="Fraction Passed")

    x = np.arange(n_scenarios)
    bar_width = 0.18
    test_keys = [
        ("n_consistency_fraction", "N"),
        ("m_consistency_fraction", "M"),
        ("s_consistency_fraction", "S"),
        ("pl_consistency_fraction", "PL"),
    ]

    for offset, (col, label) in enumerate(test_keys):
        vals = ordered[col].values
        pos = x + (offset - 1.5) * bar_width
        b = ax_consistency.bar(
            pos, vals, width=bar_width,
            color=_CONSISTENCY_PALETTE[label],
            edgecolor="white", linewidth=0.8,
            label=f"{label}-test", zorder=3,
            alpha=0.88,
        )
        # Pass/fail symbols on top of bars
        for xi, v in zip(pos, vals):
            symbol = "✓" if v >= 0.5 else "✗"
            clr = _EMERALD if v >= 0.5 else _CORAL
            ax_consistency.text(
                xi, v + 0.03, symbol,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=clr,
            )

    ax_consistency.axhline(0.5, color=_CORAL, linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")
    ax_consistency.set_xticks(x)
    ax_consistency.set_xticklabels(scenario_names, fontsize=10)
    ax_consistency.set_ylim(0, 1.15)
    ax_consistency.legend(
        loc="upper right", frameon=True, framealpha=0.95,
        edgecolor=_GRID_CLR, fontsize=8, ncol=3,
    )

    # ── Panel 3: Obs / Sim Ratio ───────────────────────────────────────
    ax_ratio = fig.add_subplot(gs[1, 2])
    _style_ax(ax_ratio, "Obs / Sim Event Count Ratio", xlabel="Ratio  (1.0 = perfect)")

    # Ideal range shading
    ax_ratio.axvspan(0.5, 2.0, color=_EMERALD, alpha=0.06, zorder=1)
    ax_ratio.axvspan(0.8, 1.25, color=_EMERALD, alpha=0.10, zorder=1)
    ax_ratio.axvline(1.0, color=_MUTED, linestyle="--", linewidth=1.2, zorder=2)

    ratios = ordered["mean_obs_to_sim_ratio"].values
    ax_ratio.scatter(
        ratios, y,
        s=120, c=[_EMERALD if 0.5 < r < 2.0 else _CORAL for r in ratios],
        edgecolors=_DEEP_NAVY, linewidths=1.2, zorder=5,
    )
    for i, r in enumerate(ratios):
        ax_ratio.text(
            r + 0.02, i,
            f"{r:.2f}",
            va="center", fontsize=10, color=_TEXT_CLR, fontweight="medium",
        )
    ax_ratio.set_yticks(y)
    ax_ratio.set_yticklabels(scenario_names, fontsize=10)
    ax_ratio.invert_yaxis()

    # Add interpretation label
    ax_ratio.text(
        0.98, 0.02,
        "< 1 = model over-predicts\n> 1 = model under-predicts",
        transform=ax_ratio.transAxes, fontsize=8, color=_SUBTEXT,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=_LIGHT_BG, edgecolor=_GRID_CLR, alpha=0.9),
    )

    # ── Panel 4: Spatial Penalties ─────────────────────────────────────
    ax_spatial = fig.add_subplot(gs[2, 0])
    _style_ax(ax_spatial, "Spatial Allocation Diagnostics", ylabel="Percentage (%)")

    empty_vals = 100 * ordered["mean_empty_cell_fraction"].values
    unsup_vals = 100 * ordered["mean_zero_rate_observed_fraction"].values

    ax_spatial.bar(
        x - 0.15, empty_vals, width=0.28,
        color=_AMBER, edgecolor="white", linewidth=0.8,
        label="Empty-cell forecast share", zorder=3,
        alpha=0.85,
    )
    ax_spatial.bar(
        x + 0.15, unsup_vals, width=0.28,
        color=_CORAL, edgecolor="white", linewidth=0.8,
        label="Unsupported observed events", zorder=3,
        alpha=0.85,
    )

    # Value labels
    for i in range(n_scenarios):
        ax_spatial.text(
            x[i] - 0.15, empty_vals[i] + 1.5,
            f"{empty_vals[i]:.1f}%", ha="center", fontsize=8, color=_AMBER, fontweight="bold",
        )
        ax_spatial.text(
            x[i] + 0.15, unsup_vals[i] + 1.5,
            f"{unsup_vals[i]:.1f}%", ha="center", fontsize=8, color=_CORAL, fontweight="bold",
        )

    ax_spatial.set_xticks(x)
    ax_spatial.set_xticklabels(scenario_names, fontsize=10)
    ax_spatial.legend(
        loc="upper right", frameon=True, framealpha=0.95,
        edgecolor=_GRID_CLR, fontsize=9,
    )

    # ── Panel 5: Horizon Ratios (bar chart) ────────────────────────────
    ax_horizon = fig.add_subplot(gs[2, 1])
    _style_ax(ax_horizon, "Obs / Sim Ratio by Forecast Horizon", ylabel="Ratio")

    ratio_pivot = horizon_df.pivot_table(
        index="scenario_name",
        columns="duration_days",
        values="observed_to_sim_mean_ratio",
    )
    ratio_pivot = ratio_pivot.reindex(scenario_names)
    durations = ratio_pivot.columns.tolist()

    if n_scenarios == 1:
        # Grouped bar for single scenario
        horizon_x = np.arange(len(durations))
        vals = ratio_pivot.iloc[0].values
        horizon_bars = ax_horizon.bar(
            horizon_x, vals, width=0.5,
            color=_HORIZON_PALETTE[:len(durations)],
            edgecolor="white", linewidth=1.2, zorder=3,
        )
        for hx, hv in zip(horizon_x, vals):
            ax_horizon.text(
                hx, hv + 0.01, f"{hv:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=_TEXT_CLR,
            )
        ax_horizon.set_xticks(horizon_x)
        ax_horizon.set_xticklabels([f"{int(d)}-day" for d in durations], fontsize=10)
        ax_horizon.axhline(1.0, color=_MUTED, linestyle="--", linewidth=1, zorder=2)
        ax_horizon.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1.5)
    else:
        # Grouped bars per scenario for multi-scenario
        group_width = 0.8
        bar_w = group_width / len(durations)
        for j, dur in enumerate(durations):
            offset = (j - (len(durations) - 1) / 2) * bar_w
            vals = ratio_pivot[dur].values
            ax_horizon.bar(
                x + offset, vals, width=bar_w * 0.9,
                color=_HORIZON_PALETTE[j % len(_HORIZON_PALETTE)],
                edgecolor="white", linewidth=0.8,
                label=f"{int(dur)}-day", zorder=3,
            )
        ax_horizon.axhline(1.0, color=_MUTED, linestyle="--", linewidth=1, zorder=2)
        ax_horizon.set_xticks(x)
        ax_horizon.set_xticklabels(scenario_names, fontsize=10)
        ax_horizon.legend(
            loc="upper right", frameon=True, framealpha=0.95,
            edgecolor=_GRID_CLR, fontsize=9,
        )

    # ── Panel 6: Parameter Landscape ───────────────────────────────────
    ax_params = fig.add_subplot(gs[2, 2])

    mu_finite = np.isfinite(ordered["final_log10_mu"])
    k0_finite = np.isfinite(ordered["final_log10_k0"])
    mask = mu_finite & k0_finite

    if mask.any() and mask.sum() > 1:
        _style_ax(
            ax_params, "Parameter Landscape",
            xlabel=r"Final $\log_{10}(\mu)$",
            ylabel=r"Final $\log_{10}(k_0)$",
        )
        sc = ax_params.scatter(
            ordered.loc[mask, "final_log10_mu"],
            ordered.loc[mask, "final_log10_k0"],
            c=ordered.loc[mask, "calibration_score"],
            cmap="RdYlGn_r",
            s=160,
            edgecolors=_DEEP_NAVY,
            linewidths=1.5,
            zorder=5,
        )
        for row in ordered.loc[mask].itertuples(index=False):
            ax_params.annotate(
                row.scenario_name,
                (row.final_log10_mu, row.final_log10_k0),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9, fontweight="medium", color=_TEXT_CLR,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )
        cbar = plt.colorbar(sc, ax=ax_params, shrink=0.75, pad=0.03)
        cbar.set_label("Calibration Score", fontsize=9, color=_TEXT_CLR)
        cbar.ax.tick_params(labelsize=8)
    elif mask.any() and mask.sum() == 1:
        # Single scenario: show a summary card instead of a meaningless scatter
        _style_ax(ax_params, "Fitted Parameters")
        row = ordered.iloc[0]
        param_lines = [
            (r"$\log_{10}(\mu)$", f"{row.get('final_log10_mu', float('nan')):.3f}"),
            (r"$\log_{10}(k_0)$", f"{row.get('final_log10_k0', float('nan')):.3f}"),
        ]
        if "final_a" in ordered.columns:
            param_lines.append((r"$\alpha$", f"{row.get('final_a', float('nan')):.3f}"))
        if "final_gamma" in ordered.columns:
            param_lines.append((r"$\gamma$", f"{row.get('final_gamma', float('nan')):.3f}"))
        if "final_rho" in ordered.columns:
            param_lines.append((r"$\rho$", f"{row.get('final_rho', float('nan')):.3f}"))
        if "beta" in ordered.columns:
            param_lines.append((r"$\beta$", f"{row.get('beta', float('nan')):.3f}"))

        y_pos = 0.88
        for label, value in param_lines:
            ax_params.text(
                0.3, y_pos, label,
                transform=ax_params.transAxes,
                fontsize=13, ha="right", va="center", color=_SUBTEXT,
            )
            ax_params.text(
                0.35, y_pos, f"  =  {value}",
                transform=ax_params.transAxes,
                fontsize=13, ha="left", va="center",
                fontweight="bold", color=_DEEP_NAVY,
                fontfamily="monospace",
            )
            y_pos -= 0.14

        ax_params.set_xticks([])
        ax_params.set_yticks([])
    else:
        ax_params.axis("off")

    fig.savefig(output_path, dpi=300, facecolor=_LIGHT_BG)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.background_rate_file = normalize_optional_path(args.background_rate_file)
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
        "background_rate_file": args.background_rate_file,
        "background_rate_mag": args.background_rate_mag,
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
