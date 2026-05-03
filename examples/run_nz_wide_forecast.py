"""
Run a single NZ-wide ETAS forecast experiment.

This script is the non-sequence-specific counterpart to
``run_parallel_simulations.py``. It fits one NZ-wide ETAS model up to a single
forecast origin, simulates one or more forecast horizons, and optionally
generates CSEP-style evaluation plots against the observed future catalog.

Examples
--------
python run_nz_wide_forecast.py --forecast-start "2018-01-01 00:00:00"
python run_nz_wide_forecast.py --forecast-start "2018-01-01 00:00:00" \
    --durations 30,90 --n-simulations 500 --skip-plots
python run_nz_wide_forecast.py --forecast-start "2018-01-01 00:00:00" \
    --skip-pycsep-analysis
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
INPUT_DATA_DIR = os.path.join(ROOT_DIR, "input_data")

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "SeismoStats"))

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation
from visualize_results import plot_csep_6panel


warnings.filterwarnings("ignore")

DEFAULT_EXPERIMENT_NAME = "nz_wide"
DEFAULT_FORECAST_DURATIONS = [30, 90, 365]
DEFAULT_FORECAST_START = "2018-01-01 00:00:00"
DEFAULT_N_SIMULATIONS = 250
DEFAULT_TIMEWINDOW_START = "1960-01-01 00:00:00"
DEFAULT_AUXILIARY_START = "1950-01-01 00:00:00"
DEFAULT_MC = 4.1
DEFAULT_M_MAX = None
LAT_RANGE = (-48.0, -34.0)
LON_RANGE = (164.0, 180.0)

NZ_INITIAL_THETA = {
    "log10_mu": -7.477863177977867,
    "log10_iota": None,
    "log10_k0": -0.8570602601363014,
    "a": 1.4333791204125566,
    "log10_c": -3.1859152978148644,
    "omega": -0.08102742585588284,
    "log10_tau": 4.038107413059718,
    "log10_d": 1.588041892797509,
    "gamma": 0.34307084228763013,
    "rho": 0.8062060642600785,
}

CATALOG_PATH = os.path.join(INPUT_DATA_DIR, "nzcat.csv")
POLYGON_PATH = os.path.join(INPUT_DATA_DIR, "nz_polygon.npy")

set_up_logger(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and simulate a single NZ-wide ETAS forecast without any "
            "sequence-specific date grid or mainshock logic."
        )
    )
    parser.add_argument(
        "--forecast-start",
        default=DEFAULT_FORECAST_START,
        help=(
            "Forecast origin in 'YYYY-MM-DD HH:MM:SS' format. "
            f"Default: {DEFAULT_FORECAST_START}"
        ),
    )
    parser.add_argument(
        "--durations",
        default=",".join(str(d) for d in DEFAULT_FORECAST_DURATIONS),
        help="Comma-separated forecast durations in days. Default: 30,90,365",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=DEFAULT_N_SIMULATIONS,
        help=f"Number of synthetic catalogs per duration. Default: {DEFAULT_N_SIMULATIONS}",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"Output prefix for this run. Default: {DEFAULT_EXPERIMENT_NAME}",
    )
    parser.add_argument(
        "--timewindow-start",
        default=DEFAULT_TIMEWINDOW_START,
        help=f"ETAS inversion start. Default: {DEFAULT_TIMEWINDOW_START}",
    )
    parser.add_argument(
        "--auxiliary-start",
        default=DEFAULT_AUXILIARY_START,
        help=f"Auxiliary start. Default: {DEFAULT_AUXILIARY_START}",
    )
    parser.add_argument(
        "--mc",
        type=float,
        default=DEFAULT_MC,
        help=f"Magnitude of completeness. Default: {DEFAULT_MC}",
    )
    parser.add_argument(
        "--m-max",
        type=float,
        default=DEFAULT_M_MAX,
        help="Maximum simulated magnitude. Default: no explicit cap.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip the 6-panel CSEP-style evaluation plots.",
    )
    parser.add_argument(
        "--theta-log10-mu-delta",
        type=float,
        default=0.0,
        help=(
            "Additive shift applied to the initial log10_mu guess before inversion. "
            "Default: 0.0"
        ),
    )
    parser.add_argument(
        "--theta-log10-k0-delta",
        type=float,
        default=0.0,
        help=(
            "Additive shift applied to the initial log10_k0 guess before inversion. "
            "Default: 0.0"
        ),
    )
    parser.add_argument(
        "--force-reinvert",
        action="store_true",
        help="Ignore any cached parameter file for this run label and rerun the inversion.",
    )
    parser.add_argument(
        "--skip-pycsep-analysis",
        action="store_true",
        help="Skip the automatic pyCSEP catalog analysis step.",
    )
    parser.add_argument(
        "--pycsep-region-source",
        choices=["forecast_domain", "nz_csep_collection"],
        default="forecast_domain",
        help=(
            "Spatial region used by the automatic pyCSEP analysis. "
            "Default: forecast_domain"
        ),
    )
    parser.add_argument(
        "--pycsep-grid-spacing",
        type=float,
        default=0.1,
        help="Grid spacing in degrees for forecast_domain pyCSEP regions. Default: 0.1",
    )
    parser.add_argument(
        "--pycsep-mag-bin",
        type=float,
        default=0.1,
        help="Magnitude bin width for pyCSEP analysis. Default: 0.1",
    )
    parser.add_argument(
        "--pycsep-max-mag",
        type=float,
        default=None,
        help="Maximum magnitude bin edge for pyCSEP analysis. Default: inferred.",
    )
    return parser.parse_args()


def parse_datetime(value: str) -> dt.datetime:
    return pd.Timestamp(value).to_pydatetime()


def parse_durations(raw_value: str) -> list[int]:
    durations = []
    for piece in raw_value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        durations.append(int(float(piece)))
    if not durations:
        raise ValueError("At least one forecast duration is required.")
    return sorted(set(durations))


def slugify(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum():
            cleaned.append(char.lower())
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "nz_wide"


def ensure_inputs_exist() -> None:
    missing = [path for path in (CATALOG_PATH, POLYGON_PATH) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Required NZ-wide input files are missing: " + ", ".join(missing)
        )


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def as_store_dir(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def load_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(CATALOG_PATH, index_col=0, parse_dates=["time"])
    catalog.sort_values(by="time", inplace=True)
    mask = (
        (catalog["latitude"] >= LAT_RANGE[0])
        & (catalog["latitude"] <= LAT_RANGE[1])
        & (catalog["longitude"] >= LON_RANGE[0])
        & (catalog["longitude"] <= LON_RANGE[1])
    )
    return catalog.loc[mask].copy()


def build_initial_theta(
    log10_mu_delta: float = 0.0,
    log10_k0_delta: float = 0.0,
) -> dict:
    theta = NZ_INITIAL_THETA.copy()
    theta["log10_mu"] = float(theta["log10_mu"] + log10_mu_delta)
    theta["log10_k0"] = float(theta["log10_k0"] + log10_k0_delta)
    return theta


def build_run_paths(experiment_name: str, forecast_start: dt.datetime) -> dict[str, str]:
    run_label = f"{slugify(experiment_name)}_{forecast_start:%Y%m%d_%H%M%S}"
    output_dir = ensure_dir(os.path.join(BASE_DIR, "output_nz_wide", run_label))
    simulation_dir = ensure_dir(os.path.join(BASE_DIR, "simulations_nz_wide", run_label))
    figure_dir = ensure_dir(os.path.join(BASE_DIR, "figures", "nz_wide", run_label))
    return {
        "run_label": run_label,
        "output_dir": output_dir,
        "simulation_dir": simulation_dir,
        "figure_dir": figure_dir,
        "metadata_path": os.path.join(output_dir, "experiment_config.json"),
        "evaluation_summary_path": os.path.join(output_dir, "evaluation_summary.csv"),
    }


def build_inversion_config(
    run_label: str,
    forecast_start: dt.datetime,
    auxiliary_start: str,
    timewindow_start: str,
    mc: float,
    initial_theta: dict,
) -> dict:
    return {
        "fn_catalog": CATALOG_PATH,
        "auxiliary_start": auxiliary_start,
        "timewindow_start": timewindow_start,
        "timewindow_end": forecast_start.strftime("%Y-%m-%d %H:%M:%S"),
        "theta_0": initial_theta.copy(),
        "mc": mc,
        "m_ref": mc,
        "delta_m": 0.1,
        "coppersmith_multiplier": 100,
        "shape_coords": POLYGON_PATH,
        "name": "nz_wide_standard",
        "id": run_label,
    }


def load_or_run_inversion(
    config: dict,
    output_dir: str,
    force_reinvert: bool = False,
) -> tuple[ETASParameterCalculation, str]:
    parameter_path = os.path.join(output_dir, f"parameters_{config['id']}.json")
    if os.path.exists(parameter_path) and not force_reinvert:
        logger.info("Loading existing inversion from %s", parameter_path)
        with open(parameter_path, "r") as f:
            inversion_output = json.load(f)
        inversion_output["fn_catalog"] = config["fn_catalog"]
        inversion_output["shape_coords"] = config["shape_coords"]
        calculation = ETASParameterCalculation.load_calculation(inversion_output)
        return calculation, parameter_path

    if os.path.exists(parameter_path) and force_reinvert:
        logger.info("Forcing reinversion for %s despite cached parameters.", config["id"])
    logger.info("Running NZ-wide inversion for %s", config["timewindow_end"])
    calculation = ETASParameterCalculation(config)
    calculation.prepare()
    calculation.invert()
    calculation.store_results(as_store_dir(output_dir), store_pij=False)
    logger.info("Stored inversion outputs in %s", output_dir)
    return calculation, parameter_path


def run_simulations(
    calculation: ETASParameterCalculation,
    durations: list[int],
    n_simulations: int,
    mc: float,
    m_max: float | None,
    simulation_dir: str,
) -> dict[int, str]:
    simulation = ETASSimulation(calculation, m_max=m_max, approx_times=True)
    simulation.prepare()

    simulation_paths = {}
    for duration in durations:
        simulation_path = os.path.join(simulation_dir, f"forecasts_{duration}days.csv")
        logger.info(
            "Simulating %s NZ-wide catalogs for %s days into %s",
            n_simulations,
            duration,
            simulation_path,
        )
        simulation.simulate_to_csv(
            simulation_path,
            duration,
            n_simulations,
            m_threshold=mc,
        )
        simulation_paths[duration] = simulation_path
    return simulation_paths


def build_observed_windows(
    catalog: pd.DataFrame,
    forecast_start: dt.datetime,
    durations: list[int],
    mc: float,
    output_dir: str,
) -> tuple[dict[int, pd.DataFrame], dict[int, str]]:
    windows = {}
    stored_paths = {}
    for duration in durations:
        window_end = forecast_start + dt.timedelta(days=duration)
        observed = catalog[
            (catalog["time"] >= forecast_start)
            & (catalog["time"] < window_end)
            & (catalog["magnitude"] >= mc)
        ].copy()
        windows[duration] = observed
        observed_path = os.path.join(output_dir, f"observed_{duration}days.csv")
        observed.to_csv(observed_path, index=False)
        stored_paths[duration] = observed_path
    return windows, stored_paths


def evaluate_forecasts(
    simulation_paths: dict[int, str],
    observed_windows: dict[int, pd.DataFrame],
    forecast_start: dt.datetime,
    mc: float,
    figure_dir: str,
    skip_plots: bool,
) -> pd.DataFrame:
    records = []

    for duration, simulation_path in simulation_paths.items():
        simulations = pd.read_csv(simulation_path, index_col=0, parse_dates=["time"])
        observed = observed_windows[duration]
        plot_path = ""

        if not skip_plots:
            plot_path = os.path.join(figure_dir, f"csep_6panel_{duration}days.png")
            plot_csep_6panel(
                simulations,
                observed,
                {
                    "mc": mc,
                    "start": forecast_start,
                    "duration": duration,
                    "shape_coords": POLYGON_PATH,
                },
                output_path=plot_path,
            )

        mean_sim_count = simulations.groupby("catalog_id").size().mean()
        max_mag = observed["magnitude"].max() if len(observed) else float("nan")
        records.append(
            {
                "duration_days": duration,
                "simulation_file": simulation_path,
                "plot_file": plot_path,
                "observed_count": len(observed),
                "observed_max_magnitude": max_mag,
                "catalogs_with_events": simulations["catalog_id"].nunique(),
                "mean_simulated_event_count": mean_sim_count,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("duration_days")


def write_metadata(
    metadata_path: str,
    args: argparse.Namespace,
    forecast_start: dt.datetime,
    durations: list[int],
    catalog: pd.DataFrame,
    output_paths: dict[str, str],
    parameter_path: str,
    simulation_paths: dict[int, str],
    observed_paths: dict[int, str],
    evaluation_summary_path: str,
    initial_theta: dict,
) -> None:
    catalog_end = pd.Timestamp(catalog["time"].max()).strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "workflow": "single_nz_wide_forecast",
        "sequence_specific": False,
        "experiment_name": args.experiment_name,
        "run_label": output_paths["run_label"],
        "forecast_start": forecast_start.strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_durations_days": durations,
        "n_simulations": args.n_simulations,
        "mc": args.mc,
        "m_max": args.m_max,
        "catalog_path": CATALOG_PATH,
        "polygon_path": POLYGON_PATH,
        "lat_range": list(LAT_RANGE),
        "lon_range": list(LON_RANGE),
        "auxiliary_start": args.auxiliary_start,
        "timewindow_start": args.timewindow_start,
        "theta_initial": initial_theta,
        "theta_log10_mu_delta": args.theta_log10_mu_delta,
        "theta_log10_k0_delta": args.theta_log10_k0_delta,
        "force_reinvert": args.force_reinvert,
        "catalog_last_time": catalog_end,
        "parameter_file": parameter_path,
        "simulation_files": {str(k): v for k, v in simulation_paths.items()},
        "observed_files": {str(k): v for k, v in observed_paths.items()},
        "evaluation_summary_file": evaluation_summary_path,
        "skip_plots": args.skip_plots,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def update_metadata_with_pycsep(metadata_path: str, pycsep_payload: dict) -> None:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata["pycsep_analysis"] = pycsep_payload
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def maybe_run_pycsep_analysis(
    args: argparse.Namespace,
    metadata_path: str,
) -> dict | None:
    if args.skip_pycsep_analysis:
        logger.info("Skipping pyCSEP analysis because --skip-pycsep-analysis was provided.")
        update_metadata_with_pycsep(
            metadata_path,
            {"status": "skipped"},
        )
        return None

    try:
        from run_nz_wide_pycsep_analysis import run_analysis_from_metadata
    except ImportError as exc:
        logger.warning("pyCSEP analysis unavailable: %s", exc)
        update_metadata_with_pycsep(
            metadata_path,
            {"status": "import_failed", "error": str(exc)},
        )
        return None

    try:
        logger.info("Running automatic pyCSEP analysis for %s", metadata_path)
        outputs = run_analysis_from_metadata(
            metadata_path=metadata_path,
            region_source=args.pycsep_region_source,
            grid_spacing=args.pycsep_grid_spacing,
            mag_bin=args.pycsep_mag_bin,
            max_mag=args.pycsep_max_mag,
            output_dir=None,
        )
    except Exception as exc:
        logger.warning("pyCSEP analysis failed: %s", exc)
        update_metadata_with_pycsep(
            metadata_path,
            {"status": "failed", "error": str(exc)},
        )
        return None

    payload = {
        "status": "completed",
        "region_source": args.pycsep_region_source,
        "grid_spacing": args.pycsep_grid_spacing,
        "mag_bin": args.pycsep_mag_bin,
        "max_mag": args.pycsep_max_mag,
        "output_dir": outputs.output_dir,
        "summary_csv": outputs.summary_csv_path,
        "results_json": outputs.results_json_path,
        "report_md": outputs.report_path,
        "overview_png": outputs.overview_path,
        "dashboard_pngs": outputs.dashboard_paths,
    }
    update_metadata_with_pycsep(metadata_path, payload)
    logger.info("pyCSEP analysis outputs stored in %s", outputs.output_dir)
    return payload


def main() -> None:
    args = parse_args()
    ensure_inputs_exist()

    forecast_start = parse_datetime(args.forecast_start)
    durations = parse_durations(args.durations)
    output_paths = build_run_paths(args.experiment_name, forecast_start)

    logger.info("Starting NZ-wide ETAS forecast run: %s", output_paths["run_label"])
    logger.info("Forecast origin: %s", forecast_start)
    logger.info("Durations: %s days", durations)
    logger.info("Simulations per duration: %s", args.n_simulations)

    catalog = load_catalog()
    if forecast_start <= pd.Timestamp(args.timewindow_start).to_pydatetime():
        raise ValueError("forecast_start must be after timewindow_start.")

    training_catalog = catalog[catalog["time"] < forecast_start]
    if training_catalog.empty:
        raise ValueError("No training events remain before the forecast origin.")

    max_duration = max(durations)
    catalog_last_time = pd.Timestamp(catalog["time"].max()).to_pydatetime()
    requested_end = forecast_start + dt.timedelta(days=max_duration)
    if requested_end > catalog_last_time:
        logger.warning(
            "Catalog ends at %s, earlier than the full requested observed window %s.",
            catalog_last_time,
            requested_end,
        )

    logger.info("Training events before forecast origin: %s", len(training_catalog))

    initial_theta = build_initial_theta(
        log10_mu_delta=args.theta_log10_mu_delta,
        log10_k0_delta=args.theta_log10_k0_delta,
    )
    config = build_inversion_config(
        output_paths["run_label"],
        forecast_start,
        args.auxiliary_start,
        args.timewindow_start,
        args.mc,
        initial_theta,
    )
    calculation, parameter_path = load_or_run_inversion(
        config,
        output_paths["output_dir"],
        force_reinvert=args.force_reinvert,
    )

    simulation_paths = run_simulations(
        calculation,
        durations,
        args.n_simulations,
        args.mc,
        args.m_max,
        output_paths["simulation_dir"],
    )
    observed_windows, observed_paths = build_observed_windows(
        catalog,
        forecast_start,
        durations,
        args.mc,
        output_paths["output_dir"],
    )
    evaluation_summary = evaluate_forecasts(
        simulation_paths,
        observed_windows,
        forecast_start,
        args.mc,
        output_paths["figure_dir"],
        args.skip_plots,
    )
    evaluation_summary.to_csv(output_paths["evaluation_summary_path"], index=False)

    write_metadata(
        output_paths["metadata_path"],
        args,
        forecast_start,
        durations,
        catalog,
        output_paths,
        parameter_path,
        simulation_paths,
        observed_paths,
        output_paths["evaluation_summary_path"],
        initial_theta,
    )
    pycsep_payload = maybe_run_pycsep_analysis(args, output_paths["metadata_path"])

    logger.info("NZ-wide run finished.")
    logger.info("Parameter file: %s", parameter_path)
    logger.info("Evaluation summary: %s", output_paths["evaluation_summary_path"])
    if pycsep_payload and pycsep_payload.get("status") == "completed":
        logger.info("pyCSEP report: %s", pycsep_payload["report_md"])
        logger.info("pyCSEP overview: %s", pycsep_payload["overview_png"])
    for duration, simulation_path in simulation_paths.items():
        logger.info("Forecast %s days: %s", duration, simulation_path)


if __name__ == "__main__":
    main()
