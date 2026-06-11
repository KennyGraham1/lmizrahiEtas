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
import multiprocessing
import os
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
INPUT_DATA_DIR = os.path.join(ROOT_DIR, "input_data")

# Insert at the front so the in-repo `etas/` and `SeismoStats/` take
# precedence over any copy pip-installed into the active environment.
# (A non-editable `etas` install in site-packages would otherwise shadow
# the repo when this script is launched from examples/, silently running
# stale code.)
sys.path.insert(0, os.path.join(ROOT_DIR, "SeismoStats"))
sys.path.insert(0, ROOT_DIR)

from etas import set_up_logger
from etas.inversion import (
    ETASParameterCalculation,
    assess_inversion_degeneracy,
)
from etas.simulation import ETASSimulation
from visualize_results import plot_csep_6panel


warnings.filterwarnings("ignore")

DEFAULT_EXPERIMENT_NAME = "nz_wide"
DEFAULT_FORECAST_DURATIONS = [365, 730, 1095, 1461, 1826]
DEFAULT_FORECAST_START = "2021-01-01 00:00:00"
DEFAULT_N_SIMULATIONS = 2000
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
BACKGROUND_RATE_COLUMN = "hft_background_rate"

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
        help=(
            "Comma-separated forecast durations in days. Default: "
            + ",".join(str(d) for d in DEFAULT_FORECAST_DURATIONS)
        ),
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=DEFAULT_N_SIMULATIONS,
        help=f"Number of synthetic catalogs per duration. Default: {DEFAULT_N_SIMULATIONS}",
    )
    parser.add_argument(
        "--simulation-workers",
        type=int,
        default=0,
        help=(
            "Worker processes simulating catalogs in parallel. Each worker "
            "writes a disjoint catalog_id range to a part file, merged into "
            "the usual forecasts_<duration>days.csv afterwards. "
            "0 picks a RAM- and CPU-aware default; 1 disables parallelism."
        ),
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
        "--force-resimulate",
        action="store_true",
        help=(
            "Delete and regenerate forecast simulation files even if they exist. "
            "Re-inversion (--force-reinvert) implies this, since stale forecasts "
            "would otherwise be reused with the old parameters."
        ),
    )
    parser.add_argument(
        "--allow-degenerate-inversion",
        action="store_true",
        help=(
            "Proceed to simulation even if the ETAS inversion collapsed to a "
            "triggering-free (background-only) solution. By default such a run "
            "aborts, because the resulting forecast is not a real ETAS forecast."
        ),
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
    parser.add_argument(
        "--background-rate-file",
        default=None,
        help=(
            "Optional long-term background-rate grid with columns "
            "'longitude latitude magnitude rate'. If provided, rates are mapped "
            "to catalog events and used as bg_term during inversion."
        ),
    )
    parser.add_argument(
        "--background-rate-mag",
        type=float,
        default=5.0,
        help=(
            "Magnitude slice to use from --background-rate-file. "
            "Default: 5.0"
        ),
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


def resolve_path(path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.join(ROOT_DIR, path),
        os.path.join(BASE_DIR, path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path)


def ensure_inputs_exist(background_rate_file: str | None = None) -> None:
    missing = [path for path in (CATALOG_PATH, POLYGON_PATH) if not os.path.exists(path)]
    if background_rate_file and not os.path.exists(background_rate_file):
        missing.append(background_rate_file)
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
    
    # Filter by the polygon instead of a bounding box
    if os.path.exists(POLYGON_PATH):
        coords = np.load(POLYGON_PATH)
        # coords is likely [lat, lon] based on how it's created and used in pyCSEP
        from matplotlib.path import Path
        poly_path = Path(np.column_stack([coords[:, 1], coords[:, 0]])) # [lon, lat]
        points = np.column_stack([catalog["longitude"], catalog["latitude"]])
        mask = poly_path.contains_points(points)
    else:
        mask = (
            (catalog["latitude"] >= LAT_RANGE[0])
            & (catalog["latitude"] <= LAT_RANGE[1])
            & (catalog["longitude"] >= LON_RANGE[0])
            & (catalog["longitude"] <= LON_RANGE[1])
        )
    return catalog.loc[mask].copy()


def load_background_rate_grid(path: str, magnitude: float) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "magnitude", "rate"],
        dtype=float,
    )
    available_magnitudes = np.sort(raw["magnitude"].unique())
    selected_magnitude = float(
        available_magnitudes[np.argmin(np.abs(available_magnitudes - magnitude))]
    )
    grid = raw[np.isclose(raw["magnitude"], selected_magnitude)].copy()
    grid = (
        grid.groupby(["longitude", "latitude"], as_index=False)["rate"]
        .sum()
        .sort_values(["longitude", "latitude"])
        .reset_index(drop=True)
    )
    grid["rate"] = grid["rate"].clip(lower=0)
    if float(grid["rate"].sum()) <= 0:
        raise ValueError(
            f"Background grid {path} has no positive rates at magnitude "
            f"{selected_magnitude}."
        )
    metadata = {
        "background_rate_file": path,
        "requested_background_rate_mag": magnitude,
        "selected_background_rate_mag": selected_magnitude,
        "background_rate_grid_points": int(len(grid)),
        "background_rate_sum": float(grid["rate"].sum()),
        "background_rate_max": float(grid["rate"].max()),
    }
    return grid, metadata


def attach_background_rates_to_catalog(
    catalog: pd.DataFrame,
    background_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    tree = cKDTree(background_grid[["longitude", "latitude"]].to_numpy())
    distances, indexes = tree.query(catalog[["longitude", "latitude"]].to_numpy(), k=1)

    augmented = catalog.copy()
    augmented[BACKGROUND_RATE_COLUMN] = background_grid["rate"].to_numpy()[indexes]
    metadata = {
        "background_rate_column": BACKGROUND_RATE_COLUMN,
        "catalog_background_rate_min": float(augmented[BACKGROUND_RATE_COLUMN].min()),
        "catalog_background_rate_max": float(augmented[BACKGROUND_RATE_COLUMN].max()),
        "catalog_background_rate_sum": float(augmented[BACKGROUND_RATE_COLUMN].sum()),
        "catalog_background_nearest_distance_max_deg": float(np.max(distances)),
        "catalog_background_nearest_distance_mean_deg": float(np.mean(distances)),
    }
    if metadata["catalog_background_rate_sum"] <= 0:
        raise ValueError("All catalog events mapped to zero background-grid rate.")
    return augmented, metadata


def prepare_background_rate_catalog(
    catalog: pd.DataFrame,
    output_dir: str,
    run_label: str,
    background_rate_file: str | None,
    background_rate_mag: float,
) -> tuple[str, pd.DataFrame | None, dict]:
    if background_rate_file is None:
        return CATALOG_PATH, None, {}

    background_grid, grid_metadata = load_background_rate_grid(
        background_rate_file,
        background_rate_mag,
    )
    augmented_catalog, catalog_metadata = attach_background_rates_to_catalog(
        catalog,
        background_grid,
    )
    catalog_path = os.path.join(output_dir, f"catalog_{run_label}_with_background.csv")
    augmented_catalog.to_csv(catalog_path)
    metadata = {
        **grid_metadata,
        **catalog_metadata,
        "background_augmented_catalog": catalog_path,
    }
    return catalog_path, background_grid, metadata


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
    catalog_path: str,
    forecast_start: dt.datetime,
    auxiliary_start: str,
    timewindow_start: str,
    mc: float,
    initial_theta: dict,
    bg_term: str | None = None,
) -> dict:
    theta = initial_theta.copy()
    if bg_term is not None and theta.get("log10_iota") is None:
        theta["log10_iota"] = theta["log10_mu"]

    config = {
        "fn_catalog": catalog_path,
        "auxiliary_start": auxiliary_start,
        "timewindow_start": timewindow_start,
        "timewindow_end": forecast_start.strftime("%Y-%m-%d %H:%M:%S"),
        "theta_0": theta,
        "mc": mc,
        "m_ref": mc,
        "delta_m": 0.1,
        "coppersmith_multiplier": 100,
        "shape_coords": POLYGON_PATH,
        "name": "nz_wide_standard",
        "id": run_label,
    }
    if bg_term is not None:
        config["bg_term"] = bg_term
    return config


def load_or_run_inversion(
    config: dict,
    output_dir: str,
    force_reinvert: bool = False,
) -> tuple[ETASParameterCalculation, str]:
    parameter_path = os.path.join(output_dir, f"parameters_{config['id']}.json")
    if os.path.exists(parameter_path) and not force_reinvert:
        with open(parameter_path, "r") as f:
            inversion_output = json.load(f)
        cached_bg_term = inversion_output.get("bg_term")
        requested_bg_term = config.get("bg_term")
        if cached_bg_term == requested_bg_term:
            logger.info("Loading existing inversion from %s", parameter_path)
            inversion_output["fn_catalog"] = config["fn_catalog"]
            inversion_output["shape_coords"] = config["shape_coords"]
            calculation = ETASParameterCalculation.load_calculation(inversion_output)
            return calculation, parameter_path
        logger.info(
            "Cached inversion bg_term=%r does not match requested bg_term=%r; "
            "rerunning inversion.",
            cached_bg_term,
            requested_bg_term,
        )

    if os.path.exists(parameter_path) and force_reinvert:
        logger.info("Forcing reinversion for %s despite cached parameters.", config["id"])
    logger.info("Running NZ-wide inversion for %s", config["timewindow_end"])
    calculation = ETASParameterCalculation(config)
    calculation.prepare()
    calculation.invert()
    calculation.store_results(as_store_dir(output_dir), store_pij=False)
    logger.info("Stored inversion outputs in %s", output_dir)
    return calculation, parameter_path


def guard_against_degenerate_inversion(
    calculation: ETASParameterCalculation,
    allow_degenerate: bool,
) -> dict:
    """
    Refuse to simulate on an inversion that collapsed to a triggering-free
    (background-only) solution, unless explicitly allowed.

    Such a fit produces a stationary inhomogeneous-Poisson "forecast" with no
    aftershock triggering, which fails spatial/likelihood CSEP tests by
    construction. Catching it here avoids spending hours simulating a dead model.
    """
    degenerate, info = assess_inversion_degeneracy(
        calculation.theta,
        calculation.beta,
        n_hat=getattr(calculation, "n_hat", None),
    )
    branching = info["branching_ratio"]
    logger.info(
        "Inversion branching ratio: %s",
        f"{branching:.4g}" if branching is not None else "undefined",
    )
    if degenerate:
        reasons = "; ".join(info["reasons"])
        message = (
            "ETAS inversion is degenerate (%s) and would produce an unreliable "
            "forecast. A sub-critical branching ratio means essentially no "
            "triggering (stationary background-only forecast, usually a "
            "spatial background term that over-explains the catalog); a "
            "super-critical ratio (>= 1) means triggering never dies out, so "
            "the forecast is non-stationary and over-predicts. Aborting before "
            "simulation."
        )
        if allow_degenerate:
            logger.warning(message + " Continuing anyway (--allow-degenerate-inversion).", reasons)
        else:
            raise RuntimeError(
                (message % reasons)
                + " For a collapsed fit, re-run without --background-rate-file "
                "(homogeneous background) or supply a declustered/independent "
                "background grid; for a supercritical fit, constrain the "
                "branching ratio below 1. Or pass --allow-degenerate-inversion "
                "to override."
            )
    return info


def resolve_simulation_workers(requested: int, n_simulations: int) -> int:
    if requested == 0:
        try:
            import psutil

            ram_workers = int(psutil.virtual_memory().available / (2.0 * 1024**3))
        except ImportError:
            ram_workers = 10
        requested = min(12, max(1, ram_workers), os.cpu_count() or 1)
    # Each worker must simulate at least 2 catalogs: ETASSimulation.simulate
    # omits the catalog_id column when its end index is 1, which would make
    # the first part file unmergeable.
    return max(1, min(requested, n_simulations // 2))


# Cached per worker process so consecutive part tasks (one per forecast
# duration) skip re-reading the catalog and source/target CSVs.
_WORKER_CALCULATION = None
_WORKER_CALCULATION_KEY = None


def _simulate_part(task: dict) -> str:
    global _WORKER_CALCULATION, _WORKER_CALCULATION_KEY

    # Simulation kernels operate on tiny per-catalog arrays; numba's default
    # thread fan-out costs more than it buys and oversubscribes the machine
    # when many workers run at once.
    import numba

    numba.set_num_threads(1)

    if _WORKER_CALCULATION_KEY != task["parameter_path"]:
        with open(task["parameter_path"], "r") as f:
            inversion_output = json.load(f)
        inversion_output["fn_catalog"] = task["fn_catalog"]
        inversion_output["shape_coords"] = task["shape_coords"]
        _WORKER_CALCULATION = ETASParameterCalculation.load_calculation(inversion_output)
        _WORKER_CALCULATION_KEY = task["parameter_path"]

    simulation = ETASSimulation(
        _WORKER_CALCULATION,
        m_max=task["m_max"],
        approx_times=True,
        induced_info=task["induced_info"],
    )
    simulation.prepare()

    part_path = task["part_path"]
    # simulate_to_csv()'s resume logic compares the stored catalog_id against
    # the local simulation count, which is wrong for i_start > 0; a leftover
    # part from an interrupted run must be regenerated, not resumed.
    if os.path.exists(part_path):
        os.remove(part_path)
    simulation.simulate_to_csv(
        part_path,
        task["duration"],
        task["n_simulations"],
        m_threshold=task["m_threshold"],
        i_start=task["i_start"],
    )
    return part_path


def _run_parallel_simulation(
    executor: ProcessPoolExecutor,
    workers: int,
    parameter_path: str,
    fn_catalog: str,
    shape_coords: str,
    m_max: float | None,
    induced_info: list | None,
    simulation_path: str,
    duration: int,
    n_simulations: int,
    m_threshold: float,
) -> None:
    base, remainder = divmod(n_simulations, workers)
    tasks = []
    i_start = 0
    for worker_index in range(workers):
        count = base + (1 if worker_index < remainder else 0)
        if count == 0:
            continue
        tasks.append(
            {
                "parameter_path": parameter_path,
                "fn_catalog": fn_catalog,
                "shape_coords": shape_coords,
                "m_max": m_max,
                "induced_info": induced_info,
                "duration": duration,
                "n_simulations": count,
                "i_start": i_start,
                "m_threshold": m_threshold,
                "part_path": f"{simulation_path}.part{worker_index:03d}",
            }
        )
        i_start += count

    # executor.map preserves task order, so parts merge in ascending
    # catalog_id order, matching the layout of a serially written file.
    part_paths = list(executor.map(_simulate_part, tasks))

    with open(simulation_path, "w") as merged:
        for k, part_path in enumerate(part_paths):
            with open(part_path, "r") as part:
                if k > 0:
                    next(part, None)
                shutil.copyfileobj(part, merged)
    for part_path in part_paths:
        os.remove(part_path)


def run_simulations(
    calculation: ETASParameterCalculation,
    durations: list[int],
    n_simulations: int,
    mc: float,
    m_max: float | None,
    simulation_dir: str,
    background_grid: pd.DataFrame | None = None,
    overwrite: bool = False,
    simulation_workers: int = 1,
    parameter_path: str | None = None,
    fn_catalog: str | None = None,
    shape_coords: str | None = None,
) -> dict[int, str]:
    simulation_paths = {}
    workers = max(1, simulation_workers) if parameter_path is not None else 1
    executor = None
    if workers > 1:
        os.makedirs(simulation_dir, exist_ok=True)
        # Spawn (not fork): the parent has already run numba parallel kernels
        # during the inversion, and forking a process with live threading
        # runtimes is unsafe.
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        logger.info("Simulating with %s parallel workers", workers)
    try:
        for duration in durations:
            induced_info = build_background_induced_info(
                calculation,
                background_grid,
                duration,
            )
            simulation_path = os.path.join(simulation_dir, f"forecasts_{duration}days.csv")
            # simulate_to_csv() resumes/skips when the file already exists, so a
            # stale forecast from a previous (e.g. degenerate) inversion would be
            # silently reused. Remove it first when the caller asks to overwrite.
            if overwrite and os.path.exists(simulation_path):
                logger.info("Removing stale forecast file %s before re-simulating", simulation_path)
                os.remove(simulation_path)
            logger.info(
                "Simulating %s NZ-wide catalogs for %s days into %s",
                n_simulations,
                duration,
                simulation_path,
            )
            if executor is not None and not os.path.exists(simulation_path):
                _run_parallel_simulation(
                    executor,
                    workers,
                    parameter_path,
                    fn_catalog,
                    shape_coords,
                    m_max,
                    induced_info,
                    simulation_path,
                    duration,
                    n_simulations,
                    mc,
                )
            else:
                # Serial path; also resumes a partially written existing file,
                # which the part-file scheme cannot do.
                simulation = ETASSimulation(
                    calculation,
                    m_max=m_max,
                    approx_times=True,
                    induced_info=induced_info,
                )
                simulation.prepare()
                simulation.simulate_to_csv(
                    simulation_path,
                    duration,
                    n_simulations,
                    m_threshold=mc,
                )
            simulation_paths[duration] = simulation_path
    finally:
        if executor is not None:
            executor.shutdown()
    return simulation_paths


def _minimum_positive_spacing(values: pd.Series) -> float:
    unique_values = np.sort(values.drop_duplicates().to_numpy())
    diffs = np.diff(unique_values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    return float(diffs.min())


def build_background_induced_info(
    calculation: ETASParameterCalculation,
    background_grid: pd.DataFrame | None,
    duration_days: int,
) -> list | None:
    if background_grid is None:
        return None

    log10_iota = calculation.theta.get("log10_iota")
    if log10_iota is None or not np.isfinite(float(log10_iota)):
        return None

    rates = background_grid["rate"].clip(lower=0)
    max_rate = float(rates.max())
    if max_rate <= 0:
        return None

    expected_count = float(np.power(10, float(log10_iota)) * calculation.area * duration_days)
    if expected_count <= 0:
        return None

    return [
        background_grid["latitude"].reset_index(drop=True),
        background_grid["longitude"].reset_index(drop=True),
        (rates / max_rate).reset_index(drop=True),
        _minimum_positive_spacing(background_grid["latitude"]),
        _minimum_positive_spacing(background_grid["longitude"]),
        expected_count,
    ]


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
                    "duration_days": duration,
                    "forecast_start": forecast_start,
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
    background_metadata: dict,
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
        **background_metadata,
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
    args.background_rate_file = resolve_path(args.background_rate_file)
    ensure_inputs_exist(args.background_rate_file)

    forecast_start = parse_datetime(args.forecast_start)
    durations = parse_durations(args.durations)
    output_paths = build_run_paths(args.experiment_name, forecast_start)

    logger.info("Starting NZ-wide ETAS forecast run: %s", output_paths["run_label"])
    logger.info("Forecast origin: %s", forecast_start)
    logger.info("Durations: %s days", durations)
    logger.info("Simulations per duration: %s", args.n_simulations)
    if args.background_rate_file:
        logger.info(
            "Using background-rate grid %s at M%.1f",
            args.background_rate_file,
            args.background_rate_mag,
        )

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
    catalog_path, background_grid, background_metadata = prepare_background_rate_catalog(
        catalog,
        output_paths["output_dir"],
        output_paths["run_label"],
        args.background_rate_file,
        args.background_rate_mag,
    )
    config = build_inversion_config(
        output_paths["run_label"],
        catalog_path,
        forecast_start,
        args.auxiliary_start,
        args.timewindow_start,
        args.mc,
        initial_theta,
        bg_term=(
            BACKGROUND_RATE_COLUMN
            if args.background_rate_file is not None
            else None
        ),
    )
    calculation, parameter_path = load_or_run_inversion(
        config,
        output_paths["output_dir"],
        force_reinvert=args.force_reinvert,
    )

    guard_against_degenerate_inversion(calculation, args.allow_degenerate_inversion)

    simulation_paths = run_simulations(
        calculation,
        durations,
        args.n_simulations,
        args.mc,
        args.m_max,
        output_paths["simulation_dir"],
        background_grid=background_grid,
        overwrite=args.force_resimulate or args.force_reinvert,
        simulation_workers=resolve_simulation_workers(
            args.simulation_workers, args.n_simulations
        ),
        parameter_path=parameter_path,
        fn_catalog=config["fn_catalog"],
        shape_coords=config["shape_coords"],
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
        background_metadata,
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
