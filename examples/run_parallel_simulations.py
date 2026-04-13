"""
Parallel ETAS Simulations for New Zealand (Kaikoura & Canterbury Sequences)

This workflow now produces two forecast arms:
1. Adaptive: re-estimate ETAS parameters at every forecast date.
2. Fixed regional: fit one regional model per sequence, then keep those
   parameters fixed while reconditioning on the evolving catalog history.
"""

import numpy as np
import pandas as pd
import json
import logging
import datetime as dt
import sys
import os
import gc
import warnings
from joblib import Parallel, delayed

from date_grids import (
    CANTERBURY_DATES,
    KAIKOURA_DATES,
    SEQUENCE_DATE_GRID_METADATA,
)

# --- PARALLEL CONFIGURATION ---
# Memory-aware worker limits to prevent OOM
def get_safe_n_jobs(per_worker_gb=1.5):
    """Calculate safe number of parallel jobs based on available RAM."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / (1024**3)
        return max(1, int(ram_gb / per_worker_gb))
    except ImportError:
        # Fallback if psutil not available
        return 10

# Inversions are memory-heavy (~14 GB per worker based on profiling)
N_JOBS_INVERSION = min(12, get_safe_n_jobs(per_worker_gb=14.0))

# Simulations are lighter (~2 GB per worker)
N_JOBS_SIMULATION = min(12, get_safe_n_jobs(per_worker_gb=2.0))

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation, parameter_dict2array
from etas.simulation import ETASSimulation

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Rounding issues.*")
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# --- Pre-compile Numba JIT functions before parallel execution ---
# This prevents race conditions where workers try to compile the same cached 
# functions simultaneously, which can cause SIGKILL crashes
def _precompile_jit_functions():
    """Trigger Numba compilation in main process before spawning workers."""
    try:
        import numpy as np
        from etas.inversion import _triggering_kernel_core, _neg_log_likelihood_core
        
        # Dummy arrays to trigger compilation
        n = 10
        dummy_arr = np.ones(n, dtype=np.float64)
        dummy_bool_arr = np.zeros(n, dtype=np.float64)
        
        # Compile _triggering_kernel_core
        _triggering_kernel_core(
            dummy_arr, dummy_arr, dummy_arr, dummy_arr, False,
            1.0, 1.0, 0.01, 0.1, 100.0, 0.1, 0.3, 0.8, 3.0
        )
        
        # Compile _neg_log_likelihood_core
        _neg_log_likelihood_core(
            dummy_arr, dummy_arr, dummy_arr, dummy_arr, dummy_arr,
            1.0, 1.0, 1.0, -2.0, 0.1, 3.0, -2.0, 0.3, 0.8, 3.0
        )
        print("Numba JIT functions pre-compiled successfully")
    except Exception as e:
        print(f"JIT pre-compilation warning (non-fatal): {e}")

_precompile_jit_functions()
set_up_logger(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION (New Zealand Data) ---

# Load NZ catalog (downloaded from GeoNet)
fn_catalog = "../input_data/nzcat.csv"
nzcat = pd.read_csv(fn_catalog, index_col=0, parse_dates=["time"])
nzcat.sort_values(by="time", inplace=True)

# Forecast-origin schedules are generated from explicit rules in date_grids.py
dates = KAIKOURA_DATES
dates_c = CANTERBURY_DATES

# NZ Inversion Config (from user's notebook)
inversion_config_base = {
    "fn_catalog": fn_catalog,
    "auxiliary_start": "1950-01-01 00:00:00",
    "timewindow_start": "1960-01-01 00:00:00",
    # "timewindow_end" will be set dynamically
    "theta_0": {
        "log10_mu": -7.477863177977867, 
        "log10_iota": None, 
        "log10_k0": -0.8570602601363014, 
        "a": 1.4333791204125566, 
        "log10_c": -3.1859152978148644, 
        "omega": -0.08102742585588284, 
        "log10_tau": 4.038107413059718, 
        "log10_d": 1.588041892797509, 
        "gamma": 0.34307084228763013, 
        "rho": 0.8062060642600785
    },
    "mc": 4.1,
    "m_ref": 4.1,
    "delta_m": 0.1,
    "coppersmith_multiplier": 100,
    "shape_coords": "../input_data/nz_polygon.npy",
    "name": "nz_standard"
}

EVALUATION_HORIZONS_DAYS = [1.0, 3.0, 7.0, 30.0]
PRIMARY_EVALUATION_HORIZON_DAYS = 7.0
SIMULATION_HORIZON_DAYS = max(EVALUATION_HORIZONS_DAYS)
SIMULATION_TAG = f"{int(SIMULATION_HORIZON_DAYS)}d"
MAINSHOCK_LOOKBACK_DAYS = 2.0

ADAPTIVE_PARAMETER_DIR = "./output_nz/"
ADAPTIVE_SIMULATION_DIR = f"./simulations_nz_{SIMULATION_TAG}/"
FIXED_PARAMETER_DIR = "./output_nz_fixed/"
FIXED_SIMULATION_DIR = f"./simulations_nz_fixed_{SIMULATION_TAG}/"
EXPERIMENT_METADATA_PATH = os.path.join(
    ADAPTIVE_PARAMETER_DIR, f"experiment_config_{SIMULATION_TAG}.json"
)

# Ensure output directories exist
for output_dir in [
    ADAPTIVE_PARAMETER_DIR,
    ADAPTIVE_SIMULATION_DIR,
    FIXED_PARAMETER_DIR,
    FIXED_SIMULATION_DIR,
]:
    os.makedirs(output_dir, exist_ok=True)

all_dates = [*dates, *dates_c]
SEQUENCE_DATES = {
    "Kaikoura": dates,
    "Canterbury": dates_c,
}


def get_sequence_and_index(forecast_date):
    """Return sequence name and index for a configured forecast date."""
    if forecast_date in dates:
        return "Kaikoura", dates.index(forecast_date)
    return "Canterbury", dates_c.index(forecast_date)


def build_inversion_config(timewindow_end, model_id, theta_0=None):
    """Create a fresh inversion config for a specific forecast date/model."""
    config = inversion_config_base.copy()
    config["timewindow_end"] = timewindow_end.strftime("%Y-%m-%d %H:%M:%S")
    config["id"] = model_id
    if theta_0 is None:
        config["theta_0"] = inversion_config_base["theta_0"].copy()
    else:
        config["theta_0"] = theta_0.copy()
    return config


def infer_sequence_mainshock(sequence, lookback_days=MAINSHOCK_LOOKBACK_DAYS):
    """Infer the sequence mainshock from the catalog just before the first forecast."""
    first_forecast_date = SEQUENCE_DATES[sequence][0]
    lookback_start = first_forecast_date - dt.timedelta(days=lookback_days)
    prior_events = nzcat[
        (nzcat["time"] >= lookback_start)
        & (nzcat["time"] < first_forecast_date)
    ].copy()
    if len(prior_events) == 0:
        raise ValueError(
            f"No candidate mainshock found for {sequence} in the {lookback_days}-day lookback window."
        )

    mainshock = prior_events.sort_values(
        by=["magnitude", "time"], ascending=[False, True]
    ).iloc[0]
    mainshock_time = pd.Timestamp(mainshock["time"]).to_pydatetime()
    baseline_end = mainshock_time - dt.timedelta(seconds=1)
    return {
        "mainshock_time": mainshock_time,
        "mainshock_magnitude": float(mainshock["magnitude"]),
        "baseline_end": baseline_end,
    }


SEQUENCE_BASELINE_INFO = {
    sequence: infer_sequence_mainshock(sequence) for sequence in SEQUENCE_DATES
}


def write_experiment_metadata():
    """Persist the experiment configuration so downstream analysis can reuse it."""
    metadata = {
        "evaluation_horizons_days": EVALUATION_HORIZONS_DAYS,
        "primary_evaluation_horizon_days": PRIMARY_EVALUATION_HORIZON_DAYS,
        "simulation_horizon_days": SIMULATION_HORIZON_DAYS,
        "simulation_tag": SIMULATION_TAG,
        "adaptive_parameter_dir": ADAPTIVE_PARAMETER_DIR,
        "adaptive_simulation_dir": ADAPTIVE_SIMULATION_DIR,
        "fixed_parameter_dir": FIXED_PARAMETER_DIR,
        "fixed_simulation_dir": FIXED_SIMULATION_DIR,
        "sequence_date_grids": SEQUENCE_DATE_GRID_METADATA,
        "sequence_baselines": {
            sequence: {
                "mainshock_time": info["mainshock_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "mainshock_magnitude": info["mainshock_magnitude"],
                "baseline_end": info["baseline_end"].strftime("%Y-%m-%d %H:%M:%S"),
            }
            for sequence, info in SEQUENCE_BASELINE_INFO.items()
        },
    }
    with open(EXPERIMENT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote experiment metadata to {EXPERIMENT_METADATA_PATH}")


def run_inversion(forecast_date):
    """Run inversion for a single date. Designed for parallel execution."""
    sequence, index = get_sequence_and_index(forecast_date)
    model_id = "nz_{}_{}".format(sequence, index)
    config = build_inversion_config(forecast_date, model_id)

    # Check if parameters already exist to save time
    param_file = os.path.join(ADAPTIVE_PARAMETER_DIR, f"parameters_{model_id}.json")
    if os.path.exists(param_file):
        logger.info(f"Parameters for {model_id} already exist, skipping inversion.")
        return f"Skipped {model_id}"
    
    logger.info(f"Inverting for {model_id}...")
    try:
        calculation = ETASParameterCalculation(config)
        calculation.prepare()
        calculation.invert()
        calculation.store_results(ADAPTIVE_PARAMETER_DIR, store_pij=True)
        result = f"Done {model_id}"
    except Exception as e:
        logger.error(f"Failed inversion for {model_id}: {e}")
        result = f"Failed {model_id}"
    finally:
        # Force garbage collection to free memory for next worker
        if 'calculation' in dir():
            del calculation
        gc.collect()
    return result


def run_regional_baseline_inversion(sequence):
    """Fit one regional model at the first forecast date for a sequence."""
    regional_model_id = f"nz_{sequence}_fixed"
    param_file = os.path.join(
        FIXED_PARAMETER_DIR, f"parameters_{regional_model_id}.json"
    )
    if os.path.exists(param_file):
        logger.info(
            f"Regional baseline for {sequence} already exists, skipping inversion."
        )
        return f"Skipped {regional_model_id}"

    baseline_info = SEQUENCE_BASELINE_INFO[sequence]
    baseline_date = baseline_info["baseline_end"]
    fixed_start = inversion_config_base["theta_0"].copy()
    config = build_inversion_config(baseline_date, regional_model_id, theta_0=fixed_start)

    logger.info(
        "Running regional baseline inversion for "
        f"{sequence} before mainshock {baseline_info['mainshock_time']} "
        f"(M{baseline_info['mainshock_magnitude']:.1f}); "
        f"baseline window ends at {baseline_date}."
    )
    try:
        calculation = ETASParameterCalculation(config)
        calculation.prepare()
        calculation.invert()
        calculation.store_results(FIXED_PARAMETER_DIR, store_pij=False)
        result = f"Done {regional_model_id}"
    except Exception as exc:
        logger.error(f"Failed regional baseline inversion for {sequence}: {exc}")
        result = f"Failed {regional_model_id}"
    finally:
        if "calculation" in locals():
            del calculation
        gc.collect()
    return result


def run_adaptive_inversions():
    """Estimate one adaptive model per configured forecast window."""
    logger.info("Running adaptive inversions to generate per-date parameters...")
    logger.info(
        f"Running {len(all_dates)} inversions in parallel with {N_JOBS_INVERSION} workers..."
    )
    inversion_results = Parallel(n_jobs=N_JOBS_INVERSION)(
        delayed(run_inversion)(forecast_date) for forecast_date in all_dates
    )
    logger.info("Adaptive inversions complete!")
    for result in inversion_results:
        logger.info(result)


def run_regional_baseline_inversions():
    """Estimate one fixed regional baseline per earthquake sequence."""
    logger.info("Running fixed regional baseline inversions...")
    results = [
        run_regional_baseline_inversion("Kaikoura"),
        run_regional_baseline_inversion("Canterbury"),
    ]
    logger.info("Fixed regional baseline inversions complete!")
    for result in results:
        logger.info(result)


# --- SIMULATION STEP ---

n_files = 10  # Number of parallel file chunks per model
n_simulations_overall = 10000  # Total simulations per model
forecast_period = SIMULATION_HORIZON_DAYS  # days

def read_combination(run_number):
    """Map run_number to date/model and file_index."""
    date_index = run_number // n_files
    file_no = run_number % n_files
    
    if date_index < len(dates):
        sequence = "Kaikoura"
        model_id = "nz_Kaikoura_{}".format(date_index)
        timewindow_end = dates[date_index]
    else:
        idx_c = date_index - len(dates)
        sequence = "Canterbury"
        model_id = "nz_Canterbury_{}".format(idx_c)
        timewindow_end = dates_c[idx_c]
        
    return sequence, timewindow_end, file_no, model_id


def run_adaptive_simulation(run_number):
    sequence, timewindow_end, file_no, model_id = read_combination(run_number)
    logger.info(f"Simulating run {run_number}: {model_id} file {file_no}")

    n_simulations = int(n_simulations_overall / n_files)

    fn_parameters = os.path.join(ADAPTIVE_PARAMETER_DIR, f"parameters_{model_id}.json")
    fn_store_simulation = os.path.join(
        ADAPTIVE_SIMULATION_DIR, f"sim_{model_id}_{file_no}.csv"
    )

    # Check if simulation file already exists
    if os.path.exists(fn_store_simulation):
        logger.info(f"Simulation {fn_store_simulation} already exists, skipping.")
        return f"Skipped {fn_store_simulation}"
    
    # Check if parameters exist
    if not os.path.exists(fn_parameters):
        logger.warning(f"Parameters not found: {fn_parameters}")
        return f"Failed {fn_store_simulation}"

    with open(fn_parameters, "r") as f:
        inversion_output = json.load(f)

    # Fix paths in loaded config
    inversion_output["fn_catalog"] = fn_catalog
    inversion_output["shape_coords"] = inversion_config_base["shape_coords"]

    etas_reload = ETASParameterCalculation.load_calculation(inversion_output)
    
    # Use approx_times=True for faster simulation (5-10x speedup)
    simulation = ETASSimulation(etas_reload, approx_times=True)
    simulation.prepare()
    
    # Update timewindow_end for this specific forecast date
    # Note: Do NOT overwrite simulation.catalog - prepare() already set it up 
    # correctly with source_events linkage
    simulation.inversion_params.timewindow_end = timewindow_end
    
    try:
        simulation.simulate_to_csv(
            fn_store_simulation, forecast_period, n_simulations, 
            m_threshold=inversion_config_base["mc"]
        )
        result = f"Done {fn_store_simulation}"
    finally:
        # Force garbage collection to free memory for next worker
        del simulation, etas_reload
        gc.collect()
    return result


def build_fixed_regional_conditioning(
    forecast_date, model_id, fixed_parameters, fixed_beta
):
    """Prepare a per-date catalog conditioning using frozen regional theta."""
    config = build_inversion_config(forecast_date, model_id, theta_0=fixed_parameters)
    config["beta"] = fixed_beta
    calculation = ETASParameterCalculation(config)
    calculation.prepare()
    calculation.theta = fixed_parameters
    theta_array = parameter_dict2array(fixed_parameters)
    (
        calculation.pij,
        calculation.target_events,
        calculation.source_events,
        calculation.n_hat,
        calculation.i_hat,
    ) = calculation.expectation_step(
        theta_array, calculation.m_ref - calculation.delta_m / 2
    )
    calculation.inversion_done = True
    calculation.i = 0
    return calculation


def load_fixed_regional_metadata(sequence):
    """Load the regional baseline parameters and beta for a sequence."""
    regional_model_id = f"nz_{sequence}_fixed"
    fn_parameters = os.path.join(
        FIXED_PARAMETER_DIR, f"parameters_{regional_model_id}.json"
    )
    if not os.path.exists(fn_parameters):
        raise FileNotFoundError(fn_parameters)

    with open(fn_parameters, "r") as f:
        inversion_output = json.load(f)
    return {
        "final_parameters": inversion_output["final_parameters"],
        "beta": inversion_output["beta"],
    }


def run_fixed_regional_simulation(run_number):
    """Simulate a forecast window with fixed regional parameters."""
    sequence, timewindow_end, file_no, model_id = read_combination(run_number)
    fixed_model_id = f"nz_fixed_{sequence}_{model_id.split('_')[-1]}"
    logger.info(
        f"Simulating fixed regional run {run_number}: {fixed_model_id} file {file_no}"
    )

    fn_store_simulation = os.path.join(
        FIXED_SIMULATION_DIR,
        f"sim_nz_fixed_{sequence}_{model_id.split('_')[-1]}_{file_no}.csv",
    )
    if os.path.exists(fn_store_simulation):
        logger.info(f"Simulation {fn_store_simulation} already exists, skipping.")
        return f"Skipped {fn_store_simulation}"

    n_simulations = int(n_simulations_overall / n_files)

    try:
        fixed_metadata = load_fixed_regional_metadata(sequence)
    except FileNotFoundError:
        logger.warning(
            f"Regional baseline parameters not found for {sequence} in {FIXED_PARAMETER_DIR}"
        )
        return f"Failed {fn_store_simulation}"

    try:
        conditioned_calc = build_fixed_regional_conditioning(
            timewindow_end,
            fixed_model_id,
            fixed_metadata["final_parameters"],
            fixed_metadata["beta"],
        )
        simulation = ETASSimulation(conditioned_calc, approx_times=True)
        simulation.prepare()
        simulation.inversion_params.timewindow_end = timewindow_end
        simulation.simulate_to_csv(
            fn_store_simulation,
            forecast_period,
            n_simulations,
            m_threshold=inversion_config_base["mc"],
        )
        result = f"Done {fn_store_simulation}"
    finally:
        if "simulation" in locals():
            del simulation
        if "conditioned_calc" in locals():
            del conditioned_calc
        gc.collect()
    return result


def run_adaptive_simulations():
    """Generate adaptive forecasts for every configured date."""
    total_runs = n_files * len(all_dates)
    logger.info("Starting adaptive simulations for NZ...")
    logger.info(f"Total adaptive runs: {total_runs}")
    logger.info(f"Using {N_JOBS_SIMULATION} workers for adaptive simulations...")
    results = Parallel(n_jobs=N_JOBS_SIMULATION)(
        delayed(run_adaptive_simulation)(i) for i in range(total_runs)
    )
    logger.info("Adaptive simulations complete!")
    for result in results[:10]:
        logger.info(result)


def run_fixed_regional_simulations():
    """Generate fixed-parameter regional forecasts for every configured date."""
    total_runs = n_files * len(all_dates)
    logger.info("Starting fixed regional simulations for NZ...")
    logger.info(f"Total fixed regional runs: {total_runs}")
    logger.info(
        f"Using {N_JOBS_SIMULATION} workers for fixed regional simulations..."
    )
    results = Parallel(n_jobs=N_JOBS_SIMULATION)(
        delayed(run_fixed_regional_simulation)(i) for i in range(total_runs)
    )
    logger.info("Fixed regional simulations complete!")
    for result in results[:10]:
        logger.info(result)


def main():
    logger.info("Starting Parallel Simulations for NZ...")
    logger.info(
        "Workflow arms: adaptive re-estimation and fixed regional baseline."
    )
    for sequence, grid_meta in SEQUENCE_DATE_GRID_METADATA.items():
        logger.info(
            f"{sequence} forecast grid: {grid_meta['n_forecast_origins']} origins "
            f"from {grid_meta['first_forecast_origin']} to "
            f"{grid_meta['last_forecast_origin']}."
        )
    logger.info(
        "Evaluation horizons: "
        f"{EVALUATION_HORIZONS_DAYS} days "
        f"(primary={PRIMARY_EVALUATION_HORIZON_DAYS}, simulated={SIMULATION_HORIZON_DAYS})."
    )
    write_experiment_metadata()
    run_adaptive_inversions()
    run_regional_baseline_inversions()
    run_adaptive_simulations()
    run_fixed_regional_simulations()
    logger.info("NZ parallel workflow complete.")


if __name__ == "__main__":
    main()
