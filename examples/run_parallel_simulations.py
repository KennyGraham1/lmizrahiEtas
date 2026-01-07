"""
Parallel ETAS Simulations for New Zealand (Kaikoura & Canterbury Sequences)

Based on the user's original notebook code, adapted for the downloaded GeoNet data.
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
N_JOBS_SIMULATION = min(40, get_safe_n_jobs(per_worker_gb=2.0))

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation

warnings.filterwarnings("ignore")
set_up_logger(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION (New Zealand Data) ---

# Load NZ catalog (downloaded from GeoNet)
fn_catalog = "../input_data/nzcat.csv"
nzcat = pd.read_csv(fn_catalog, index_col=0, parse_dates=["time"])
nzcat.sort_values(by="time", inplace=True)

# Kaikoura sequence dates (from user's notebook)
dates = [
    dt.datetime(2016, 11, 13, 12, 0, 0),
    dt.datetime(2016, 11, 14, 12, 0, 0),
    dt.datetime(2016, 11, 15, 12, 0, 0),
    dt.datetime(2016, 11, 16, 0, 0, 0),
    dt.datetime(2016, 11, 16, 12, 0, 0),
    dt.datetime(2016, 11, 17, 0, 0, 0),
    dt.datetime(2016, 11, 17, 12, 0, 0),
    dt.datetime(2016, 11, 18, 0, 0, 0),
    dt.datetime(2016, 11, 18, 12, 0, 0),
    dt.datetime(2016, 11, 19, 0, 0, 0),
    dt.datetime(2016, 11, 19, 12, 0, 0),
    dt.datetime(2016, 11, 20, 12, 0, 0),
    dt.datetime(2016, 11, 21, 12, 0, 0),
    dt.datetime(2016, 11, 22, 12, 0, 0),
    dt.datetime(2016, 11, 23, 12, 0, 0),
    dt.datetime(2016, 11, 24, 12, 0, 0),
    dt.datetime(2016, 11, 25, 12, 0, 0),
    dt.datetime(2016, 11, 26, 12, 0, 0),
    dt.datetime(2016, 11, 27, 12, 0, 0),
    dt.datetime(2016, 11, 28, 12, 0, 0),
    dt.datetime(2016, 11, 29, 12, 0, 0),
    dt.datetime(2016, 11, 30, 12, 0, 0),
    dt.datetime(2016, 12, 1, 12, 0, 0),
    dt.datetime(2016, 12, 2, 12, 0, 0),
    dt.datetime(2016, 12, 7, 12, 0, 0),
    dt.datetime(2016, 12, 11, 12, 0, 0),
    dt.datetime(2016, 12, 14, 12, 0, 0),
    dt.datetime(2016, 12, 18, 12, 0, 0),
    dt.datetime(2016, 12, 21, 12, 0, 0),
    dt.datetime(2016, 12, 25, 12, 0, 0),
    dt.datetime(2016, 12, 28, 12, 0, 0),
    dt.datetime(2017, 1, 1, 12, 0, 0),
    dt.datetime(2017, 1, 8, 12, 0, 0),
    dt.datetime(2017, 1, 15, 12, 0, 0),
    dt.datetime(2017, 1, 22, 12, 0, 0),
    dt.datetime(2017, 1, 29, 12, 0, 0),
    dt.datetime(2017, 2, 5, 12, 0, 0),
    dt.datetime(2017, 2, 12, 12, 0, 0),
    dt.datetime(2017, 2, 19, 12, 0, 0),
    dt.datetime(2017, 2, 26, 12, 0, 0),
    dt.datetime(2017, 3, 5, 12, 0, 0),
    dt.datetime(2017, 3, 12, 12, 0, 0),
    dt.datetime(2017, 3, 19, 12, 0, 0),
    dt.datetime(2017, 3, 26, 12, 0, 0),
    dt.datetime(2017, 4, 2, 12, 0, 0),
]

# Canterbury sequence dates
dates_c = [
    dt.datetime(2010, 9, 3, 17, 0, 0),
    dt.datetime(2010, 9, 4, 17, 0, 0),
    dt.datetime(2010, 9, 5, 17, 0, 0),
    dt.datetime(2010, 9, 6, 17, 0, 0),
    dt.datetime(2010, 9, 7, 17, 0, 0),
    dt.datetime(2010, 9, 8, 17, 0, 0),
    dt.datetime(2010, 9, 9, 17, 0, 0),
    dt.datetime(2010, 9, 10, 17, 0, 0),
    dt.datetime(2010, 9, 11, 17, 0, 0),
    dt.datetime(2010, 9, 12, 17, 0, 0),
    dt.datetime(2010, 9, 13, 17, 0, 0),
    dt.datetime(2010, 9, 14, 17, 0, 0),
    dt.datetime(2010, 9, 15, 17, 0, 0),
    dt.datetime(2010, 9, 16, 17, 0, 0),
]

# NZ Inversion Config (from user's notebook)
inversion_config_base = {
    "fn_catalog": fn_catalog,
    "data_path": "./output_nz/",
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

# Ensure output directories exist
os.makedirs(inversion_config_base["data_path"], exist_ok=True)
os.makedirs("simulations_nz", exist_ok=True)

# --- INVERSION STEP ---

logger.info("Running initial inversions to generate parameters...")

all_dates = [*dates, *dates_c]


def run_inversion(date):
    """Run inversion for a single date. Designed for parallel execution."""
    config = inversion_config_base.copy()
    # Deep copy theta_0
    config["theta_0"] = inversion_config_base["theta_0"].copy()
    config['timewindow_end'] = date.strftime('%Y-%m-%d %H:%M:%S')
    
    sequence = "Kaikoura" if date in dates else "Canterbury"
    index = dates.index(date) if date in dates else dates_c.index(date)
    model_id = "nz_{}_{}".format(sequence, index)
    config['id'] = model_id

    # Check if parameters already exist to save time
    param_file = os.path.join(config["data_path"], f"parameters_{model_id}.json")
    if os.path.exists(param_file):
        logger.info(f"Parameters for {model_id} already exist, skipping inversion.")
        return f"Skipped {model_id}"
    
    logger.info(f"Inverting for {model_id}...")
    try:
        calculation = ETASParameterCalculation(config)
        calculation.prepare()
        calculation.invert()
        calculation.store_results(config['data_path'], store_pij=True)
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


# Run inversions in parallel (uses memory-aware worker count)
logger.info(f"Running {len(all_dates)} inversions in parallel with {N_JOBS_INVERSION} workers...")
inversion_results = Parallel(n_jobs=N_JOBS_INVERSION)(delayed(run_inversion)(date) for date in all_dates)
logger.info("Inversions complete!")
for r in inversion_results:
    logger.info(r)


# --- SIMULATION STEP ---

n_files = 10  # Number of parallel file chunks per model
n_simulations_overall = 10000  # Total simulations per model
forecast_period = 7.0  # days

def read_combination(run_number):
    """Map run_number to date/model and file_index."""
    date_index = run_number // n_files
    file_no = run_number % n_files
    
    if date_index < len(dates):
        model_id = "nz_Kaikoura_{}".format(date_index)
        timewindow_end = dates[date_index]
    else:
        idx_c = date_index - len(dates)
        model_id = "nz_Canterbury_{}".format(idx_c)
        timewindow_end = dates_c[idx_c]
        
    return timewindow_end, file_no, model_id

def run_simulation(run_number):
    timewindow_end, file_no, model_id = read_combination(run_number)
    logger.info(f"Simulating run {run_number}: {model_id} file {file_no}")

    parameter_path = inversion_config_base["data_path"]
    store_path = "./simulations_nz/"
    
    n_simulations = int(n_simulations_overall / n_files)

    fn_parameters = os.path.join(parameter_path, f"parameters_{model_id}.json")
    fn_store_simulation = os.path.join(store_path, f"sim_{model_id}_{file_no}.csv")

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
    
    # Update catalog for this specific timewindow
    simulation.inversion_params.timewindow_end = timewindow_end
    simulation.catalog = nzcat.query("time < @timewindow_end").copy()
    simulation.catalog["xi_plus_1"] = 1.0
    
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


if __name__ == "__main__":
    logger.info("Starting Parallel Simulations for NZ...")
    
    total_runs = n_files * len(all_dates)
    logger.info(f"Total runs: {total_runs} (dates: {len(all_dates)}, files: {n_files})")
    
    # Run in parallel (uses memory-aware worker count)
    logger.info(f"Using {N_JOBS_SIMULATION} workers for simulations...")
    results = Parallel(n_jobs=N_JOBS_SIMULATION)(delayed(run_simulation)(i) for i in range(total_runs))
    
    logger.info("Parallel simulations complete!")
    for r in results[:10]:
        logger.info(r)
