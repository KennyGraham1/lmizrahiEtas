"""
4-Stage ETAS Pipeline Template
==============================

This script demonstrates a standard modular pipeline for Earthquake Forecasting:
1. DATA: Prepare and standardize catalog data.
2. ESTIMATE: Invert for ETAS parameters (or load custom model parameters).
3. SIMULATE: Generate synthetic catalogs (Swap this for different models!).
4. EVALUATE: Validate using standard CSEP-style tests.

Usage:
    python pipeline_template.py
"""

import os
import sys

# Set matplotlib backend to Agg (non-interactive) to prevent threading crashes with Tkinter
import matplotlib
matplotlib.use('Agg')

import json
import logging
import pandas as pd
import numpy as np
import datetime as dt
import warnings
from joblib import Parallel, delayed

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add SeismoStats submodule to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SeismoStats')))


from etas import set_up_logger
from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation
from visualize_results import plot_csep_6panel

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
EXPERIMENT_NAME = "pipeline_example_kaikoura"
OUTPUT_DIR = f"output_{EXPERIMENT_NAME}"
SIMULATION_DIR = f"simulations_{EXPERIMENT_NAME}"
FIGURE_dIR = "figures"

# Path definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
INPUT_DATA_DIR = os.path.join(ROOT_DIR, "input_data")

# User Settings
# User Settings
FORECAST_START = dt.datetime(2016, 11, 14, 0, 0, 0)  # Just after Kaikoura mainshock
FORECAST_DURATIONS = [30, 90, 365]  # Days
N_SIMULATIONS = 1000  # Number of synthetic catalogs
N_CORES = 4  # Parallel workers

# Model Constraints
MC = 4.1  # Magnitude of completeness
LAT_RANGE = [-48.0, -34.0]
LON_RANGE = [164.0, 180.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SIMULATION_DIR, exist_ok=True)
os.makedirs(FIGURE_dIR, exist_ok=True)

set_up_logger(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# STAGE 1: PREPARE DATA
# =============================================================================
def stage_1_prepare_data():
    """
    Load raw catalog and convert to standard format.
    Expected Output: DataFrame with columns [latitude, longitude, time, magnitude, depth]
    """
    logger.info("STAGE 1: Preparing Data (NZ GeoNet Catalog)...")
    
    # In a real scenario, you might download this from GeoNet/USGS
    fn_catalog = os.path.join(INPUT_DATA_DIR, "nzcat.csv")
    if not os.path.exists(fn_catalog):
        logger.error(f"Catalog file not found: {fn_catalog}")
        sys.exit(1)
        
    catalog = pd.read_csv(fn_catalog, index_col=0, parse_dates=["time"])
    catalog.sort_values(by="time", inplace=True)
    
    # Filter to region of interest (optional but recommended)
    mask = (
        (catalog["latitude"] >= LAT_RANGE[0]) & 
        (catalog["latitude"] <= LAT_RANGE[1]) & 
        (catalog["longitude"] >= LON_RANGE[0]) & 
        (catalog["longitude"] <= LON_RANGE[1])
    )
    catalog = catalog[mask]
    
    # Split into TRAINING (Historical) and TESTING (observed future)
    training_cat = catalog[catalog["time"] < FORECAST_START]
    
    # Get all potential testing data (up to max duration)
    max_duration = max(FORECAST_DURATIONS)
    testing_cat = catalog[(catalog["time"] >= FORECAST_START) & 
                          (catalog["time"] < FORECAST_START + dt.timedelta(days=max_duration)) &
                          (catalog["magnitude"] >= MC)]
                          
    logger.info(f"Training events: {len(training_cat)}")
    logger.info(f"Full Testing/Observed events (max {max_duration} days): {len(testing_cat)}")
    
    return training_cat, testing_cat

# =============================================================================
# STAGE 2: ESTIMATE PARAMETERS
# =============================================================================
def stage_2_estimate_parameters(training_catalog):
    """
    Calculate model parameters based on training history.
    For standard ETAS, this runs the inversion.
    """
    logger.info("STAGE 2: Estimating ETAS Parameters...")
    
    # For this example, we use a fixed set of parameters (e.g. from previous calibration)
    # to keep the example fast. In a real run, you would call ETASParameterCalculation.
    
    # Example parameters (Kaikoura region approximated)
    params = {
        "log10_mu": -7.5,
        "log10_k0": -2.4,
        "a": 1.7,
        "log10_c": -2.8,
        "omega": -0.05,
        "log10_tau": 3.8,
        "log10_d": 0.8,
        "gamma": 1.2,
        "rho": 0.75,
        "mc": MC
    }
    
    # Save parameters for reference
    param_file = os.path.join(OUTPUT_DIR, "model_parameters.json")
    with open(param_file, "w") as f:
        json.dump(params, f, indent=4)
        
    logger.info(f"Parameters saved to {param_file}")
    return params

# =============================================================================
# STAGE 3: SIMULATE FORECASTS
# =============================================================================
def prepare_etas_model(training_catalog, parameters):
    """
    Initialize and prepare the ETAS model object. 
    This is computationally expensive (distance matrix), so we do it once.
    """
    logger.info("Initializing ETAS model (calculating distances, etc.)...")
    
    # Standard ETAS setup
    config = {
        "fn_catalog": os.path.join(INPUT_DATA_DIR, "nzcat.csv"), 
        "auxiliary_start": training_catalog["time"].min(),
        "timewindow_start": training_catalog["time"].min(),
        "timewindow_end": FORECAST_START,
        "mc": MC,
        "shape_coords": os.path.join(INPUT_DATA_DIR, "nz_polygon.npy"),
        "delta_m": 0.1,
        "coppersmith_multiplier": 100,
        "theta_0": {k:v for k,v in parameters.items() if k != "mc"}
    }
    
    calc = ETASParameterCalculation(config)
    calc.catalog = training_catalog
    calc.prepare()

    # Hack: Manually set P_background as we skipped inversion
    if "P_background" not in calc.target_events.columns:
        calc.target_events["P_background"] = 0.1
    # Also ensure zeta_plus_1 is present
    if "zeta_plus_1" not in calc.target_events.columns:
        calc.target_events["zeta_plus_1"] = 1.0
        
    # CRITICAL: Set theta (parameters) manually since we skipped inversion
    calc.theta = parameters
        
    return calc

def _run_single_simulation(sim_id, calc, duration):
    """
    Worker function for a single simulation.
    """
    # Initialize simulation
    sim = ETASSimulation(calc, approx_times=True)
    sim.prepare()
    
    # Run simulation
    # simulate_to_df returns a ForecastCatalog object, which IS a DataFrame
    forecast_cat = sim.simulate_to_df(duration, n_simulations=1)
    sim_data = forecast_cat
    
    # Add ID
    sim_data["catalog_id"] = sim_id
    
    return sim_data

def stage_3_simulate_forecasts(calc, duration):
    """
    Generate multiple synthetic future catalogs using the prepared model.
    """
    logger.info(f"STAGE 3: Simulating {N_SIMULATIONS} forecasts for {duration} days...")
    
    # Run in parallel
    results = Parallel(n_jobs=N_CORES, backend="threading")(
        delayed(_run_single_simulation)(
            i, calc, duration
        ) for i in range(N_SIMULATIONS)
    )
    
    # Combine all forecasts into one DataFrame
    all_simulations = pd.concat(results, ignore_index=True)
    
    # Save raw simulation data
    sim_file = os.path.join(SIMULATION_DIR, f"forecasts_{duration}days.csv")
    all_simulations.to_csv(sim_file, index=False)
    logger.info(f"Simulations saved to {sim_file}")
    
    return all_simulations

# =============================================================================
# STAGE 4: EVALUATE & PLOT
# =============================================================================
def stage_4_evaluate(simulations, observed, duration):
    """
    Run CSEP tests and generate the master verification plot.
    """
    logger.info(f"STAGE 4: Evaluating & Plotting for {duration} days...")
    
    plot_file = os.path.join(FIGURE_dIR, f"csep_6panel_evaluation_{duration}days.png")
    
    config = {
        "mc": MC,
        "start": FORECAST_START,
        "duration": duration,
        "shape_coords": os.path.join(INPUT_DATA_DIR, "nz_polygon.npy")
    }
    
    plot_csep_6panel(simulations, observed, config, output_path=plot_file)
    logger.info(f"Evaluation complete. Check {plot_file}")

# =============================================================================
# MAIN PIPELINE RUNNER
# =============================================================================
def main():
    logger.info(f"--- STARTING PIPELINE: {EXPERIMENT_NAME} ---")
    
    # 1. Data
    training_data, full_observed_data = stage_1_prepare_data()
    
    # 2. Estimate
    params = stage_2_estimate_parameters(training_data)
    
    # Prepare Model (Once)
    model_calc = prepare_etas_model(training_data, params)
    
    # Loop over durations
    for duration in FORECAST_DURATIONS:
        logger.info(f"--- Processing {duration}-day Forecast ---")
        
        # 3. Simulate
        simulations = stage_3_simulate_forecasts(model_calc, duration)
        
        # Filter observed data for this duration
        obs_end_time = FORECAST_START + dt.timedelta(days=duration)
        observed_subset = full_observed_data[full_observed_data["time"] < obs_end_time]
        
        # 4. Evaluate
        stage_4_evaluate(simulations, observed_subset, duration)
    
    logger.info("--- PIPELINE FINISHED SUCCESSFULY ---")

if __name__ == "__main__":
    main()
