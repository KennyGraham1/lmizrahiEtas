import json
import logging
import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import etas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation

set_up_logger(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # reads configuration for example ETAS parameter inversion
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        forecast_config = json.load(f)
    
    logger.info("Starting ETAS parameter inversion...")
    etas_invert = ETASParameterCalculation(forecast_config)
    etas_invert.prepare()
    theta = etas_invert.invert()
    logger.info(f"Inversion complete. Parameters: {theta}")

    m_max = forecast_config.get('m_max', None)
    simulation = ETASSimulation(etas_invert, m_max=m_max)
    simulation.prepare()
    fn_store_simulation = forecast_config['fn_store_simulation']
    forecast_duration = forecast_config['forecast_duration']
    n_simulations = forecast_config['n_simulations']
    
    logger.info(f"Starting simulation ({n_simulations} simulations)...")

    store = pd.DataFrame()
    for chunk in simulation.simulate(forecast_duration, n_simulations):
        store = pd.concat([store, chunk], ignore_index=False)
        
    store.to_csv(fn_store_simulation)
    logger.info(f"Simulation complete. Results stored in {fn_store_simulation}")
    logger.info(store.head())
