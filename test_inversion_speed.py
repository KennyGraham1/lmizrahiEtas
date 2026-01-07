
import time
import numpy as np
import pandas as pd
from etas.inversion import neg_log_likelihood, NUMBA_AVAILABLE, expected_aftershocks

# Mock data generation
def generate_mock_data(n_sources=1000, n_targets=5000):
    # Parameters
    theta = [
        -2.5,  # log10_k0 (k0 approx 0.003)
        1.5,   # a
        -2.5,  # log10_c (c approx 0.003)
        -0.1,  # omega
        2.0,   # log10_tau (tau approx 100)
        -1.0,  # log10_d (d approx 0.1)
        0.5,   # gamma
        0.8    # rho
    ]
    mc_min = 2.5
    
    # Source events
    source_events = pd.DataFrame({
        "source_magnitude": np.random.uniform(2.5, 6.0, n_sources),
        "pos_source_to_start_time_distance": np.random.uniform(0, 1000, n_sources),
        "source_to_end_time_distance": np.random.uniform(0, 1000, n_sources),
        "l_hat": np.random.randint(0, 10, n_sources)
    })
    source_events.index.name = "source_id"
    source_events["G"] = 0.0 # Placeholder
    
    # Pij (interactions)
    # Create random interactions
    n_interactions = n_targets * 50 # 50 sources per target on average
    
    source_ids = np.random.randint(0, n_sources, n_interactions)
    target_ids = np.random.randint(0, n_targets, n_interactions)
    
    pij_df = pd.DataFrame({
        "source_id": source_ids,
        "target_id": target_ids,
        "source_magnitude": source_events.iloc[source_ids]["source_magnitude"].values,
        "spatial_distance_squared": np.random.exponential(10, n_interactions),
        "time_distance": np.random.exponential(5, n_interactions),
        "Pij": np.random.random(n_interactions),
        "zeta_plus_1": np.ones(n_interactions)
    })
    
    # Normalize Pij roughly
    pij_df["Pij"] = pij_df["Pij"] / pij_df.groupby("target_id")["Pij"].transform("sum")
    
    # MultiIndex
    pij_df.set_index(["source_id", "target_id"], inplace=True)
    
    return theta, pij_df, source_events, mc_min

def benchmark():
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    print("Generating mock data...")
    theta, pij_df, source_events, mc_min = generate_mock_data(n_sources=1000, n_targets=40000)
    print(f"Pij DataFrame size: {len(pij_df)} rows")
    
    # Warmup
    print("Warming up JIT...")
    neg_log_likelihood(theta, pij_df, source_events, mc_min)
    
    # Benchmark JIT
    n_loops = 20
    print(f"Running {n_loops} iterations (JIT ENABLED)...")
    start_time = time.time()
    for i in range(n_loops):
        val = neg_log_likelihood(theta, pij_df, source_events, mc_min)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_loops
    print(f"Average time per iteration (JIT): {avg_time*1000:.2f} ms")
    
    # Benchmark Pandas (Force Disable)
    import etas.inversion
    original_numba = etas.inversion.NUMBA_AVAILABLE
    etas.inversion.NUMBA_AVAILABLE = False
    
    # Run once to ensure no compilation overhead (though pure python so no big deal)
    neg_log_likelihood(theta, pij_df, source_events, mc_min)
    
    n_loops_pandas = 5 # It will be slower
    print(f"Running {n_loops_pandas} iterations (JIT DISABLED)...")
    start_time = time.time()
    for i in range(n_loops_pandas):
        val = neg_log_likelihood(theta, pij_df, source_events, mc_min)
    end_time = time.time()
    
    avg_time_pandas = (end_time - start_time) / n_loops_pandas
    print(f"Average time per iteration (Pandas): {avg_time_pandas*1000:.2f} ms")
    
    print(f"Speedup: {avg_time_pandas / avg_time:.1f}x")
    
    etas.inversion.NUMBA_AVAILABLE = original_numba

if __name__ == "__main__":
    benchmark()
