"""
Advanced ETAS Forecast Analysis Script

Performs 4 additional statistical analyses:
1. Magnitude-dependent N-tests
2. Information Gain timeline  
3. Spatial L-test
4. Adaptive vs Fixed parameter comparison
"""

import sys
sys.path.insert(0, '.')
from visualize_results import *

def run_advanced_analyses(sequence="Kaikoura"):
    """Execute all 4 advanced analyses for a sequence."""
    
    print(f"\n{'='*70}")
    print(f"Advanced Statistical Analyses: {sequence} Sequence")
    print(f"{'='*70}\n")
    
    # Load data
    catalog = load_catalog()
    params_df = load_parameters(sequence)
    dates_all = KAIKOURA_DATES if sequence == "Kaikoura" else CANTERBURY_DATES
    dates = [params_df.loc[i, "date"] for i in params_df.index if params_df.loc[i, "date"] is not None]
    
    print(f"Loaded {len(params_df)} parameter sets\n")
    
    # ==================== 1. MAGNITUDE-DEPENDENT N-TESTS ====================
    print("1. Running magnitude-dependent N-tests...")
    mag_thresholds = [4.1, 4.5, 5.0, 5.5]
    n_test_results_by_mag = {}
    
    for mag_thresh in mag_thresholds:
        print(f"   Processing M ≥ {mag_thresh}...")
        results = []
        
        for _, row in params_df.iterrows():
            model_idx = row["index"]
            forecast_start = row["date"]
            if forecast_start is None:
                continue
            
            forecast_end = forecast_start + timedelta(days=FORECAST_DAYS)
            sims = load_simulations(sequence, model_idx)
            if len(sims) == 0:
                continue
            
            # Filter by magnitude
            sims_mag = sims[sims["magnitude"] >= mag_thresh]
            observed = get_observed_in_window(catalog, forecast_start, forecast_end)
            observed_mag = observed[observed["magnitude"] >= mag_thresh]
            
            result = n_test(sims_mag, len(observed_mag))
            result["date"] = forecast_start
            result["model_idx"] = model_idx
            results.append(result)
        
        n_test_results_by_mag[mag_thresh] = results
        consistent = sum(1 for r in results if r["consistent"])
        print(f"   → {consistent}/{len(results)} ({100*consistent/len(results):.0f}%) consistent")
    
    plot_magnitude_dependent_ntests(n_test_results_by_mag, sequence, dates,
                                    os.path.join(OUTPUT_DIR, f"mag_dependent_ntest_{sequence.lower()}.png"))
    
    # ==================== 2. INFORMATION GAIN TIMELINE ====================
    print("\n2. Calculating Information Gain timeline...")
    ig_results = []
    
    for _, row in params_df.iterrows():
        model_idx = row["index"]
        forecast_start = row["date"]
        if forecast_start is None:
            continue
        
        forecast_end = forecast_start + timedelta(days=FORECAST_DAYS)
        sims = load_simulations(sequence, model_idx)
        if len(sims) == 0:
            continue
        
        observed = get_observed_in_window(catalog, forecast_start, forecast_end)
        skill = calculate_forecast_skill(sims, observed)
        skill["date"] = forecast_start
        skill["model_idx"] = model_idx
        ig_results.append(skill)
    
    mean_ig = np.mean([r["information_gain"] for r in ig_results])
    print(f"   Mean Information Gain: {mean_ig:.3f} nats")
    
    plot_information_gain_timeline(ig_results, sequence, dates,
                                   os.path.join(OUTPUT_DIR, f"information_gain_{sequence.lower()}.png"))
    
    # ==================== 3. SPATIAL L-TEST ====================
    print("\n3. Computing Spatial L-tests...")
    ltest_results = []
    
    for _, row in params_df.iterrows():
        model_idx = row["index"]
        forecast_start = row["date"]
        if forecast_start is None:
            continue
        
        forecast_end = forecast_start + timedelta(days=FORECAST_DAYS)
        sims = load_simulations(sequence, model_idx)
        if len(sims) == 0:
            continue
        
        observed = get_observed_in_window(catalog, forecast_start, forecast_end)
        ltest = calculate_spatial_ltest(sims, observed)
        ltest["date"] = forecast_start
        ltest["model_idx"] = model_idx
        ltest_results.append(ltest)
    
    mean_ltest = np.nanmean([r["l_test_stat"] for r in ltest_results])
    print(f"   Mean L-test statistic: {mean_ltest:.3f}")
    
    plot_spatial_ltest_results(ltest_results, sequence, dates,
                               os.path.join(OUTPUT_DIR, f"spatial_ltest_{sequence.lower()}.png"))
    
    # ==================== 4. ADAPTIVE VS FIXED COMPARISON ====================
    print("\n4. Comparing adaptive vs fixed parameters...")
    
    # Adaptive: Use current parameters (already done in main analysis)
    adaptive_results = []
    for _, row in params_df.iterrows():
        model_idx = row["index"]
        forecast_start = row["date"]
        if forecast_start is None:
            continue
        
        forecast_end = forecast_start + timedelta(days=FORECAST_DAYS)
        sims = load_simulations(sequence, model_idx)
        if len(sims) == 0:
            continue
        
        observed = get_observed_in_window(catalog, forecast_start, forecast_end)
        result = n_test(sims, len(observed))
        result["date"] = forecast_start
        result["model_idx"] = model_idx
        adaptive_results.append(result)
    
    # Fixed: Use Model 0 parameters for all forecasts
    # Note: This is a simulation - we use Model 0 simulations but evaluate at different times
    fixed_results = []
    model_0_sims = load_simulations(sequence, 0)  # Use model 0 simulations as proxy
    
    for _, row in params_df.iterrows():
        forecast_start = row["date"]
        if forecast_start is None:
            continue
        forecast_end = forecast_start + timedelta(days=FORECAST_DAYS)
        
        # Use Model 0 simulations (representing fixed parameters)
        observed = get_observed_in_window(catalog, forecast_start, forecast_end)
        result = n_test(model_0_sims, len(observed))
        result["date"] = forecast_start
        result["model_idx"] = 0  # Fixed to model 0
        fixed_results.append(result)
    
    adaptive_consistent = sum(1 for r in adaptive_results if r["consistent"])
    fixed_consistent = sum(1 for r in fixed_results if r["consistent"])
    print(f"   Adaptive: {adaptive_consistent}/{len(adaptive_results)} ({100*adaptive_consistent/len(adaptive_results):.0f}%) consistent")
    print(f"   Fixed:    {fixed_consistent}/{len(fixed_results)} ({100*fixed_consistent/len(fixed_results):.0f}%) consistent")
    
    plot_adaptive_vs_fixed_comparison(adaptive_results, fixed_results, sequence, dates,
                                      os.path.join(OUTPUT_DIR, f"adaptive_vs_fixed_{sequence.lower()}.png"))
    
    print(f"\n{'='*70}")
    print(f"✅ Advanced analyses complete!")
    print(f"{'='*70}\n")
    
    return {
        "mag_dependent": n_test_results_by_mag,
        "information_gain": ig_results,
        "spatial_ltest": ltest_results,
        "adaptive_vs_fixed": (adaptive_results, fixed_results)
    }


if __name__ == "__main__":
    # Run for both sequences
    kaikoura_advanced = run_advanced_analyses("Kaikoura")
    canterbury_advanced = run_advanced_analyses("Canterbury")
    
    print("\n" + "="*70)
    print("📊 ADVANCED ANALYSIS SUMMARY")
    print("="*70)
    print("\nGenerated 8 new advanced analysis figures:")
    print("  • Magnitude-dependent N-tests (4 mag thresholds)")
    print("  • Information Gain timelines")
    print("  • Spatial L-test evolution")
    print("  • Adaptive vs Fixed parameter comparison")
    print(f"\nAll saved to: {OUTPUT_DIR}/")
    print("="*70 + "\n")
