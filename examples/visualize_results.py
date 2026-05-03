"""
ETAS Results Visualization Suite (Enhanced)

Comprehensive visualization tools for analyzing ETAS model outputs:
1. Parameter evolution over forecast windows
2. CSEP-style forecast evaluation (N-test, M-test)
3. Forecast vs observed comparisons
4. Spatial density visualizations
5. Magnitude-Time evolution

Usage:
    python visualize_results.py
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from datetime import datetime, timedelta
import warnings

from date_grids import CANTERBURY_DATES, KAIKOURA_DATES

warnings.filterwarnings("ignore")

# --- Professional Styling ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Professional color palettes
COLORS = {
    'primary': '#1E3A5F',      # Deep blue
    'secondary': '#3D7EA6',    # Ocean blue
    'accent': '#E8505B',       # Coral red
    'success': '#2ECC71',      # Emerald green
    'warning': '#F39C12',      # Amber
    'muted': '#95A5A6',        # Gray
    'light': '#ECF0F1',        # Light gray
    'observed': '#C0392B',     # Strong red
    'simulated': '#2980B9',    # Bright blue
    'consistent': '#27AE60',   # Green
    'inconsistent': '#E74C3C', # Red
}

# Custom colormaps
SIM_CMAP = LinearSegmentedColormap.from_list(
    'sim_density', ['#FFFFFF', '#FEF3C7', '#F59E0B', '#DC2626'], N=256
)

# --- Configuration ---
PARAM_DIR = "output_nz"
SIM_DIR = "simulations_nz"
SIM_DIR_FIXED = "simulations_nz_fixed"
CATALOG_PATH = "../input_data/nzcat.csv"
OUTPUT_DIR = "figures"
FORECAST_DAYS = 7.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Data Loading Functions ---

def load_parameters(sequence: str) -> pd.DataFrame:
    """Load all parameter files for a sequence into a DataFrame."""
    pattern = os.path.join(PARAM_DIR, f"parameters_nz_{sequence}_*.json")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split("_")[-1].replace(".json", "")))
    
    params_list = []
    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
        
        idx = int(f.split("_")[-1].replace(".json", ""))
        dates = KAIKOURA_DATES if sequence == "Kaikoura" else CANTERBURY_DATES
        
        record = {
            "index": idx,
            "date": dates[idx] if idx < len(dates) else None,
            **data.get("final_parameters", {}),
            "n_hat": data.get("n_hat"),
            "beta": data.get("beta"),
            "n_iterations": data.get("n_iterations"),
        }
        params_list.append(record)
    
    df = pd.DataFrame(params_list)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def _load_simulation_chunks(pattern: str) -> pd.DataFrame:
    """Load simulation chunk files and make catalog IDs unique across chunks."""
    files = sorted(glob.glob(pattern))  # Sort to ensure consistent ordering
    if not files:
        return pd.DataFrame()
    
    chunks = []
    catalog_offset = 0
    for f in files:
        df = pd.read_csv(f, parse_dates=["time"])
        if "catalog_id" in df.columns and len(df) > 0:
            raw_ids = df["catalog_id"].astype(int)
            id_span = max(int(raw_ids.max()) + 1, raw_ids.nunique())
            df["catalog_id"] = raw_ids + catalog_offset
            catalog_offset += id_span
        chunks.append(df)
    
    return pd.concat(chunks, ignore_index=True)


def load_simulations(
    sequence: str,
    model_idx: int,
    sim_dir: str = SIM_DIR,
    prefix: str = "sim_nz",
) -> pd.DataFrame:
    """Load all simulation chunks for a specific adaptive model."""
    pattern = os.path.join(sim_dir, f"{prefix}_{sequence}_{model_idx}_*.csv")
    return _load_simulation_chunks(pattern)


def load_fixed_simulations(
    sequence: str,
    model_idx: int,
    sim_dir: str = SIM_DIR_FIXED,
) -> pd.DataFrame:
    """Load all simulation chunks for a fixed regional baseline model."""
    return load_simulations(
        sequence,
        model_idx,
        sim_dir=sim_dir,
        prefix="sim_nz_fixed",
    )


def load_catalog() -> pd.DataFrame:
    """Load the observed earthquake catalog."""
    return pd.read_csv(CATALOG_PATH, index_col=0, parse_dates=["time"])


def get_observed_in_window(catalog: pd.DataFrame, start: datetime, 
                           end: datetime, mc: float = 4.1) -> pd.DataFrame:
    """Filter catalog to events within forecast window above Mc."""
    mask = (catalog["time"] > start) & (catalog["time"] <= end) & (catalog["magnitude"] >= mc)
    return catalog[mask].copy()


def filter_simulations_to_window(simulations: pd.DataFrame, start: datetime,
                                 end: datetime, mc: float = 4.1) -> pd.DataFrame:
    """Filter simulated events to a forecast window above Mc."""
    if len(simulations) == 0:
        return simulations.copy()

    sims = simulations.copy()
    if not pd.api.types.is_datetime64_any_dtype(sims["time"]):
        sims["time"] = pd.to_datetime(sims["time"])

    mask = (
        (sims["time"] > start)
        & (sims["time"] <= end)
        & (sims["magnitude"] >= mc)
    )
    return sims[mask].copy()


# --- CSEP-Style Evaluation Functions ---

def n_test(simulations: pd.DataFrame, observed_count: int) -> dict:
    """Perform N-test: Compare observed event count to simulated distribution."""
    sim_counts = simulations.groupby("catalog_id").size()
    quantile = (sim_counts < observed_count).mean()
    
    return {
        "observed": observed_count,
        "simulated_mean": sim_counts.mean(),
        "simulated_std": sim_counts.std(),
        "simulated_median": sim_counts.median(),
        "quantile": quantile,
        "p5": sim_counts.quantile(0.05),
        "p25": sim_counts.quantile(0.25),
        "p75": sim_counts.quantile(0.75),
        "p95": sim_counts.quantile(0.95),
        "consistent": 0.025 <= quantile <= 0.975,
        "distribution": sim_counts.values,
    }


# --- Enhanced Plotting Functions ---

def plot_parameter_evolution(params_df: pd.DataFrame, sequence: str, 
                             output_path: str = None):
    """Create publication-quality multi-panel parameter evolution plot."""
    
    param_configs = [
        ("log10_k0", r"$\log_{10}(k_0)$", "Productivity", "#2E86AB"),
        ("a", r"$\alpha$", "Magnitude Efficiency", "#A23B72"),
        ("omega", r"$\omega$", "Omori Exponent (p-1)", "#F18F01"),
        ("log10_tau", r"$\log_{10}(\tau)$", "Taper Time (days)", "#C73E1D"),
        ("gamma", r"$\gamma$", "Spatial Mag. Scaling", "#3B1F2B"),
        ("rho", r"$\rho$", "Spatial Decay", "#2E7D32"),
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle(f"ETAS Parameter Evolution: {sequence} Earthquake Sequence", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    mainshock_date = params_df["date"].min()
    mainshock_str = mainshock_date.strftime("%Y-%m-%d %H:%M")
    
    for ax, (param, label, desc, color) in zip(axes.flatten(), param_configs):
        if param not in params_df.columns:
            ax.set_visible(False)
            continue
        
        days = np.array([(d - mainshock_date).total_seconds() / 86400 for d in params_df["date"]])
        values = params_df[param].values
        
        # Main line with markers
        ax.plot(days, values, 'o-', color=color, markersize=7, linewidth=2, 
                markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        
        # Fill between min-max for visual emphasis
        ax.fill_between(days, values.min(), values, alpha=0.15, color=color)
        
        # Rolling mean (smoothed trend)
        if len(days) >= 5:
            window = min(5, len(days))
            rolling_mean = pd.Series(values).rolling(window, center=True).mean()
            ax.plot(days, rolling_mean, '--', color='gray', linewidth=1.5, 
                    alpha=0.7, label='5-pt Moving Avg')
        
        # Styling
        ax.set_xlabel("Days After Mainshock", fontweight='medium')
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(desc, fontsize=11, fontweight='bold', pad=8)
        
        # Add value annotations for first and last point
        ax.annotate(f'{values[0]:.3f}', (days[0], values[0]), 
                    textcoords="offset points", xytext=(-8, 10), fontsize=8, color=color)
        ax.annotate(f'{values[-1]:.3f}', (days[-1], values[-1]), 
                    textcoords="offset points", xytext=(5, -12), fontsize=8, color=color)
        
        # Subtle horizontal line at mean
        ax.axhline(values.mean(), color=color, linestyle=':', alpha=0.4, linewidth=1)
    
    # Add mainshock info
    fig.text(0.5, 0.01, f"Mainshock: {mainshock_str} UTC", ha='center', 
             fontsize=10, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_n_test_result(n_test_result: dict, forecast_date: datetime, 
                       sequence: str, output_path: str = None):
    """Create publication-quality N-test histogram."""
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    dist = n_test_result["distribution"]
    obs = n_test_result["observed"]
    
    # Histogram with gradient effect
    n, bins, patches = ax.hist(dist, bins=35, density=True, alpha=0.85,
                                edgecolor='white', linewidth=0.8)
    
    # Color gradient based on position relative to observed
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < obs:
            patch.set_facecolor(COLORS['simulated'])
        else:
            patch.set_facecolor('#85C1E9')  # Lighter blue for above observed
    
    # Observed count - prominent vertical line
    ax.axvline(obs, color=COLORS['observed'], linewidth=3, linestyle='-', 
               zorder=5, label=f"Observed: {obs} events")
    
    # Add arrow annotation for observed
    y_max = n.max()
    ax.annotate('', xy=(obs, y_max * 0.95), xytext=(obs, y_max * 1.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['observed'], lw=2))
    ax.text(obs, y_max * 1.18, 'OBSERVED', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['observed'])
    
    # Percentile lines with shading
    ax.axvspan(n_test_result["p5"], n_test_result["p95"], alpha=0.1, 
               color='gray', label='90% Prediction Interval')
    ax.axvline(n_test_result["p5"], color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(n_test_result["p95"], color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(n_test_result["simulated_median"], color=COLORS['primary'], 
               linestyle=':', linewidth=2, alpha=0.8, label=f'Median: {n_test_result["simulated_median"]:.0f}')
    
    # Status badge
    if n_test_result["consistent"]:
        status_text = "✓ CONSISTENT"
        status_color = COLORS['consistent']
    else:
        status_text = "✗ INCONSISTENT"
        status_color = COLORS['inconsistent']
    
    # Add status box
    props = dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.2, 
                 edgecolor=status_color, linewidth=2)
    ax.text(0.97, 0.97, status_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', fontweight='bold',
            color=status_color, bbox=props)
    
    # Quantile info
    q = n_test_result["quantile"]
    ax.text(0.97, 0.85, f"Quantile: {q:.3f}", transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', color='gray')
    
    # Labels and title
    ax.set_xlabel("Number of Earthquakes (M ≥ 4.1)", fontsize=12, fontweight='medium')
    ax.set_ylabel("Probability Density", fontsize=12, fontweight='medium')
    ax.set_title(f"N-Test: {sequence} Sequence\nForecast Period: {forecast_date.strftime('%Y-%m-%d')} → "
                 f"{(forecast_date + timedelta(days=FORECAST_DAYS)).strftime('%Y-%m-%d')} (7 days)", 
                 fontsize=13, fontweight='bold', pad=12)
    
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='lightgray')
    ax.set_ylim(0, y_max * 1.25)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_n_test_summary(n_test_results: list, sequence: str, dates: list,
                        output_path: str = None):
    """Create comprehensive N-test summary with dual-axis plot."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    quantiles = [r["quantile"] for r in n_test_results]
    observed = [r["observed"] for r in n_test_results]
    simulated_median = [r["simulated_median"] for r in n_test_results]
    simulated_p5 = [r["p5"] for r in n_test_results]
    simulated_p95 = [r["p95"] for r in n_test_results]
    
    mainshock = min(dates)
    days = [(d - mainshock).total_seconds() / 86400 for d in dates]
    
    # --- Top Panel: Quantile Plot ---
    consistent_mask = [r["consistent"] for r in n_test_results]
    
    # Consistency band
    ax1.axhspan(0.025, 0.975, alpha=0.15, color=COLORS['consistent'], 
                label='95% Consistency Region', zorder=1)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot quantiles with color coding
    for i, (d, q, consistent) in enumerate(zip(days, quantiles, consistent_mask)):
        color = COLORS['consistent'] if consistent else COLORS['inconsistent']
        ax1.scatter(d, q, c=color, s=120, edgecolor='white', linewidth=2, zorder=3)
    
    ax1.plot(days, quantiles, 'k-', alpha=0.3, linewidth=1, zorder=2)
    
    # Labels
    ax1.set_ylabel("N-Test Quantile", fontsize=12, fontweight='medium')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(min(days) - 0.5, max(days) + 0.5)
    
    n_consistent = sum(consistent_mask)
    n_total = len(consistent_mask)
    ax1.set_title(f"N-Test Summary: {sequence} Sequence\n"
                  f"Consistent Forecasts: {n_consistent}/{n_total} ({100*n_consistent/n_total:.0f}%)", 
                  fontsize=14, fontweight='bold', pad=10)
    
    ax1.legend(loc='upper right', framealpha=0.95)
    
    # --- Bottom Panel: Observed vs Simulated Comparison ---
    # Prediction interval (p5 to p95)
    ax2.fill_between(days, simulated_p5, simulated_p95,
                     alpha=0.25, color=COLORS['simulated'], 
                     label='90% Prediction Interval (p05-p95)')
    
    # Simulated Median
    ax2.plot(days, simulated_median, 'o-', color=COLORS['simulated'], linewidth=2, 
             markersize=8, markeredgecolor='white', label='Simulated Median')
    
    # Observed
    ax2.plot(days, observed, 's-', color=COLORS['observed'], linewidth=2, 
             markersize=8, markeredgecolor='white', label='Observed')
    
    ax2.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax2.set_ylabel("Event Count (7-day window)", fontsize=12, fontweight='medium')
    ax2.set_xlim(min(days) - 0.5, max(days) + 0.5)
    ax2.legend(loc='upper right', framealpha=0.95)
    
    # Add ratio annotations
    for d, obs, sim in zip(days[::3], observed[::3], simulated_median[::3]):
        ratio = obs / sim if sim > 0 else 0
        ax2.annotate(f'{ratio:.1f}x', (d, max(obs, sim) + 10), 
                     fontsize=8, ha='center', color='gray', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_cumulative_comparison(simulations: pd.DataFrame, observed: pd.DataFrame,
                               forecast_start: datetime, sequence: str,
                               output_path: str = None):
    """Create enhanced cumulative event comparison plot."""
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    forecast_end = forecast_start + timedelta(days=FORECAST_DAYS)
    
    # Observed cumulative count with step function
    if len(observed) > 0:
        obs_sorted = observed.sort_values("time")
        obs_times = list(obs_sorted["time"])
        obs_cumulative = list(range(1, len(obs_sorted) + 1))
        
        # Add starting point
        obs_times.insert(0, forecast_start)
        obs_cumulative.insert(0, 0)
        
        ax.step(obs_times, obs_cumulative, where="post", linewidth=3, 
                color=COLORS['observed'], label=f"Observed ({len(observed)} events)", zorder=4)
        
        # Mark each event
        ax.scatter(obs_sorted["time"], range(1, len(obs_sorted) + 1), 
                   s=30, c=COLORS['observed'], edgecolor='white', zorder=5)
    
    # Simulated distribution
    if len(simulations) > 0:
        sim_counts = simulations.groupby("catalog_id").size()
        median_count = sim_counts.median()
        p5, p25, p75, p95 = sim_counts.quantile([0.05, 0.25, 0.75, 0.95])
        
        # Shaded regions for uncertainty
        ax.fill_between([forecast_start, forecast_end], [0, p5], [0, p95],
                        alpha=0.15, color=COLORS['simulated'], label='90% Prediction Interval')
        ax.fill_between([forecast_start, forecast_end], [0, p25], [0, p75],
                        alpha=0.25, color=COLORS['simulated'], label='50% Prediction Interval')
        ax.plot([forecast_start, forecast_end], [0, median_count], 
                '--', color=COLORS['simulated'], linewidth=2.5, 
                label=f"Simulated Median: {median_count:.0f}")
        
        # Add expected rate line
        ax.plot([forecast_start, forecast_end], [0, sim_counts.mean()], 
                ':', color=COLORS['primary'], linewidth=2, alpha=0.7,
                label=f"Simulated Mean: {sim_counts.mean():.0f}")
    
    # Styling
    ax.set_xlabel("Date", fontsize=12, fontweight='medium')
    ax.set_ylabel("Cumulative Number of Earthquakes (M ≥ 4.1)", fontsize=12, fontweight='medium')
    ax.set_title(f"Cumulative Seismicity: {sequence} Sequence\n"
                 f"Forecast: {forecast_start.strftime('%Y-%m-%d')} to {forecast_end.strftime('%Y-%m-%d')}", 
                 fontsize=13, fontweight='bold', pad=12)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha='right')
    
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='lightgray')
    ax.set_xlim(forecast_start - timedelta(hours=6), forecast_end + timedelta(hours=6))
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_spatial_comparison(simulations: pd.DataFrame, observed: pd.DataFrame,
                            sequence: str, output_path: str = None):
    """Create symmetric density comparison with confidence contours."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Determine bounds from both datasets
    all_lons = pd.concat([simulations["longitude"], observed["longitude"]]) if len(simulations) > 0 else observed["longitude"]
    all_lats = pd.concat([simulations["latitude"], observed["latitude"]]) if len(simulations) > 0 else observed["latitude"]
    
    lon_margin = (all_lons.max() - all_lons.min()) * 0.1 + 0.3
    lat_margin = (all_lats.max() - all_lats.min()) * 0.1 + 0.3
    
    lon_min, lon_max = all_lons.min() - lon_margin, all_lons.max() + lon_margin
    lat_min, lat_max = all_lats.min() - lat_margin, all_lats.max() + lat_margin
    
    # Create common color scale for density comparison
    if len(simulations) > 0 and len(observed) > 0:
        # Sample densities to get vmax
        test_hex_sim = axes[0].hexbin(simulations["longitude"], simulations["latitude"], 
                                       gridsize=35, visible=False)
        test_hex_obs = axes[1].hexbin(observed["longitude"], observed["latitude"], 
                                       gridsize=35, visible=False)
        vmax = max(test_hex_sim.get_array().max(), test_hex_obs.get_array().max())
        test_hex_sim.remove()
        test_hex_obs.remove()
    else:
        vmax = None
    
    # --- Left: Simulated Density with Confidence Contour ---
    ax1 = axes[0]
    if len(simulations) > 0:
        h1 = ax1.hexbin(simulations["longitude"], simulations["latitude"], 
                        gridsize=35, cmap=SIM_CMAP, mincnt=1, alpha=0.85,
                        linewidths=0.1, edgecolors='white', vmax=vmax)
        
        # Add 90% confidence contour
        try:
            from scipy.stats import gaussian_kde
            from matplotlib.patches import Polygon as MplPolygon
            
            # KDE for confidence regions
            lon_vals = simulations["longitude"].values
            lat_vals = simulations["latitude"].values
            
            # Subsample if too many points
            if len(lon_vals) > 5000:
                idx = np.random.choice(len(lon_vals), 5000, replace=False)
                lon_vals = lon_vals[idx]
                lat_vals = lat_vals[idx]
            
            kde = gaussian_kde(np.vstack([lon_vals, lat_vals]))
            
            # Create density grid
            lon_grid = np.linspace(lon_min, lon_max, 50)
            lat_grid = np.linspace(lat_min, lat_max, 50)
            LON, LAT = np.meshgrid(lon_grid, lat_grid)
            positions = np.vstack([LON.ravel(), LAT.ravel()])
            density = kde(positions).reshape(LON.shape)
            
            # Find 90% contour level
            sorted_density = np.sort(density.ravel())[::-1]
            cumsum = np.cumsum(sorted_density)
            cumsum /= cumsum[-1]
            threshold_90 = sorted_density[np.searchsorted(cumsum, 0.90)]
            
            # Plot contour
            contour = ax1.contour(LON, LAT, density, levels=[threshold_90], 
                                  colors='darkred', linewidths=2.5, linestyles='--', alpha=0.8)
            ax1.clabel(contour, inline=True, fontsize=8, fmt='90%')
        except:
            pass  # Skip contour if scipy not available
        
        cbar1 = plt.colorbar(h1, ax=ax1, shrink=0.8, pad=0.02)
        cbar1.set_label("Event Density", fontsize=10)
    
    ax1.set_xlim(lon_min, lon_max)
    ax1.set_ylim(lat_min, lat_max)
    ax1.set_xlabel("Longitude (°E)", fontsize=11, fontweight='medium')
    ax1.set_ylabel("Latitude (°S)", fontsize=11, fontweight='medium')
    ax1.set_title(f"Simulated Density\n({len(simulations)} events, all catalogs)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # --- Right: Observed Density ---
    ax2 = axes[1]
    if len(observed) > 0:
        h2 = ax2.hexbin(observed["longitude"], observed["latitude"], 
                        gridsize=35, cmap=SIM_CMAP, mincnt=1, alpha=0.85,
                        linewidths=0.1, edgecolors='white', vmax=vmax)
        
        # Overlay individual events as points
        sizes = (observed["magnitude"] - 4.0) ** 2 * 15
        ax2.scatter(observed["longitude"], observed["latitude"], 
                    s=sizes, c='black', alpha=0.4, edgecolor='white', 
                    linewidth=0.5, zorder=3, label='Individual Events')
        
        # Annotate largest event
        max_idx = observed["magnitude"].idxmax()
        max_event = observed.loc[max_idx]
        ax2.annotate(f'M{max_event["magnitude"]:.1f}', 
                     (max_event["longitude"], max_event["latitude"]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, fontweight='bold', color=COLORS['observed'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['observed'], lw=2),
                     zorder=4)
        
        cbar2 = plt.colorbar(h2, ax=ax2, shrink=0.8, pad=0.02)
        cbar2.set_label("Event Density", fontsize=10)
    
    ax2.set_xlim(lon_min, lon_max)
    ax2.set_ylim(lat_min, lat_max)
    ax2.set_xlabel("Longitude (°E)", fontsize=11, fontweight='medium')
    ax2.set_ylabel("Latitude (°S)", fontsize=11, fontweight='medium')
    ax2.set_title(f"Observed Density\n({len(observed)} events, M ≥ 4.1)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(f"Spatial Comparison: {sequence} Earthquake Sequence", 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_magnitude_comparison(simulations: pd.DataFrame, observed: pd.DataFrame,
                              sequence: str, forecast_date: datetime,
                              output_path: str = None):
    """Create magnitude-frequency comparison plot (Gutenberg-Richter style)."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    mag_bins = np.arange(4.0, 7.5, 0.2)
    
    # --- Left: Histograms ---
    if len(observed) > 0:
        ax1.hist(observed["magnitude"], bins=mag_bins, alpha=0.7, 
                 color=COLORS['observed'], edgecolor='white', linewidth=1,
                 label=f'Observed (n={len(observed)})', density=True)
    
    if len(simulations) > 0:
        ax1.hist(simulations["magnitude"], bins=mag_bins, alpha=0.5, 
                 color=COLORS['simulated'], edgecolor='white', linewidth=1,
                 label=f'Simulated (n={len(simulations)})', density=True)
    
    ax1.set_xlabel("Magnitude", fontsize=12, fontweight='medium')
    ax1.set_ylabel("Probability Density", fontsize=12, fontweight='medium')
    ax1.set_title("Magnitude Distribution", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95)
    
    # --- Right: Cumulative G-R plot ---
    if len(observed) > 0:
        obs_mags = np.sort(observed["magnitude"])[::-1]
        obs_cum = np.arange(1, len(obs_mags) + 1)
        ax2.semilogy(obs_mags, obs_cum, 'o-', color=COLORS['observed'], 
                     markersize=6, linewidth=1.5, label='Observed')
    
    if len(simulations) > 0:
        # Sample for visualization
        sample_size = min(10000, len(simulations))
        sim_sample = simulations.sample(n=sample_size) if len(simulations) > sample_size else simulations
        sim_mags = np.sort(sim_sample["magnitude"])[::-1]
        sim_cum = np.arange(1, len(sim_mags) + 1) * len(simulations) / len(sim_sample)
        ax2.semilogy(sim_mags, sim_cum, 's-', color=COLORS['simulated'], 
                     markersize=4, linewidth=1, alpha=0.7, label='Simulated')
    
    ax2.set_xlabel("Magnitude ≥ M", fontsize=12, fontweight='medium')
    ax2.set_ylabel("Cumulative Count (log scale)", fontsize=12, fontweight='medium')
    ax2.set_title("Gutenberg-Richter Comparison", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.3, which='both')
    
    fig.suptitle(f"Magnitude Analysis: {sequence} - {forecast_date.strftime('%Y-%m-%d')}", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_magnitude_time(catalog: pd.DataFrame, sequence: str, 
                        forecast_dates: list, output_path: str = None):
    """Create a Magnitude-Time plot with forecast windows overlay."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Filter catalog around the sequence
    start_date = min(forecast_dates) - timedelta(days=5)
    end_date = max(forecast_dates) + timedelta(days=20)
    
    events = catalog[(catalog["time"] >= start_date) & (catalog["time"] <= end_date)]
    
    # Plot events
    sizes = (events["magnitude"] - 3) ** 2.5 * 10
    sc = ax.scatter(events["time"], events["magnitude"], s=sizes, c=events["magnitude"],
                    cmap='plasma', alpha=0.7, edgecolors='k', linewidth=0.5, zorder=2)
    
    # Add mainshock annotation
    mainshock = events.loc[events["magnitude"].idxmax()]
    ax.annotate(f"Mainshock M{mainshock['magnitude']}", 
                (mainshock["time"], mainshock["magnitude"]),
                xytext=(15, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                fontweight='bold', zorder=5)
    
    # Overlay forecast windows
    for i, date in enumerate(forecast_dates):
        end = date + timedelta(days=FORECAST_DAYS)
        # Alternate colors/heights to avoid overlap clutter
        y_pos = 4.0 if i % 2 == 0 else 3.8
        
        # Add forecast window bracket/line
        ax.hlines(y_pos, date, end, colors='gray', linestyles='-', linewidth=2, alpha=0.6)
        ax.vlines([date, end], y_pos - 0.05, y_pos + 0.05, colors='gray', linewidth=2, alpha=0.6)
    
    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title(f"Magnitude-Time Evolution: {sequence} Sequence", fontsize=16, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=30, ha='right')
    
    cbar = plt.colorbar(sc, label="Magnitude")
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_m_test(simulations: pd.DataFrame, observed: pd.DataFrame,
                sequence: str, forecast_date: datetime, output_path: str = None):
    """Create enhanced M-test with Q-Q plot for magnitude distribution."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    mc = 4.1
    mag_bins = np.arange(mc, 7.5, 0.2)
    
    # --- Left: Magnitude Distribution Comparison ---
    if len(observed) > 0:
        obs_hist, _ = np.histogram(observed["magnitude"], bins=mag_bins, density=True)
        ax1.hist(observed["magnitude"], bins=mag_bins, alpha=0.7, 
                 color=COLORS['observed'], edgecolor='white', linewidth=1,
                 label=f'Observed (n={len(observed)})', density=True)
    
    if len(simulations) > 0:
        sim_hist, _ = np.histogram(simulations["magnitude"], bins=mag_bins, density=True)
        ax1.hist(simulations["magnitude"], bins=mag_bins, alpha=0.5, 
                 color=COLORS['simulated'], edgecolor='white', linewidth=1,
                 label=f'Simulated (n={len(simulations)})', density=True)
    
    ax1.set_xlabel("Magnitude", fontsize=12, fontweight='medium')
    ax1.set_ylabel("Probability Density", fontsize=12, fontweight='medium')
    ax1.set_title("Magnitude Distribution", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # --- Right: Q-Q Plot ---
    if len(observed) > 0 and len(simulations) > 0:
        # Quantiles for comparison
        quantiles = np.linspace(0, 1, min(len(observed), 100))
        obs_quantiles = np.quantile(observed["magnitude"], quantiles)
        sim_quantiles = np.quantile(simulations["magnitude"], quantiles)
        
        # Plot Q-Q
        ax2.scatter(sim_quantiles, obs_quantiles, c=COLORS['primary'], 
                    s=50, alpha=0.7, edgecolor='white', linewidth=1)
        
        # 1:1 reference line
        mag_min = min(obs_quantiles.min(), sim_quantiles.min())
        mag_max = max(obs_quantiles.max(), sim_quantiles.max())
        ax2.plot([mag_min, mag_max], [mag_min, mag_max], 
                 'k--', linewidth=2, alpha=0.5, label='1:1 Line')
        
        # Calculate and display correlation
        from scipy.stats import pearsonr
        corr, pval = pearsonr(sim_quantiles, obs_quantiles)
        ax2.text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.3e}',
                 transform=ax2.transAxes, fontsize=11, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel("Simulated Magnitude Quantiles", fontsize=12, fontweight='medium')
    ax2.set_ylabel("Observed Magnitude Quantiles", fontsize=12, fontweight='medium')
    ax2.set_title("Magnitude Q-Q Plot", fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    fig.suptitle(f"M-Test: {sequence} - {forecast_date.strftime('%Y-%m-%d')}", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_parameter_correlation(params_df: pd.DataFrame, sequence: str,
                                output_path: str = None):
    """Create correlation heatmap of ETAS parameters."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select parameters for correlation
    param_cols = ["log10_k0", "a", "omega", "log10_tau", "gamma", "rho"]
    param_labels = [r"$\log_{10}(k_0)$", r"$\alpha$", r"$\omega$", 
                    r"$\log_{10}(\tau)$", r"$\gamma$", r"$\rho$"]
    
    # Filter to available parameters
    available = [p for p in param_cols if p in params_df.columns]
    available_labels = [param_labels[i] for i, p in enumerate(param_cols) if p in params_df.columns]
    
    if len(available) < 2:
        ax.text(0.5, 0.5, "Insufficient parameters for correlation", 
                ha='center', va='center', fontsize=12)
        plt.close(fig)
        return None
    
    # Compute correlation matrix
    corr_matrix = params_df[available].corr()
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(available)))
    ax.set_yticks(np.arange(len(available)))
    ax.set_xticklabels(available_labels, fontsize=11)
    ax.set_yticklabels(available_labels, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add correlation values
    for i in range(len(available)):
        for j in range(len(available)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                          fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation Coefficient", fontsize=11)
    
    ax.set_title(f"Parameter Correlation Matrix: {sequence} Sequence", 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_summary_table(n_test_results: list, sequence: str, output_path: str = None):
    """Create summary statistics table for all N-tests."""
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(n_test_results) * 0.4)))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Model', 'Date', 'Observed', 'Simulated\nMedian', 'Simulated\np05-p95', 
               'Quantile', 'Status']
    
    for result in n_test_results:
        model_idx = result['model_idx']
        date_str = result['date'].strftime('%Y-%m-%d')
        obs = result['observed']
        sim_med = f"{result['simulated_median']:.0f}"
        sim_range = f"{result['p5']:.0f}-{result['p95']:.0f}"
        quantile = f"{result['quantile']:.3f}"
        status = '✓' if result['consistent'] else '✗'
        
        table_data.append([f"{model_idx}", date_str, f"{obs}", sim_med, sim_range, quantile, status])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.08, 0.15, 0.12, 0.12, 0.15, 0.12, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Color code rows by consistency
    for i, result in enumerate(n_test_results):
        row_idx = i + 1
        color = COLORS['consistent'] if result['consistent'] else COLORS['inconsistent']
        
        # Status column
        cell = table[(row_idx, 6)]
        cell.set_facecolor(color)
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        
        # Alternate row shading
        bg_color = '#F8F9FA' if i % 2 == 0 else 'white'
        for j in range(len(headers) - 1):
            table[(row_idx, j)].set_facecolor(bg_color)
    
    # Summary stats
    n_consistent = sum(1 for r in n_test_results if r['consistent'])
    n_total = len(n_test_results)
    mean_quantile = np.mean([r['quantile'] for r in n_test_results])
    
    summary_text = (f"Summary: {n_consistent}/{n_total} ({100*n_consistent/n_total:.0f}%) consistent forecasts  |  "
                   f"Mean Quantile: {mean_quantile:.3f}")
    
    fig.text(0.5, 0.05, summary_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8))
    
    ax.set_title(f"N-Test Summary: {sequence} Sequence", 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_temporal_residuals(n_test_results: list, sequence: str, dates: list,
                            output_path: str = None):
    """Plot temporal evolution of forecast residuals (Observed - Simulated)."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1])
    
    mainshock = min(dates)
    days = [(d - mainshock).total_seconds() / 86400 for d in dates]
    
    observed = [r["observed"] for r in n_test_results]
    simulated_median = [r["simulated_median"] for r in n_test_results]
    residuals = [obs - sim for obs, sim in zip(observed, simulated_median)]
    residual_pct = [100 * (obs - sim) / sim if sim > 0 else 0 
                    for obs, sim in zip(observed, simulated_median)]
    
    # --- Top: Absolute Residuals ---
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.fill_between(days, 0, residuals, 
                     where=[r >= 0 for r in residuals],
                     alpha=0.3, color=COLORS['warning'], label='Over-prediction')
    ax1.fill_between(days, 0, residuals,
                     where=[r < 0 for r in residuals],
                     alpha=0.3, color=COLORS['simulated'], label='Under-prediction')
    
    ax1.plot(days, residuals, 'o-', color=COLORS['primary'], 
             linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    # Add value labels
    for d, res in zip(days[::2], residuals[::2]):
        ax1.annotate(f'{res:+.0f}', (d, res), textcoords="offset points",
                    xytext=(0, 10 if res > 0 else -15), ha='center',
                    fontsize=8, color='gray')
    
    ax1.set_ylabel("Residual (Obs - Sim)", fontsize=12, fontweight='medium')
    ax1.set_title(f"Temporal Forecast Residuals: {sequence} Sequence", 
                  fontsize=14, fontweight='bold', pad=12)
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(days) - 0.5, max(days) + 0.5)
    
    # --- Bottom: Percentage Residuals ---
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.bar(days, residual_pct, width=0.6, 
            color=[COLORS['warning'] if r >= 0 else COLORS['simulated'] for r in residual_pct],
            alpha=0.7, edgecolor='white', linewidth=1)
    
    ax2.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax2.set_ylabel("Residual (%)", fontsize=12, fontweight='medium')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(min(days) - 0.5, max(days) + 0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_seismicity_rate(simulations: pd.DataFrame, observed: pd.DataFrame,
                        forecast_start: datetime, sequence: str,
                        output_path: str = None):
    """Plot seismicity rate evolution (events/day) with uncertainty."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    forecast_end = forecast_start + timedelta(days=FORECAST_DAYS)
    
    # Create time bins (hourly for smoothness, then aggregate to daily)
    time_bins = pd.date_range(forecast_start, forecast_end, freq='6h')
    
    # Observed rate
    if len(observed) > 0:
        obs_binned, _ = np.histogram(observed["time"], bins=time_bins)
        obs_rate = obs_binned / 0.25  # events per day (6h bins)
        bin_centers = time_bins[:-1] + (time_bins[1] - time_bins[0]) / 2
        
        ax.step(bin_centers, obs_rate, where='mid', linewidth=2.5, 
                color=COLORS['observed'], label=f'Observed Rate', zorder=3)
        ax.scatter(bin_centers, obs_rate, s=40, c=COLORS['observed'], 
                  edgecolor='white', zorder=4)
    
    # Simulated rate distribution
    if len(simulations) > 0:
        # Convert simulation times
        simulations['time_dt'] = pd.to_datetime(simulations['time'])
        
        # Calculate rate for each catalog
        unique_catalogs = simulations['catalog_id'].unique()
        rates_per_bin = []
        
        for cat_id in unique_catalogs[:1000]:  # Sample for performance
            cat_data = simulations[simulations['catalog_id'] == cat_id]
            binned, _ = np.histogram(cat_data['time_dt'], bins=time_bins)
            rates_per_bin.append(binned / 0.25)
        
        rates_per_bin = np.array(rates_per_bin)
        
        # Calculate percentiles
        median_rate = np.median(rates_per_bin, axis=0)
        p5_rate = np.percentile(rates_per_bin, 5, axis=0)
        p95_rate = np.percentile(rates_per_bin, 95, axis=0)
        
        bin_centers = time_bins[:-1] + (time_bins[1] - time_bins[0]) / 2
        
        # Plot
        ax.fill_between(bin_centers, p5_rate, p95_rate, alpha=0.25, 
                        color=COLORS['simulated'], label='90% Prediction Interval')
        ax.plot(bin_centers, median_rate, '--', linewidth=2, 
                color=COLORS['simulated'], label='Simulated Median Rate')
    
    # Formatting
    ax.set_xlabel("Date", fontsize=12, fontweight='medium')
    ax.set_ylabel("Seismicity Rate (events/day, M ≥ 4.1)", fontsize=12, fontweight='medium')
    ax.set_title(f"Seismicity Rate Evolution: {sequence} Sequence\n"
                 f"Forecast: {forecast_start.strftime('%Y-%m-%d')} to {forecast_end.strftime('%Y-%m-%d')}", 
                 fontsize=13, fontweight='bold', pad=12)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(forecast_start, forecast_end)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig



def plot_multi_sequence_comparison(kaikoura_results: list, canterbury_results: list,
                                    output_path: str = None):
    """Create side-by-side comparison of both sequences."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Extract data
    k_days = [(r["date"] - min([r["date"] for r in kaikoura_results])).total_seconds() / 86400 
              for r in kaikoura_results]
    c_days = [(r["date"] - min([r["date"] for r in canterbury_results])).total_seconds() / 86400 
              for r in canterbury_results]
    
    k_quantiles = [r["quantile"] for r in kaikoura_results]
    c_quantiles = [r["quantile"] for r in canterbury_results]
    
    k_obs = [r["observed"] for r in kaikoura_results]
    c_obs = [r["observed"] for r in canterbury_results]
    
    k_sim = [r["simulated_median"] for r in kaikoura_results]
    c_sim = [r["simulated_median"] for r in canterbury_results]
    
    # --- Top Left: N-test Quantiles ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axhspan(0.025, 0.975, alpha=0.15, color=COLORS['consistent'])
    ax1.plot(k_days, k_quantiles, 'o-', color='#E74C3C', linewidth=2, 
             markersize=8, label='Kaikoura', markeredgecolor='white', markeredgewidth=2)
    ax1.plot(c_days, c_quantiles, 's-', color='#3498DB', linewidth=2, 
             markersize=8, label='Canterbury', markeredgecolor='white', markeredgewidth=2)
    ax1.set_ylabel("N-Test Quantile", fontsize=12, fontweight='medium')
    ax1.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax1.set_title("Forecast Consistency Comparison", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # --- Top Right: Consistency Rates ---
    ax2 = fig.add_subplot(gs[0, 1])
    k_consistent = sum(1 for r in kaikoura_results if r["consistent"])
    c_consistent = sum(1 for r in canterbury_results if r["consistent"])
    
    categories = ['Kaikoura', 'Canterbury']
    consistent = [k_consistent, c_consistent]
    total = [len(kaikoura_results), len(canterbury_results)]
    inconsistent = [t - c for t, c in zip(total, consistent)]
    
    x = np.arange(len(categories))
    width = 0.5
    
    p1 = ax2.bar(x, consistent, width, label='Consistent', color=COLORS['consistent'], alpha=0.8)
    p2 = ax2.bar(x, inconsistent, width, bottom=consistent, label='Inconsistent', 
                 color=COLORS['inconsistent'], alpha=0.8)
    
    ax2.set_ylabel('Number of Forecasts', fontsize=12, fontweight='medium')
    ax2.set_title('Consistency Rate Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    # Add percentage labels
    for i, (c, t) in enumerate(zip(consistent, total)):
        pct = 100 * c / t
        ax2.text(i, t + 0.5, f'{pct:.0f}%', ha='center', fontsize=12, fontweight='bold')
    
    # --- Bottom Left: Observed vs Simulated ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(k_days, k_obs, 'o-', color='#E74C3C', linewidth=2, markersize=8,
             label='Kaikoura Observed', markeredgecolor='white', markeredgewidth=2)
    ax3.plot(k_days, k_sim, 's--', color='#E74C3C', linewidth=1.5, markersize=6,
             alpha=0.6, label='Kaikoura Simulated')
    ax3.plot(c_days, c_obs, 'o-', color='#3498DB', linewidth=2, markersize=8,
             label='Canterbury Observed', markeredgecolor='white', markeredgewidth=2)
    ax3.plot(c_days, c_sim, 's--', color='#3498DB', linewidth=1.5, markersize=6,
             alpha=0.6, label='Canterbury Simulated')
    
    ax3.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax3.set_ylabel("Event Count (7-day window)", fontsize=12, fontweight='medium')
    ax3.set_title("Event Count Comparison", fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', ncol=2, fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- Bottom Right: Statistics Table ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    stats_data = [
        ['Total Forecasts', f'{len(kaikoura_results)}', f'{len(canterbury_results)}'],
        ['Consistent', f'{k_consistent} ({100*k_consistent/len(kaikoura_results):.0f}%)',
         f'{c_consistent} ({100*c_consistent/len(canterbury_results):.0f}%)'],
        ['Mean Quantile', f'{np.mean(k_quantiles):.3f}', f'{np.mean(c_quantiles):.3f}'],
        ['Mean Obs Count', f'{np.mean(k_obs):.1f}', f'{np.mean(c_obs):.1f}'],
        ['Mean Sim Count', f'{np.mean(k_sim):.1f}', f'{np.mean(c_sim):.1f}'],
    ]
    
    table = ax4.table(cellText=stats_data, colLabels=['Metric', 'Kaikoura', 'Canterbury'],
                      cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(3):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    fig.suptitle("Multi-Sequence Comparison: Kaikoura vs Canterbury", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def calculate_forecast_skill(simulations: pd.DataFrame, observed: pd.DataFrame) -> dict:
    """Calculate forecast skill metrics (Information Gain, Brier Score)."""
    
    n_obs = len(observed)
    
    # Get simulated counts distribution
    sim_counts = simulations.groupby("catalog_id").size()
    
    # Information Gain (log-likelihood ratio vs Poisson reference model)
    mean_rate = sim_counts.mean()
    
    # Likelihood of observation under ETAS forecast
    from scipy.stats import poisson
    etas_likelihood = (sim_counts == n_obs).mean()  # Empirical probability
    if etas_likelihood == 0:
        etas_likelihood = 1 / len(sim_counts)  # Smoothing
    
    # Likelihood under Poisson reference
    poisson_likelihood = poisson.pmf(n_obs, mean_rate)
    
    information_gain = np.log(etas_likelihood / max(poisson_likelihood, 1e-10))
    
    # Brier Score (for probabilistic forecast)
    # Discretize into bins for probability forecast
    bins = np.arange(0, sim_counts.max() + 20, 20)
    bin_probs, _ = np.histogram(sim_counts, bins=bins, density=True)
    bin_probs = bin_probs / bin_probs.sum()  # Normalize
    
    # Find which bin observed falls into
    obs_bin_idx = np.digitize(n_obs, bins) - 1
    obs_bin_idx = min(max(obs_bin_idx, 0), len(bin_probs) - 1)
    
    # Brier score = mean squared error of probabilities
    outcomes = np.zeros(len(bin_probs))
    outcomes[obs_bin_idx] = 1
    brier_score = np.mean((bin_probs - outcomes) ** 2)
    
    return {
        "information_gain": information_gain,
        "brier_score": brier_score,
        "mean_rate": mean_rate,
        "etas_likelihood": etas_likelihood,
        "poisson_likelihood": poisson_likelihood
    }


def plot_publication_multipanel(params_df: pd.DataFrame, n_test_results: list,
                                sequence: str, dates: list, output_path: str = None):
    """Create publication-ready multi-panel figure combining key results."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25,
                  height_ratios=[1, 1, 1.2])
    
    mainshock = min(dates)
    days = [(d - mainshock).total_seconds() / 86400 for d in dates]
    
    # --- Panel A: Parameter Evolution (2 key parameters) ---
    ax1 = fig.add_subplot(gs[0, :])
    if "log10_k0" in params_df.columns and "omega" in params_df.columns:
        param_days = np.array([(d - mainshock).total_seconds() / 86400 
                               for d in params_df["date"]])
        
        ax1_twin = ax1.twinx()
        
        # Productivity
        l1 = ax1.plot(param_days, params_df["log10_k0"], 'o-', color='#2E86AB',
                     linewidth=2, markersize=7, label=r'$\log_{10}(k_0)$ (Productivity)',
                     markeredgecolor='white', markeredgewidth=1.5)
        ax1.set_ylabel(r'$\log_{10}(k_0)$', fontsize=12, fontweight='medium', color='#2E86AB')
        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        
        # Omori exponent
        l2 = ax1_twin.plot(param_days, params_df["omega"], 's-', color='#F18F01',
                          linewidth=2, markersize=7, label=r'$\omega$ (Omori p-1)',
                          markeredgecolor='white', markeredgewidth=1.5)
        ax1_twin.set_ylabel(r'$\omega$', fontsize=12, fontweight='medium', color='#F18F01')
        ax1_twin.tick_params(axis='y', labelcolor='#F18F01')
        
        ax1.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
        ax1.set_title("(A) Parameter Evolution", fontsize=13, fontweight='bold', loc='left')
        
        # Combined legend
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis to show all data with padding
        ax1.set_xlim(-0.5, max(param_days) + 0.5)
    
    # --- Panel B: N-Test Quantiles ---
    ax2 = fig.add_subplot(gs[1, 0])
    quantiles = [r["quantile"] for r in n_test_results]
    consistent_mask = [r["consistent"] for r in n_test_results]
    
    ax2.axhspan(0.025, 0.975, alpha=0.15, color=COLORS['consistent'])
    for i, (d, q, consistent) in enumerate(zip(days, quantiles, consistent_mask)):
        color = COLORS['consistent'] if consistent else COLORS['inconsistent']
        ax2.scatter(d, q, c=color, s=120, edgecolor='white', linewidth=2, zorder=3)
    ax2.plot(days, quantiles, 'k-', alpha=0.3, linewidth=1, zorder=2)
    
    ax2.set_ylabel("N-Test Quantile", fontsize=12, fontweight='medium')
    ax2.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax2.set_title("(B) Forecast Consistency", fontsize=13, fontweight='bold', loc='left')
    ax2.set_xlim(-0.5, max(days) + 0.5)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    # --- Panel C: Observed vs Simulated ---
    ax3 = fig.add_subplot(gs[1, 1])
    observed = [r["observed"] for r in n_test_results]
    simulated_median = [r["simulated_median"] for r in n_test_results]
    simulated_p5 = [r["p5"] for r in n_test_results]
    simulated_p95 = [r["p95"] for r in n_test_results]
    
    ax3.fill_between(days, simulated_p5, simulated_p95, alpha=0.25, 
                     color=COLORS['simulated'], label='90% Prediction Interval')
    ax3.plot(days, simulated_median, 'o-', color=COLORS['simulated'], 
            linewidth=2, markersize=8, label='Simulated Median')
    ax3.plot(days, observed, 's-', color=COLORS['observed'], 
            linewidth=2, markersize=8, label='Observed')
    
    ax3.set_ylabel("Event Count (7-day window)", fontsize=12, fontweight='medium')
    ax3.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax3.set_title("(C) Event Count Comparison", fontsize=13, fontweight='bold', loc='left')
    ax3.set_xlim(-0.5, max(days) + 0.5)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # --- Panel D: Summary Table ---
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    n_consistent = sum(consistent_mask)
    n_total = len(consistent_mask)
    
    summary_text = f"""
Key Results for {sequence} Sequence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Forecast Performance:
  • Consistency Rate: {n_consistent}/{n_total} ({100*n_consistent/n_total:.0f}%) of forecasts pass N-test
  • Mean Quantile: {np.mean(quantiles):.3f} (expected: 0.5 for unbiased forecast)
  • Median Quantile: {np.median(quantiles):.3f}

Event Count Statistics:
  • Total Observed Events: {sum(observed)} (mean: {np.mean(observed):.1f} per window)
  • Total Simulated Events: {np.mean(simulated_median) * len(simulated_median):.0f} (mean: {np.mean(simulated_median):.1f} per window)
  • Bias: {(np.mean(observed) - np.mean(simulated_median)) / np.mean(simulated_median) * 100:+.1f}%

Parameter Stability:
  • Productivity (log₁₀k₀) range: [{params_df['log10_k0'].min():.3f}, {params_df['log10_k0'].max():.3f}]
  • Omori exponent (ω) range: [{params_df['omega'].min():.3f}, {params_df['omega'].max():.3f}]
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(f"ETAS Forecast Evaluation: {sequence} Earthquake Sequence", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_spatial_coverage(sequence: str, catalog: pd.DataFrame, 
                          polygon_path: str = "../input_data/nz_polygon.npy",
                          output_path: str = None):
    """Create spatial coverage map showing NZ polygon and sequence epicenters."""
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Load polygon
    try:
        polygon = np.load(polygon_path, allow_pickle=True)
        
        # Plot polygon boundary
        ax.plot(polygon[:, 1], polygon[:, 0], 'k-', linewidth=2, 
                label='Forecast Region Boundary', zorder=2)
        ax.fill(polygon[:, 1], polygon[:, 0], color='lightblue', 
                alpha=0.2, zorder=1)
        
        # Get sequence events
        if sequence == "Kaikoura":
            mainshock_date = datetime(2016, 11, 13, 11, 2, 56)
        else:  # Canterbury
            mainshock_date = datetime(2010, 9, 4, 4, 35, 43)
        
        start = mainshock_date - timedelta(days=1)
        end = mainshock_date + timedelta(days=30)
        
        seq_events = catalog[(catalog["time"] >= start) & (catalog["time"] <= end)]
        
        # Plot all sequence events
        sizes = (seq_events["magnitude"] - 3) ** 2.5 * 15
        sc = ax.scatter(seq_events["longitude"], seq_events["latitude"],
                       c=seq_events["magnitude"], s=sizes, cmap='YlOrRd',
                       edgecolor='black', linewidth=0.5, alpha=0.7,
                       vmin=4.0, vmax=seq_events["magnitude"].max(), zorder=3)
        
        # Highlight mainshock
        mainshock = seq_events.loc[seq_events["magnitude"].idxmax()]
        ax.scatter(mainshock["longitude"], mainshock["latitude"],
                  marker='*', s=800, c='red', edgecolor='black',
                  linewidth=2, zorder=4, label=f'Mainshock M{mainshock["magnitude"]:.1f}')
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("Magnitude", fontsize=11)
        
        # Labels and title
        ax.set_xlabel("Longitude (°E)", fontsize=12, fontweight='medium')
        ax.set_ylabel("Latitude (°S)", fontsize=12, fontweight='medium')
        ax.set_title(f"Spatial Coverage: {sequence} Sequence\n"
                    f"Region Boundary & {len(seq_events)} Events (M ≥ 4.0)", 
                    fontsize=14, fontweight='bold', pad=15)
        
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # Set limits based on polygon
        margin = 1.0
        ax.set_xlim(polygon[:, 1].min() - margin, polygon[:, 1].max() + margin)
        ax.set_ylim(polygon[:, 0].min() - margin, polygon[:, 0].max() + margin)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error loading polygon: {e}", 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def create_html_dashboard(output_dir: str = "figures"):
    """Generate interactive HTML dashboard with all plots organized."""
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETAS Forecast Evaluation Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            overflow-x: auto;
        }
        
        .tab {
            padding: 15px 25px;
            cursor: pointer;
            background: #e9ecef;
            border: none;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
            white-space: nowrap;
        }
        
        .tab:hover {
            background: #d6d8db;
        }
        
        .tab.active {
            background: white;
            color: #2980b9;
            border-bottom: 3px solid #2980b9;
        }
        
        .content {
            padding: 30px;
            display: none;
        }
        
        .content.active {
            display: block;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }
        
        .plot-card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .plot-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        }
        
        .plot-card img {
            width: 100%;
            display: block;
            cursor: pointer;
        }
        
        .plot-title {
            padding: 15px;
            background: #f8f9fa;
            font-weight: 600;
            border-top: 3px solid #2980b9;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
        }
        
        .modal-content {
            margin: 2% auto;
            display: block;
            max-width: 95%;
            max-height: 95%;
        }
        
        .close {
            position: absolute;
            top: 30px;
            right: 50px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .stats {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .stats h3 {
            color: #1e3a5f;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌏 ETAS Forecast Evaluation Dashboard</h1>
            <p>Comprehensive Analysis of New Zealand Earthquake Sequences</p>
        </header>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">📊 Overview</button>
            <button class="tab" onclick="showTab('kaikoura')">🏔️ Kaikoura</button>
            <button class="tab" onclick="showTab('canterbury')">🏙️ Canterbury</button>
            <button class="tab" onclick="showTab('comparison')">⚖️ Comparison</button>
            <button class="tab" onclick="showTab('details')">🔬 Detailed Analysis</button>
        </div>
        
        <div id="overview" class="content active">
            <div class="stats">
                <h3>Summary Statistics</h3>
                <p><strong>Sequences Analyzed:</strong> Kaikoura (2016) & Canterbury (2010)</p>
                <p><strong>Total Forecast Windows:</strong> 24 (10 Kaikoura + 14 Canterbury)</p>
                <p><strong>Overall Consistency:</strong> 77% of forecasts pass N-test</p>
            </div>
            
            <div class="plot-grid">
                <div class="plot-card">
                    <div class="plot-title">Multi-Sequence Comparison</div>
                    <img src="multi_sequence_comparison.png" onclick="openModal(this.src)" alt="Multi-Sequence Comparison">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Publication Figure (Kaikoura)</div>
                    <img src="publication_kaikoura.png" onclick="openModal(this.src)" alt="Publication Kaikoura">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Publication Figure (Canterbury)</div>
                    <img src="publication_canterbury.png" onclick="openModal(this.src)" alt="Publication Canterbury">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Spatial Coverage (Kaikoura)</div>
                    <img src="spatial_coverage_kaikoura.png" onclick="openModal(this.src)" alt="Spatial Coverage Kaikoura">
                </div>
            </div>
        </div>
        
        <div id="kaikoura" class="content">
            <h2 style="margin-bottom: 20px;">Kaikoura Sequence Analysis</h2>
            <div class="plot-grid">
                <div class="plot-card">
                    <div class="plot-title">Parameter Evolution</div>
                    <img src="parameter_evolution_kaikoura.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">N-Test Summary</div>
                    <img src="ntest_summary_kaikoura.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Parameter Correlation</div>
                    <img src="param_corr_kaikoura.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Summary Table</div>
                    <img src="summary_table_kaikoura.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Temporal Residuals</div>
                    <img src="residuals_kaikoura.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Magnitude-Time Evolution</div>
                    <img src="mag_time_kaikoura.png" onclick="openModal(this.src)">
                </div>
            </div>
        </div>
        
        <div id="canterbury" class="content">
            <h2 style="margin-bottom: 20px;">Canterbury Sequence Analysis</h2>
            <div class="plot-grid">
                <div class="plot-card">
                    <div class="plot-title">Parameter Evolution</div>
                    <img src="parameter_evolution_canterbury.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">N-Test Summary</div>
                    <img src="ntest_summary_canterbury.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Parameter Correlation</div>
                    <img src="param_corr_canterbury.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Summary Table</div>
                    <img src="summary_table_canterbury.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Temporal Residuals</div>
                    <img src="residuals_canterbury.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Magnitude-Time Evolution</div>
                    <img src="mag_time_canterbury.png" onclick="openModal(this.src)">
                </div>
            </div>
        </div>
        
        <div id="comparison" class="content">
            <h2 style="margin-bottom: 20px;">Comparative Analysis</h2>
            <div class="plot-grid">
                <div class="plot-card">
                    <div class="plot-title">Multi-Sequence Comparison</div>
                    <img src="multi_sequence_comparison.png" onclick="openModal(this.src)">
                </div>
            </div>
        </div>
        
        <div id="details" class="content">
            <h2 style="margin-bottom: 20px;">Detailed Model-by-Model Analysis</h2>
            <h3>Kaikoura - First 3 Models</h3>
            <div class="plot-grid">
                <div class="plot-card">
                    <div class="plot-title">N-Test (Model 0)</div>
                    <img src="ntest_kaikoura_0.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Spatial Comparison (Model 0)</div>
                    <img src="spatial_kaikoura_0.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">M-Test (Model 0)</div>
                    <img src="mtest_kaikoura_0.png" onclick="openModal(this.src)">
                </div>
                <div class="plot-card">
                    <div class="plot-title">Rate Evolution (Model 0)</div>
                    <img src="rate_kaikoura_0.png" onclick="openModal(this.src)">
                </div>
            </div>
        </div>
    </div>
    
    <div id="modal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>
    
    <script>
        function showTab(tabName) {
            const contents = document.querySelectorAll('.content');
            const tabs = document.querySelectorAll('.tab');
            
            contents.forEach(content => content.classList.remove('active'));
            tabs.forEach(tab => tab.classList.remove('active'));
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        function openModal(src) {
            document.getElementById('modal').style.display = 'block';
            document.getElementById('modalImg').src = src;
        }
        
        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }
    </script>
</body>
</html>
"""
    
    output_path = os.path.join(output_dir, "dashboard.html")
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Saved: {output_path}")
    print(f"\n🌐 Open {output_path} in your browser to view the interactive dashboard!")
    
    return output_path


def plot_magnitude_dependent_ntests(n_test_results_by_mag: dict, sequence: str,
                                     dates: list, output_path: str = None):
    """Plot N-test results separated by magnitude threshold."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    mainshock = min(dates)
    days = [(d - mainshock).total_seconds() / 86400 for d in dates]
    
    # Color scheme for different magnitude thresholds
    mag_colors = {
        4.1: '#3498DB',
        4.5: '#2ECC71',
        5.0: '#F39C12',
        5.5: '#E74C3C'
    }
    
    # --- Top: Quantile Evolution ---
    ax1.axhspan(0.025, 0.975, alpha=0.15, color=COLORS['consistent'])
    
    for mag_thresh, results in sorted(n_test_results_by_mag.items()):
        quantiles = [r["quantile"] for r in results]
        color = mag_colors.get(mag_thresh, '#95A5A6')
        ax1.plot(days[:len(quantiles)], quantiles, 'o-', color=color, linewidth=2,
                markersize=6, label=f'M ≥ {mag_thresh}', alpha=0.8)
    
    ax1.set_ylabel("N-Test Quantile", fontsize=12, fontweight='medium')
    ax1.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax1.set_title("Magnitude-Dependent Forecast Consistency", fontsize=13, fontweight='bold')
    ax1.legend(loc='best', ncol=2)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # --- Bottom: Consistency Rate ---
    mag_thresholds = sorted(n_test_results_by_mag.keys())
    consistency_rates = []
    total_forecasts = []
    
    for mag_thresh in mag_thresholds:
        results = n_test_results_by_mag[mag_thresh]
        n_consistent = sum(1 for r in results if r["consistent"])
        consistency_rates.append(100 * n_consistent / len(results))
        total_forecasts.append(len(results))
    
    x = np.arange(len(mag_thresholds))
    bars = ax2.bar(x, consistency_rates, color=[mag_colors.get(m, '#95A5A6') for m in mag_thresholds],
                   alpha=0.7, edgecolor='white', linewidth=2)
    
    # Add value labels
    for i, (rate, total) in enumerate(zip(consistency_rates, total_forecasts)):
        ax2.text(i, rate + 2, f'{rate:.0f}%\n(n={total})', 
                ha='center', fontsize=10, fontweight='bold')
    
    ax2.axhline(95, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='95% Target')
    ax2.set_ylabel("Consistency Rate (%)", fontsize=12, fontweight='medium')
    ax2.set_xlabel("Magnitude Threshold", fontsize=12, fontweight='medium')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'M ≥ {m}' for m in mag_thresholds])
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"Magnitude-Dependent N-Test Analysis: {sequence} Sequence",
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_information_gain_timeline(ig_results: list, sequence: str, 
                                   dates: list, output_path: str = None):
    """Plot Information Gain evolution over time."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    mainshock = min(dates)
    days = [(d - mainshock).total_seconds() / 86400 for d in dates]
    
    information_gains = [r["information_gain"] for r in ig_results]
    brier_scores = [r["brier_score"] for r in ig_results]
    
    # --- Top: Information Gain ---
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.plot(days, information_gains, 'o-', color=COLORS['primary'], linewidth=2.5,
            markersize=8, markeredgecolor='white', markeredgewidth=2, label='Information Gain')
    
    # Add trend line
    if len(days) > 5:
        z = np.polyfit(days, information_gains, 2)
        p = np.poly1d(z)
        ax1.plot(days, p(days), '--', color='gray', linewidth=2, alpha=0.5, label='Trend (quadratic)')
    
    # Shade positive/negative regions
    ax1.fill_between(days, 0, information_gains, 
                     where=[ig >= 0 for ig in information_gains],
                     alpha=0.2, color='green', label='Positive IG (Better than Poisson)')
    ax1.fill_between(days, 0, information_gains,
                     where=[ig < 0 for ig in information_gains],
                     alpha=0.2, color='red', label='Negative IG (Worse than Poisson)')
    
    ax1.set_ylabel("Information Gain (nats)", fontsize=12, fontweight='medium')
    ax1.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax1.set_title("Forecast Skill Evolution", fontsize=13, fontweight='bold')
    ax1.legend(loc='best', ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Annotate key points
    max_ig_idx = np.argmax(information_gains)
    ax1.annotate(f'Peak IG: {information_gains[max_ig_idx]:.2f}\nDay {days[max_ig_idx]:.0f}',
                xy=(days[max_ig_idx], information_gains[max_ig_idx]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # --- Bottom: Brier Score ---
    ax2.plot(days, brier_scores, 's-', color=COLORS['warning'], linewidth=2.5,
            markersize=8, markeredgecolor='white', markeredgewidth=2)
    ax2.set_ylabel("Brier Score\n(lower = better)", fontsize=12, fontweight='medium')
    ax2.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax2.set_title("Probabilistic Forecast Accuracy", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    fig.suptitle(f"Forecast Skill Metrics: {sequence} Sequence",
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def calculate_spatial_ltest(simulations: pd.DataFrame, observed: pd.DataFrame) -> dict:
    """Calculate CSEP-standard spatial L-test using kernel density estimation."""
    
    try:
        from scipy.stats import gaussian_kde
        
        # Create KDE from simulations
        if len(simulations) < 10:
            return {"l_test_stat": np.nan, "spatial_ll": np.nan, "n_obs": len(observed)}
        
        # Subsample simulations for computational efficiency
        if len(simulations) > 10000:
            sim_sample = simulations.sample(n=10000)
        else:
            sim_sample = simulations
        
        # Build KDE
        sim_points = np.vstack([sim_sample["longitude"], sim_sample["latitude"]])
        kde = gaussian_kde(sim_points)
        
        # Evaluate log-likelihood for each observed event
        if len(observed) == 0:
            return {"l_test_stat": np.nan, "spatial_ll": np.nan, "n_obs": 0}
        
        obs_points = np.vstack([observed["longitude"], observed["latitude"]])
        log_likelihoods = kde.logpdf(obs_points)
        
        # Total spatial log-likelihood
        spatial_ll = np.sum(log_likelihoods)
        
        # L-test statistic (average log-likelihood per event)
        l_test_stat = spatial_ll / len(observed) if len(observed) > 0 else np.nan
        
        return {
            "l_test_stat": l_test_stat,
            "spatial_ll": spatial_ll,
            "n_obs": len(observed),
            "mean_log_density": np.mean(log_likelihoods),
            "std_log_density": np.std(log_likelihoods)
        }
    
    except Exception as e:
        return {"l_test_stat": np.nan, "spatial_ll": np.nan, "n_obs": len(observed), "error": str(e)}


def plot_spatial_ltest_results(ltest_results: list, sequence: str, 
                               dates: list, output_path: str = None):
    """Plot spatial L-test results over time."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    mainshock = min(dates)
    days = [(d - mainshock).total_seconds() / 86400 for d in dates]
    
    l_stats = [r["l_test_stat"] for r in ltest_results]
    spatial_lls = [r["spatial_ll"] for r in ltest_results]
    
    # --- Top: L-test Statistic ---
    ax1.plot(days, l_stats, 'o-', color=COLORS['secondary'], linewidth=2.5,
            markersize=8, markeredgecolor='white', markeredgewidth=2)
    ax1.set_ylabel("L-Test Statistic\n(higher = better fit)", fontsize=12, fontweight='medium')
    ax1.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax1.set_title("Spatial Forecast Quality (Mean Log-Likelihood per Event)", 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add reference line at typical value
    median_l = np.nanmedian(l_stats)
    ax1.axhline(median_l, color='gray', linestyle='--', alpha=0.5, label=f'Median: {median_l:.2f}')
    ax1.legend()
    
    # --- Bottom: Total Spatial Log-Likelihood ---
    ax2.plot(days, spatial_lls, 's-', color=COLORS['simulated'], linewidth=2.5,
            markersize=8, markeredgecolor='white', markeredgewidth=2)
    ax2.set_ylabel("Total Spatial Log-Likelihood", fontsize=12, fontweight='medium')
    ax2.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax2.set_title("Cumulative Spatial Fit Quality", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f"Spatial L-Test Analysis: {sequence} Sequence",
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_adaptive_vs_fixed_comparison(adaptive_results: list, fixed_results: list,
                                      sequence: str, dates: list, output_path: str = None,
                                      horizon_days: float = FORECAST_DAYS):
    """Compare adaptive parameter updating vs fixed parameters."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    mainshock = min(dates)
    days = [(d - mainshock).total_seconds() / 86400 for d in dates]
    
    # Extract data
    adaptive_q = [r["quantile"] for r in adaptive_results]
    fixed_q = [r["quantile"] for r in fixed_results]
    
    adaptive_obs = [r["observed"] for r in adaptive_results]
    adaptive_sim = [r["simulated_median"] for r in adaptive_results]
    fixed_sim = [r["simulated_median"] for r in fixed_results]
    
    # --- Panel A: Quantile Comparison ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axhspan(0.025, 0.975, alpha=0.15, color=COLORS['consistent'])
    ax1.plot(days, adaptive_q, 'o-', color='#2ECC71', linewidth=2.5, markersize=8,
            label='Adaptive Parameters', markeredgecolor='white', markeredgewidth=2)
    ax1.plot(days, fixed_q, 's-', color='#E74C3C', linewidth=2.5, markersize=8,
            label='Fixed Regional Parameters', markeredgecolor='white', markeredgewidth=2)
    ax1.set_ylabel("N-Test Quantile", fontsize=12, fontweight='medium')
    ax1.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax1.set_title("(A) Forecast Consistency: Adaptive vs Fixed", fontsize=13, fontweight='bold', loc='left')
    ax1.legend(loc='best', fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel B: Consistency Rate ---
    ax2 = fig.add_subplot(gs[1, 0])
    adaptive_consistent = sum(1 for r in adaptive_results if r["consistent"])
    fixed_consistent = sum(1 for r in fixed_results if r["consistent"])
    
    categories = ['Adaptive', 'Fixed']
    consistent = [adaptive_consistent, fixed_consistent]
    total = [len(adaptive_results), len(fixed_results)]
    inconsistent = [t - c for t, c in zip(total, consistent)]
    
    x = np.arange(len(categories))
    width = 0.5
    
    p1 = ax2.bar(x, consistent, width, label='Consistent', color='#2ECC71', alpha=0.8)
    p2 = ax2.bar(x, inconsistent, width, bottom=consistent, label='Inconsistent',
                color='#E74C3C', alpha=0.8)
    
    ax2.set_ylabel('Number of Forecasts', fontsize=12, fontweight='medium')
    ax2.set_title('(B) Consistency Rate Comparison', fontsize=13, fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    for i, (c, t) in enumerate(zip(consistent, total)):
        pct = 100 * c / t
        ax2.text(i, t + 0.5, f'{pct:.0f}%', ha='center', fontsize=12, fontweight='bold')
    
    # --- Panel C: Event Count Prediction ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(days, adaptive_obs, 'o-', color='black', linewidth=2, markersize=6,
            label='Observed', zorder=3)
    ax3.plot(days, adaptive_sim, 's-', color='#2ECC71', linewidth=2, markersize=6,
            label='Adaptive Forecast', alpha=0.7)
    ax3.plot(days, fixed_sim, '^-', color='#E74C3C', linewidth=2, markersize=6,
            label='Fixed Forecast', alpha=0.7)
    ax3.set_ylabel(f"Event Count ({horizon_days:g}-day window)", fontsize=12, fontweight='medium')
    ax3.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax3.set_title("(C) Forecast Accuracy Comparison", fontsize=13, fontweight='bold', loc='left')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # --- Panel D: Improvement Metric ---
    ax4 = fig.add_subplot(gs[2, :])
    
    # Calculate improvement (difference in absolute error)
    adaptive_error = [abs(obs - sim) for obs, sim in zip(adaptive_obs, adaptive_sim)]
    fixed_error = [abs(obs - sim) for obs, sim in zip(adaptive_obs, fixed_sim)]
    improvement = [fixed - adapt for fixed, adapt in zip(fixed_error, adaptive_error)]
    
    colors_imp = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvement]
    ax4.bar(days, improvement, width=max(days)/len(days)*0.8, color=colors_imp, alpha=0.7,
           edgecolor='white', linewidth=1)
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel("Improvement\n(Fixed Error - Adaptive Error)", fontsize=12, fontweight='medium')
    ax4.set_xlabel("Days After Mainshock", fontsize=12, fontweight='medium')
    ax4.set_title("(D) Adaptive Parameter Benefit (positive = better)", 
                 fontsize=13, fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Summary stats
    mean_improvement = np.mean(improvement)
    pct_better = 100 * sum(1 for i in improvement if i > 0) / len(improvement)
    ax4.text(0.02, 0.98, f'Mean Improvement: {mean_improvement:+.1f} events\n' +
                          f'Adaptive Better: {pct_better:.0f}% of time',
            transform=ax4.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(
                f"Adaptive vs Fixed Parameter Forecasting: {sequence} Sequence "
                f"({horizon_days:g}-day windows)",
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)
    return fig



    """Run full visualization suite for a sequence."""
    
    print(f"\n{'='*60}")
    print(f"ETAS Visualization: {sequence} Sequence")
    print(f"{'='*60}\n")
    
    params_df = load_parameters(sequence)
    print(f"Loaded {len(params_df)} parameter sets")
    
    catalog = load_catalog()
    print(f"Catalog has {len(catalog)} events")
    
    dates = KAIKOURA_DATES if sequence == "Kaikoura" else CANTERBURY_DATES
    
    if max_models:
        params_df = params_df.head(max_models)
        dates = dates[:max_models]
    
    # 0. Magnitude-Time Plot (NEW)
    print("\n0. Creating Magnitude-Time plot...")
    plot_magnitude_time(catalog, sequence, [r["date"] for _, r in params_df.iterrows() if r["date"]],
                        os.path.join(OUTPUT_DIR, f"mag_time_{sequence.lower()}.png"))
    
    # 1. Parameter Evolution
    print("\n1. Creating parameter evolution plot...")
    plot_parameter_evolution(params_df, sequence,
        os.path.join(OUTPUT_DIR, f"parameter_evolution_{sequence.lower()}.png"))
    
    # 2. N-Tests
    print("\n2. Running N-tests...")
    n_test_results = []
    
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
        n_test_results.append(result)
        
        status = "✓" if result["consistent"] else "✗"
        print(f"  Model {model_idx}: Obs={result['observed']}, "
              f"Sim={result['simulated_mean']:.1f}±{result['simulated_std']:.1f}, "
              f"q={result['quantile']:.3f} {status}")
    
    # 3. N-Test summary
    if n_test_results:
        print("\n3. Creating N-test summary...")
        plot_n_test_summary(n_test_results, sequence, [r["date"] for r in n_test_results],
            os.path.join(OUTPUT_DIR, f"ntest_summary_{sequence.lower()}.png"))
        
        # 3a. Parameter Correlation Heatmap
        print("\n3a. Creating parameter correlation heatmap...")
        plot_parameter_correlation(params_df, sequence,
            os.path.join(OUTPUT_DIR, f"param_corr_{sequence.lower()}.png"))
        
        # 3b. Summary Statistics Table
        print("\n3b. Creating summary statistics table...")
        plot_summary_table(n_test_results, sequence,
            os.path.join(OUTPUT_DIR, f"summary_table_{sequence.lower()}.png"))
        
        # 3c. Temporal Residuals
        print("\n3c. Creating temporal residual plot...")
        plot_temporal_residuals(n_test_results, sequence, [r["date"] for r in n_test_results],
            os.path.join(OUTPUT_DIR, f"residuals_{sequence.lower()}.png"))
    
    # 4. Detailed plots for first few models
    print("\n4. Creating detailed comparison plots...")
    for result in n_test_results[:3]:
        model_idx = result["model_idx"]
        forecast_start = result["date"]
        forecast_end = forecast_start + timedelta(days=FORECAST_DAYS)
        
        sims = load_simulations(sequence, model_idx)
        observed = get_observed_in_window(catalog, forecast_start, forecast_end)
        
        plot_n_test_result(result, forecast_start, sequence,
            os.path.join(OUTPUT_DIR, f"ntest_{sequence.lower()}_{model_idx}.png"))
        
        plot_cumulative_comparison(sims, observed, forecast_start, sequence,
            os.path.join(OUTPUT_DIR, f"cumulative_{sequence.lower()}_{model_idx}.png"))
        
        plot_spatial_comparison(sims, observed, sequence,
            os.path.join(OUTPUT_DIR, f"spatial_{sequence.lower()}_{model_idx}.png"))
        
        plot_magnitude_comparison(sims, observed, sequence, forecast_start,
            os.path.join(OUTPUT_DIR, f"magnitude_{sequence.lower()}_{model_idx}.png"))
        
        # NEW: M-test with Q-Q plot
        plot_m_test(sims, observed, sequence, forecast_start,
            os.path.join(OUTPUT_DIR, f"mtest_{sequence.lower()}_{model_idx}.png"))
        
        # NEW: Seismicity rate evolution
        plot_seismicity_rate(sims, observed, forecast_start, sequence,
            os.path.join(OUTPUT_DIR, f"rate_{sequence.lower()}_{model_idx}.png"))
    
    # 5. Additional advanced plots
    print("\n5. Creating publication and coverage plots...")
    plot_publication_multipanel(params_df, n_test_results, sequence, 
                               [r["date"] for r in n_test_results],
                               os.path.join(OUTPUT_DIR, f"publication_{sequence.lower()}.png"))
    
    polygon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "input_data", "nz_polygon.npy")
    plot_spatial_coverage(sequence, catalog, polygon_path,
                         os.path.join(OUTPUT_DIR, f"spatial_coverage_{sequence.lower()}.png"))
    
    print(f"\n{'='*60}")
    print(f"Complete! Figures saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")
    
    return params_df, n_test_results


if __name__ == "__main__":
    # Run visualizations for both sequences
    print("\n" + "="*60)
    print("COMPREHENSIVE ETAS FORECAST EVALUATION SUITE")
    print("="*60)
    
    kaikoura_params, kaikoura_results = run_visualization("Kaikoura", max_models=None)  # Show ALL models
    canterbury_params, canterbury_results = run_visualization("Canterbury", max_models=None)  # Show ALL models
    
    # Generate multi-sequence comparison
    print("\n" + "="*60)
    print("Creating Multi-Sequence Comparison")
    print("="*60 + "\n")
    
    plot_multi_sequence_comparison(kaikoura_results, canterbury_results,
                                   os.path.join(OUTPUT_DIR, "multi_sequence_comparison.png"))
    
    # Generate HTML dashboard
    print("\n" + "="*60)
    print("Generating Interactive HTML Dashboard")
    print("="*60 + "\n")
    
    dashboard_path = create_html_dashboard(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print(f"\nGenerated comprehensive ETAS evaluation suite:")
    print(f"  • Parameter evolution plots")
    print(f"  • N-test histograms and summaries")
    print(f"  • Cumulative event comparisons")
    print(f"  • Spatial density maps with confidence contours")
    print(f"  • Magnitude-frequency analysis (M-test with Q-Q plots)")
    print(f"  • Temporal residual plots")
    print(f"  • Seismicity rate evolution")
    print(f"  • Parameter correlation heatmaps")
    print(f"  • Summary statistics tables")
    print(f"  • Publication-ready multi-panel figures")
    print(f"  • Spatial coverage maps")
    print(f"  • Multi-sequence comparison")
    print(f"  • Interactive HTML dashboard")
    print(f"\n📂 All figures saved to: {OUTPUT_DIR}/")
    print(f"🌐 Open dashboard: {dashboard_path}")
    print("="*60 + "\n")

def plot_csep_6panel(simulations: pd.DataFrame, observed: pd.DataFrame, 
                     forecast_config: dict, output_path: str = None):
    """
    Create a publication-quality 6-panel CSEP-style validation plot.
    """
    import matplotlib.patheffects as pe

    # ── Design tokens ──────────────────────────────────────────────────
    _BG = "#F7F9FC"
    _PANEL = "#FFFFFF"
    _GRID = "#E2E8F0"
    _TEXT = "#2D3748"
    _SUB = "#718096"
    _NAVY = "#0D1B2A"
    _SIM_CLR = "#3B82F6"
    _OBS_CLR = "#DC2626"
    _MEAN_CLR = "#6366F1"
    _P95_CLR = "#F59E0B"
    _PASS = "#10B981"
    _FAIL = "#EF4444"

    def _style(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(_PANEL)
        ax.set_title(title, fontsize=13, fontweight="bold", color=_NAVY, pad=10, loc="left")
        if xlabel: ax.set_xlabel(xlabel, fontsize=10, color=_TEXT)
        if ylabel: ax.set_ylabel(ylabel, fontsize=10, color=_TEXT)
        for s in ax.spines.values():
            s.set_color(_GRID); s.set_linewidth(0.8)
        ax.tick_params(colors=_TEXT, labelsize=9)
        ax.grid(True, alpha=0.35, color=_GRID, linewidth=0.6)

    # ── Common data ────────────────────────────────────────────────────
    obs_count = len(observed)
    sim_counts = simulations.groupby("catalog_id").size()
    sim_max_mags = simulations.groupby("catalog_id")["magnitude"].max()
    obs_max_mag = observed["magnitude"].max() if obs_count > 0 else 0
    mc = forecast_config.get("mc", 4.1)

    # ── Figure layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 15), facecolor=_BG)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.12, 1, 1, 1],
                  hspace=0.30, wspace=0.22, left=0.06, right=0.96, top=0.97, bottom=0.04)

    # ── Row 0: Context banner ──────────────────────────────────────────
    ax_ban = fig.add_subplot(gs[0, :])
    ax_ban.set_xlim(0, 1); ax_ban.set_ylim(0, 1); ax_ban.axis("off")
    duration = forecast_config.get("duration_days", "?")
    f_start = str(forecast_config.get("forecast_start", ""))[:10]
    n_sims_val = simulations["catalog_id"].nunique()
    ax_ban.text(0.5, 0.75, "CSEP Forecast Evaluation", ha="center", va="top",
                fontsize=18, fontweight="bold", color=_NAVY)
    meta = f"Horizon: {duration} days  ·  Origin: {f_start}  ·  Mc = {mc}  ·  {n_sims_val} simulations  ·  {obs_count} observed"
    ax_ban.text(0.5, 0.15, meta, ha="center", va="center", fontsize=10, color=_SUB)
    ax_ban.axhline(0.0, color=_GRID, linewidth=1.5)

    # ── Panel 1: N-Test ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    _style(ax1, "N-Test  (Event Count)", xlabel="Number of Events", ylabel="Frequency")
    ax1.hist(sim_counts, bins=30, color=_SIM_CLR, edgecolor="white", linewidth=0.8,
             alpha=0.75, label="Forecast simulations", zorder=3)
    ax1.axvline(obs_count, color=_OBS_CLR, linewidth=2.5, label=f"Observed: {obs_count}", zorder=5)
    ax1.axvline(sim_counts.mean(), color=_MEAN_CLR, linestyle="--", linewidth=1.5,
                label=f"Mean: {sim_counts.mean():.0f}", zorder=4)
    quantile = (sim_counts < obs_count).mean()
    status = "PASS" if 0.025 < quantile < 0.975 else "FAIL"
    badge_clr = _PASS if status == "PASS" else _FAIL
    ax1.text(0.98, 0.95, f"q = {quantile:.3f}\n{status}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=11, fontweight="bold", color=badge_clr,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=badge_clr, alpha=0.95))
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor=_GRID, fontsize=9)

    # ── Panel 2: M-Test ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    _style(ax2, "M-Test  (Maximum Magnitude)", xlabel="Maximum Magnitude", ylabel="Frequency")
    ax2.hist(sim_max_mags, bins=20, color=_P95_CLR, edgecolor="white", linewidth=0.8,
             alpha=0.75, label="Forecast simulations", zorder=3)
    ax2.axvline(obs_max_mag, color=_SIM_CLR, linewidth=2.5,
                label=f"Observed: M{obs_max_mag:.1f}", zorder=5)
    p95 = sim_max_mags.quantile(0.95)
    ax2.axvline(p95, color=_OBS_CLR, linestyle="--", linewidth=1.5, label="95th percentile", zorder=4)
    delta_1 = ((sim_max_mags < obs_max_mag).sum()) / len(sim_max_mags)
    delta_2 = ((sim_max_mags <= obs_max_mag).sum()) / len(sim_max_mags)
    delta = (delta_1 + delta_2) / 2
    m_status = "PASS" if 0.025 < delta < 0.975 else "FAIL"
    m_badge = _PASS if m_status == "PASS" else _FAIL
    ax2.text(0.98, 0.95, f"δ = {delta:.3f}\n{m_status}", transform=ax2.transAxes,
             ha="right", va="top", fontsize=11, fontweight="bold", color=m_badge,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=m_badge, alpha=0.95))
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor=_GRID, fontsize=9)

    # ── Panel 3: Magnitude Distribution ────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    _style(ax3, "Magnitude Distribution", xlabel="Magnitude", ylabel="Probability Density")
    mag_bins = np.arange(mc, 8.0, 0.2)
    ax3.hist(simulations["magnitude"], bins=mag_bins, density=True, alpha=0.55,
             color=_SIM_CLR, edgecolor="white", linewidth=0.6,
             label=f"Forecast ({n_sims_val} sims)", zorder=3)
    if obs_count > 0:
        ax3.hist(observed["magnitude"], bins=mag_bins, density=True, histtype="step",
                 linewidth=2.5, color=_OBS_CLR, label="Observed", zorder=5)
    ax3.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor=_GRID, fontsize=9)

    # Spatial Setup
    # Calculate extent from data
    all_lons = pd.concat([simulations["longitude"], observed["longitude"]])
    all_lats = pd.concat([simulations["latitude"], observed["latitude"]])
    lon_margin = 0.5
    lat_margin = 0.5
    
    # Data extent in standard coordinates
    lon_min, lon_max = all_lons.min()-lon_margin, all_lons.max()+lon_margin
    lat_min, lat_max = all_lats.min()-lat_margin, all_lats.max()+lat_margin
    central_lon = (lon_min + lon_max) / 2  # Center on data
    
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
        # Center projection on data to handle near-antimeridian regions (e.g. NZ)
        proj = ccrs.PlateCarree(central_longitude=central_lon)
        data_crs = ccrs.PlateCarree()  # Data is always in standard geographic coordinates
        # Extent in standard geographic coordinates (passed with explicit crs)
        extent = [lon_min, lon_max, lat_min, lat_max]
    except ImportError:
        has_cartopy = False
        proj = None
        data_crs = None
        extent = [lon_min, lon_max, lat_min, lat_max]

    # Region boundary
    region_border = None
    if "shape_coords" in forecast_config:
        try:
            rc = forecast_config["shape_coords"]
            if isinstance(rc, str) and os.path.exists(rc) and rc.endswith('.npy'):
                region_border = np.load(rc)
            elif isinstance(rc, np.ndarray):
                region_border = rc
            elif isinstance(rc, list):
                region_border = np.array(rc)
        except Exception:
            pass

    # Helper for map setup
    def setup_map(ax):
        if has_cartopy:
             ax.add_feature(cfeature.LAND, facecolor='#F0F0F0', alpha=0.7)
             ax.add_feature(cfeature.OCEAN, facecolor='#E8F4FD', alpha=0.4)
             ax.add_feature(cfeature.COASTLINE, linewidth=1.0, color='#4A5568')
             ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, color='#A0AEC0')
             try:
                 gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='#CBD5E0', alpha=0.4, linestyle='--')
                 gl.top_labels = False
                 gl.right_labels = False
             except: pass
        else:
             ax.set_facecolor('#FAFAFA')
             ax.grid(True, linestyle="--", alpha=0.3, color='gray')
             ax.set_aspect('equal')
             ax.set_xlim(extent[0], extent[1])
             ax.set_ylim(extent[2], extent[3])
             ax.set_xlabel("Longitude")
             ax.set_ylabel("Latitude")
             for spine in ax.spines.values():
                 spine.set_linewidth(1.0)
                 spine.set_color('#CBD5E0')
        
        # Plot region border
        if region_border is not None:
            poly_closed = np.vstack([region_border, region_border[0]])
            if has_cartopy:
                ax.plot(poly_closed[:, 1], poly_closed[:, 0], '-', color='#2D3748',
                        linewidth=1.8, label='Region', transform=data_crs)
            else:
                ax.plot(poly_closed[:, 1], poly_closed[:, 0], '-', color='#2D3748',
                        linewidth=1.8, label='Region')

    # Panel 4: Combined Spatial Distribution
    try:
        if has_cartopy:
            ax4 = fig.add_subplot(gs[2, 1], projection=proj)
        else:
            raise ImportError("Manual Fallback")
    except Exception as e:
        print(f"Comparison map fallback due to: {e}")
        has_cartopy = False
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.set_xlim(extent[0], extent[1])
        ax4.set_ylim(extent[2], extent[3])
    
    setup_map(ax4)
    if has_cartopy:
        ax4.set_extent(extent, crs=data_crs)

    sample_sim = simulations.sample(n=min(5000, len(simulations)))
    sim_sizes = 2 * (10 ** (sample_sim["magnitude"] - mc))
    sim_sizes = np.clip(sim_sizes, 2, 100)
    
    if has_cartopy:
        ax4.scatter(sample_sim["longitude"], sample_sim["latitude"], s=sim_sizes, alpha=0.15,
                    color=_SIM_CLR, label="Forecast sample", transform=data_crs, zorder=3)
    else:
        ax4.scatter(sample_sim["longitude"], sample_sim["latitude"], s=sim_sizes, alpha=0.15,
                    color=_SIM_CLR, label="Forecast sample")

    obs_sizes = 5 * (10 ** (observed["magnitude"] - mc))
    obs_sizes = np.clip(obs_sizes, 10, 300)

    if has_cartopy:
        ax4.scatter(observed["longitude"], observed["latitude"], s=obs_sizes, alpha=0.9,
                    color=_OBS_CLR, edgecolors='white', linewidth=0.8,
                    label=f"Observed ({obs_count})", transform=data_crs, zorder=5)
    else:
        ax4.scatter(observed["longitude"], observed["latitude"], s=obs_sizes, alpha=0.9,
                    color=_OBS_CLR, edgecolors='white', linewidth=0.8,
                    label=f"Observed ({obs_count})")
    
    ax4.set_title("Combined Spatial Distribution", fontsize=12, fontweight="bold",
                  color=_NAVY, pad=8)
    ax4.legend(loc='lower left', fontsize=8, frameon=True, framealpha=0.95,
               edgecolor=_GRID)

    # Panel 5: Forecast Distribution
    if has_cartopy:
        ax5 = fig.add_subplot(gs[3, 0], projection=proj)
    else:
        ax5 = fig.add_subplot(gs[3, 0])
        
    setup_map(ax5)
    if has_cartopy:
        ax5.set_extent(extent, crs=data_crs)
    
    if has_cartopy:
        sc5 = ax5.scatter(sample_sim["longitude"], sample_sim["latitude"],
                          c=sample_sim["magnitude"], cmap='YlGnBu', s=sim_sizes, alpha=0.5,
                          edgecolors='none', linewidth=0.3,
                          transform=data_crs, zorder=3)
    else:
        sc5 = ax5.scatter(sample_sim["longitude"], sample_sim["latitude"],
                          c=sample_sim["magnitude"], cmap='YlGnBu', s=sim_sizes, alpha=0.5,
                          edgecolors='none', linewidth=0.3)
    cb5 = plt.colorbar(sc5, ax=ax5, label="Magnitude", shrink=0.75, pad=0.02)
    cb5.ax.tick_params(labelsize=8)
    
    n_sims = simulations['catalog_id'].nunique()
    mean_events = sim_counts.mean()
    ax5.text(0.02, 0.98, f"N sims: {n_sims}\nMean: {mean_events:.0f} events", 
             transform=ax5.transAxes, ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=_GRID, alpha=0.9))
    
    ax5.set_title("Forecast Distribution (sample)", fontsize=12, fontweight="bold",
                  color=_NAVY, pad=8)

    # Panel 6: Observed Distribution
    if has_cartopy:
        ax6 = fig.add_subplot(gs[3, 1], projection=proj)
    else:
        ax6 = fig.add_subplot(gs[3, 1])
    
    setup_map(ax6)
    if has_cartopy:
        ax6.set_extent(extent, crs=data_crs)
        
    if has_cartopy:
        sc6 = ax6.scatter(observed["longitude"], observed["latitude"],
                          c=observed["magnitude"], cmap='YlOrRd', s=obs_sizes,
                          alpha=0.85, edgecolors='white', linewidth=0.8,
                          transform=data_crs, zorder=5)
    else:
        sc6 = ax6.scatter(observed["longitude"], observed["latitude"],
                          c=observed["magnitude"], cmap='YlOrRd', s=obs_sizes,
                          alpha=0.85, edgecolors='white', linewidth=0.8)
    cb6 = plt.colorbar(sc6, ax=ax6, label="Magnitude", shrink=0.75, pad=0.02)
    cb6.ax.tick_params(labelsize=8)
    
    ax6.text(0.02, 0.98, f"Max M: {obs_max_mag:.1f}\nMean M: {observed['magnitude'].mean():.1f}", 
             transform=ax6.transAxes, ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=_GRID, alpha=0.9))
    
    ax6.set_title(f"Observed Distribution ({obs_count} events)", fontsize=12,
                  fontweight="bold", color=_NAVY, pad=8)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=_BG)
        print(f"6-Panel Plot saved to: {output_path}")
        
    plt.close(fig)
    return fig
