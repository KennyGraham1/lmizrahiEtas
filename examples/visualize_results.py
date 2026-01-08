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
CATALOG_PATH = "../input_data/nzcat.csv"
OUTPUT_DIR = "figures"
FORECAST_DAYS = 7.0

# Kaikoura sequence dates
KAIKOURA_DATES = [
    datetime(2016, 11, 13, 12, 0, 0), datetime(2016, 11, 14, 12, 0, 0),
    datetime(2016, 11, 15, 12, 0, 0), datetime(2016, 11, 16, 0, 0, 0),
    datetime(2016, 11, 16, 12, 0, 0), datetime(2016, 11, 17, 0, 0, 0),
    datetime(2016, 11, 17, 12, 0, 0), datetime(2016, 11, 18, 0, 0, 0),
    datetime(2016, 11, 18, 12, 0, 0), datetime(2016, 11, 19, 0, 0, 0),
    datetime(2016, 11, 19, 12, 0, 0), datetime(2016, 11, 20, 12, 0, 0),
    datetime(2016, 11, 21, 12, 0, 0), datetime(2016, 11, 22, 12, 0, 0),
    datetime(2016, 11, 23, 12, 0, 0), datetime(2016, 11, 24, 12, 0, 0),
    datetime(2016, 11, 25, 12, 0, 0), datetime(2016, 11, 26, 12, 0, 0),
    datetime(2016, 11, 27, 12, 0, 0), datetime(2016, 11, 28, 12, 0, 0),
    datetime(2016, 11, 29, 12, 0, 0), datetime(2016, 11, 30, 12, 0, 0),
    datetime(2016, 12, 1, 12, 0, 0), datetime(2016, 12, 2, 12, 0, 0),
    datetime(2016, 12, 7, 12, 0, 0), datetime(2016, 12, 11, 12, 0, 0),
    datetime(2016, 12, 14, 12, 0, 0), datetime(2016, 12, 18, 12, 0, 0),
    datetime(2016, 12, 21, 12, 0, 0), datetime(2016, 12, 25, 12, 0, 0),
    datetime(2016, 12, 28, 12, 0, 0), datetime(2017, 1, 1, 12, 0, 0),
    datetime(2017, 1, 8, 12, 0, 0), datetime(2017, 1, 15, 12, 0, 0),
    datetime(2017, 1, 22, 12, 0, 0), datetime(2017, 1, 29, 12, 0, 0),
    datetime(2017, 2, 5, 12, 0, 0), datetime(2017, 2, 12, 12, 0, 0),
    datetime(2017, 2, 19, 12, 0, 0), datetime(2017, 2, 26, 12, 0, 0),
    datetime(2017, 3, 5, 12, 0, 0), datetime(2017, 3, 12, 12, 0, 0),
    datetime(2017, 3, 19, 12, 0, 0), datetime(2017, 3, 26, 12, 0, 0),
    datetime(2017, 4, 2, 12, 0, 0),
]

CANTERBURY_DATES = [
    datetime(2010, 9, 3, 17, 0, 0), datetime(2010, 9, 4, 17, 0, 0),
    datetime(2010, 9, 5, 17, 0, 0), datetime(2010, 9, 6, 17, 0, 0),
    datetime(2010, 9, 7, 17, 0, 0), datetime(2010, 9, 8, 17, 0, 0),
    datetime(2010, 9, 9, 17, 0, 0), datetime(2010, 9, 10, 17, 0, 0),
    datetime(2010, 9, 11, 17, 0, 0), datetime(2010, 9, 12, 17, 0, 0),
    datetime(2010, 9, 13, 17, 0, 0), datetime(2010, 9, 14, 17, 0, 0),
    datetime(2010, 9, 15, 17, 0, 0), datetime(2010, 9, 16, 17, 0, 0),
]

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


def load_simulations(sequence: str, model_idx: int) -> pd.DataFrame:
    """Load all simulation chunks for a specific model.
    
    IMPORTANT: Each chunk file contains simulations with catalog_id 0-999.
    These are INDEPENDENT simulations, so we must offset the IDs to prevent
    merging events from different simulations when concatenating.
    """
    pattern = os.path.join(SIM_DIR, f"sim_nz_{sequence}_{model_idx}_*.csv")
    files = sorted(glob.glob(pattern))  # Sort to ensure consistent ordering
    if not files:
        return pd.DataFrame()
    
    chunks = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        # Offset catalog_id by chunk number to create unique IDs across all chunks
        # Chunk 0: IDs 0-999, Chunk 1: IDs 1000-1999, etc.
        df["catalog_id"] = df["catalog_id"] + i * 1000
        chunks.append(df)
    
    return pd.concat(chunks, ignore_index=True)


def load_catalog() -> pd.DataFrame:
    """Load the observed earthquake catalog."""
    return pd.read_csv(CATALOG_PATH, index_col=0, parse_dates=["time"])


def get_observed_in_window(catalog: pd.DataFrame, start: datetime, 
                           end: datetime, mc: float = 4.1) -> pd.DataFrame:
    """Filter catalog to events within forecast window above Mc."""
    mask = (catalog["time"] > start) & (catalog["time"] <= end) & (catalog["magnitude"] >= mc)
    return catalog[mask].copy()


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
    """Create enhanced spatial comparison with side-by-side maps."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Determine bounds from both datasets
    all_lons = pd.concat([simulations["longitude"], observed["longitude"]]) if len(simulations) > 0 else observed["longitude"]
    all_lats = pd.concat([simulations["latitude"], observed["latitude"]]) if len(simulations) > 0 else observed["latitude"]
    
    lon_margin = (all_lons.max() - all_lons.min()) * 0.1 + 0.3
    lat_margin = (all_lats.max() - all_lats.min()) * 0.1 + 0.3
    
    lon_min, lon_max = all_lons.min() - lon_margin, all_lons.max() + lon_margin
    lat_min, lat_max = all_lats.min() - lat_margin, all_lats.max() + lat_margin
    
    # --- Left: Simulated Density ---
    ax1 = axes[0]
    if len(simulations) > 0:
        h = ax1.hexbin(simulations["longitude"], simulations["latitude"], 
                       gridsize=35, cmap=SIM_CMAP, mincnt=1, alpha=0.9,
                       linewidths=0.1, edgecolors='white')
        cbar1 = plt.colorbar(h, ax=ax1, shrink=0.8, pad=0.02)
        cbar1.set_label("Simulated Event Density", fontsize=10)
    
    ax1.set_xlim(lon_min, lon_max)
    ax1.set_ylim(lat_min, lat_max)
    ax1.set_xlabel("Longitude (°E)", fontsize=11, fontweight='medium')
    ax1.set_ylabel("Latitude (°S)", fontsize=11, fontweight='medium')
    ax1.set_title("Simulated Seismicity\n(Aggregated over all catalogs)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_aspect('equal', adjustable='box')
    
    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # --- Right: Observed Events ---
    ax2 = axes[1]
    if len(observed) > 0:
        sizes = (observed["magnitude"] - 3.5) ** 2.5 * 25
        scatter = ax2.scatter(observed["longitude"], observed["latitude"], 
                              c=observed["magnitude"], s=sizes, cmap='plasma',
                              edgecolor='black', linewidth=0.8, alpha=0.85,
                              vmin=4.0, vmax=observed["magnitude"].max())
        cbar2 = plt.colorbar(scatter, ax=ax2, shrink=0.8, pad=0.02)
        cbar2.set_label("Magnitude", fontsize=10)
        
        # Annotate largest event
        max_idx = observed["magnitude"].idxmax()
        max_event = observed.loc[max_idx]
        ax2.annotate(f'M{max_event["magnitude"]:.1f}', 
                     (max_event["longitude"], max_event["latitude"]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, fontweight='bold', color=COLORS['observed'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['observed']))
    
    ax2.set_xlim(lon_min, lon_max)
    ax2.set_ylim(lat_min, lat_max)
    ax2.set_xlabel("Longitude (°E)", fontsize=11, fontweight='medium')
    ax2.set_ylabel("Latitude (°S)", fontsize=11, fontweight='medium')
    ax2.set_title(f"Observed Seismicity\n({len(observed)} events, M ≥ 4.1)", 
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


# --- Main Execution ---

def run_visualization(sequence: str = "Kaikoura", max_models: int = None):
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
    
    print(f"\n{'='*60}")
    print(f"Complete! Figures saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")
    
    return params_df, n_test_results


if __name__ == "__main__":
    run_visualization("Kaikoura", max_models=10)
    run_visualization("Canterbury", max_models=10)
