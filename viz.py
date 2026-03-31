"""
Visualization utilities for simplETAS.

Functions for plotting earthquake catalogs, frequency-magnitude distributions,
and ETAS model results.
"""

import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime


def plot_catalog_map(
    zmap: np.ndarray,
    title: str = "Earthquake Catalog",
    color_by: str = 'magnitude',
    size_scale: float = 10.0,
    polygon: Optional[np.ndarray] = None,
    use_cartopy: bool = True,
    ax=None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot earthquake catalog on a map with optional geographic features.
    
    Parameters
    ----------
    zmap : np.ndarray
        ZMAP format catalog
    title : str
        Plot title
    color_by : str
        'magnitude', 'depth', or 'time'
    size_scale : float
        Marker size scaling factor
    polygon : np.ndarray, optional
        Study region polygon to overlay
    use_cartopy : bool
        If True, use Cartopy for coastlines and terrain (default: True)
    ax : matplotlib axis, optional
        Existing axis to plot on
    figsize : tuple
        Figure size (default: (10, 8))
    show : bool
        Call plt.show() (default: True)
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib axis
    """
    import matplotlib.pyplot as plt
    
    lons = zmap[:, 0]
    lats = zmap[:, 1]
    mags = zmap[:, 5]
    
    # Size by magnitude
    sizes = size_scale * (10 ** (mags - mags.min()))
    sizes = np.clip(sizes, 5, 500)
    
    # Color by selected property
    if color_by == 'magnitude':
        colors = mags
        cmap = 'YlOrRd'
        label = 'Magnitude'
    elif color_by == 'depth':
        colors = zmap[:, 6]
        cmap = 'viridis_r'
        label = 'Depth (km)'
    elif color_by == 'time':
        colors = zmap[:, 2] + zmap[:, 3] / 12
        cmap = 'viridis'
        label = 'Time (year)'
    else:
        colors = mags
        cmap = 'YlOrRd'
        label = 'Magnitude'
    
    # Try to use Cartopy for geographic features
    if use_cartopy and ax is None:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set extent with padding
            lon_pad = (lons.max() - lons.min()) * 0.1
            lat_pad = (lats.max() - lats.min()) * 0.1
            ax.set_extent([lons.min() - lon_pad, lons.max() + lon_pad,
                           lats.min() - lat_pad, lats.max() + lat_pad],
                          crs=ccrs.PlateCarree())
            
            # Add geographic features
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.gridlines(draw_labels=True, alpha=0.3)
            
            scatter = ax.scatter(lons, lats, c=colors, s=sizes, alpha=0.7,
                                 cmap=cmap, edgecolors='k', linewidth=0.3,
                                 transform=ccrs.PlateCarree(), zorder=5)
            plt.colorbar(scatter, ax=ax, label=label, shrink=0.7)
            
            if polygon is not None:
                poly_closed = np.vstack([polygon, polygon[0]])
                ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'b-', linewidth=2,
                        transform=ccrs.PlateCarree(), label='Study region')
                ax.legend(loc='lower left')
            
            ax.set_title(title)
            
        except Exception as e:
            print(f"Cartopy error ({e}), falling back to simple plot")
            use_cartopy = False
    
    # Fallback to simple matplotlib
    if not use_cartopy or ax is not None:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(lons, lats, c=colors, s=sizes, alpha=0.6,
                             cmap=cmap, edgecolors='k', linewidth=0.3)
        plt.colorbar(scatter, ax=ax, label=label)
        
        if polygon is not None:
            poly_closed = np.vstack([polygon, polygon[0]])
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'b-', linewidth=2,
                    label='Study region')
            ax.legend()
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax




def plot_fmd(
    magnitudes: np.ndarray,
    mc: Optional[float] = None,
    mbin: float = 0.1,
    ax=None,
    figsize: Tuple[int, int] = (8, 6),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot frequency-magnitude distribution (FMD).
    
    Parameters
    ----------
    magnitudes : np.ndarray
        Array of magnitudes
    mc : float, optional
        Magnitude of completeness (auto-estimated if None)
    mbin : float
        Bin width (default: 0.1)
    ax : matplotlib axis, optional
    figsize : tuple
        Figure size
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib axis
    """
    import matplotlib.pyplot as plt
    from utils.data import estimate_mc, calculate_b_value
    
    magnitudes = np.asarray(magnitudes)
    
    if mc is None:
        mc = estimate_mc(magnitudes)
    
    # Compute FMD
    mag_min = np.floor(magnitudes.min() / mbin) * mbin
    mag_max = np.ceil(magnitudes.max() / mbin) * mbin
    bins = np.arange(mag_min, mag_max + mbin, mbin)
    counts, _ = np.histogram(magnitudes, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Cumulative count
    cum_counts = np.cumsum(counts[::-1])[::-1]
    
    # Calculate b-value
    b, b_std, n = calculate_b_value(magnitudes, mc, mbin)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Non-cumulative
    ax.bar(bin_centers, counts, width=mbin * 0.8, alpha=0.5, 
           label='Non-cumulative', color='steelblue')
    
    # Cumulative
    ax.scatter(bin_centers, cum_counts, c='darkred', s=30, zorder=5,
               label='Cumulative')
    
    # GR fit line
    if not np.isnan(b):
        mask = bin_centers >= mc
        fit_mags = bin_centers[mask]
        fit_n = cum_counts[bin_centers >= mc][0] if np.any(bin_centers >= mc) else 1
        fit_line = fit_n * 10 ** (-b * (fit_mags - mc))
        ax.plot(fit_mags, fit_line, 'k--', linewidth=2,
                label=f'GR fit (b={b:.2f}±{b_std:.2f})')
    
    # Mc line
    ax.axvline(mc, color='green', linestyle=':', linewidth=2,
               label=f'Mc = {mc:.1f}')
    
    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Number of Events')
    ax.set_title('Frequency-Magnitude Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_time_series(
    zmap: np.ndarray,
    mc: Optional[float] = None,
    ax=None,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot earthquake time series (magnitude vs time).
    
    Parameters
    ----------
    zmap : np.ndarray
        ZMAP format catalog
    mc : float, optional
        Show Mc line
    ax : matplotlib axis, optional
    figsize : tuple
        Figure size
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib axis
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to decimal year
    years = zmap[:, 2] + (zmap[:, 3] - 1) / 12 + (zmap[:, 4] - 1) / 365
    mags = zmap[:, 5]
    
    # Size by magnitude
    sizes = 5 * (mags - mags.min() + 1) ** 2
    
    ax.scatter(years, mags, s=sizes, alpha=0.5, c='steelblue', edgecolors='k', linewidth=0.2)
    
    if mc is not None:
        ax.axhline(mc, color='red', linestyle='--', label=f'Mc = {mc:.1f}')
        ax.legend()
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Magnitude')
    ax.set_title('Earthquake Time Series')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_cumulative_events(
    zmap: np.ndarray,
    ax=None,
    figsize: Tuple[int, int] = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot cumulative number of events over time.
    
    Parameters
    ----------
    zmap : np.ndarray
        ZMAP format catalog
    ax : matplotlib axis, optional
    figsize : tuple
        Figure size
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib axis
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to decimal year
    years = zmap[:, 2] + (zmap[:, 3] - 1) / 12 + (zmap[:, 4] - 1) / 365
    
    # Sort and cumulative count
    sorted_years = np.sort(years)
    cumulative = np.arange(1, len(sorted_years) + 1)
    
    ax.plot(sorted_years, cumulative, 'b-', linewidth=1.5)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Number of Events')
    ax.set_title('Cumulative Earthquake Count')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_background_rate(
    bg_matrix: np.ndarray,
    ax=None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'hot_r',
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot background seismicity rate as a heatmap.
    
    Parameters
    ----------
    bg_matrix : np.ndarray
        Background rate matrix [lon, lat, rate]
    ax : matplotlib axis, optional
    figsize : tuple
        Figure size
    cmap : str
        Colormap (default: 'hot_r')
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib axis
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    lons = bg_matrix[:, 0]
    lats = bg_matrix[:, 1]
    rates = bg_matrix[:, 2]
    
    # Scatter plot with rate as color
    scatter = ax.scatter(lons, lats, c=rates, s=50, cmap=cmap, 
                         marker='s', alpha=0.8)
    plt.colorbar(scatter, ax=ax, label='Background Rate')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Background Seismicity Rate')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_forecast_summary(
    forecast_dir: str,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot summary statistics of forecast simulations.
    
    Parameters
    ----------
    forecast_dir : str
        Directory containing simulation files
    figsize : tuple
        Figure size
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, axes)
    """
    import matplotlib.pyplot as plt
    import glob
    import os
    
    sim_files = sorted(glob.glob(os.path.join(forecast_dir, 'Simulation_*.txt')))
    
    if not sim_files:
        print(f"No simulation files found in {forecast_dir}")
        return None
    
    event_counts = []
    max_mags = []
    
    for sim_file in sim_files:
        try:
            data = np.loadtxt(sim_file)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            event_counts.append(len(data))
            if len(data) > 0:
                max_mags.append(data[:, 5].max())
        except:
            continue
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Event count histogram
    axes[0].hist(event_counts, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(event_counts), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(event_counts):.1f}')
    axes[0].set_xlabel('Number of Events')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Event Count Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Max magnitude histogram
    axes[1].hist(max_mags, bins=20, color='darkred', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(max_mags), color='blue', linestyle='--',
                     label=f'Mean: {np.mean(max_mags):.2f}')
    axes[1].set_xlabel('Maximum Magnitude')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Maximum Magnitude Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig, axes


def plot_depth_cross_section(
    zmap: np.ndarray,
    direction: str = 'lon',
    ax=None,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot depth cross-section of earthquakes.
    
    Parameters
    ----------
    zmap : np.ndarray
        ZMAP format catalog
    direction : str
        'lon' for E-W section, 'lat' for N-S section
    ax : matplotlib axis, optional
    figsize : tuple
        Figure size
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    lons = zmap[:, 0]
    lats = zmap[:, 1]
    depths = zmap[:, 6]
    mags = zmap[:, 5]
    
    # Size by magnitude
    sizes = 10 * (mags - mags.min() + 1) ** 2
    sizes = np.clip(sizes, 5, 200)
    
    if direction == 'lon':
        x = lons
        xlabel = 'Longitude'
    else:
        x = lats
        xlabel = 'Latitude'
    
    scatter = ax.scatter(x, depths, c=mags, s=sizes, alpha=0.6,
                         cmap='YlOrRd', edgecolors='k', linewidth=0.3)
    plt.colorbar(scatter, ax=ax, label='Magnitude')
    
    ax.invert_yaxis()  # Depth increases downward
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Depth (km)')
    ax.set_title(f'Depth Cross-Section ({direction.upper()})')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_omori_decay(
    zmap: np.ndarray,
    mainshock_time: Tuple[int, int, int, int, int, int],
    min_mag: float = 0.0,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot Omori-Utsu aftershock decay (event rate vs time since mainshock).
    
    Parameters
    ----------
    zmap : np.ndarray
        ZMAP format catalog
    mainshock_time : tuple
        (year, month, day, hour, minute, second) of mainshock
    min_mag : float
        Minimum magnitude to include
    ax : matplotlib axis, optional
    figsize : tuple
        Figure size
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Convert mainshock time to decimal days
    ms_year, ms_month, ms_day, ms_hour, ms_min, ms_sec = mainshock_time
    ms_decimal = ms_year + (ms_month - 1) / 12 + (ms_day - 1) / 365 + ms_hour / 8760
    
    # Convert catalog times to decimal years
    years = zmap[:, 2] + (zmap[:, 3] - 1) / 12 + (zmap[:, 4] - 1) / 365 + zmap[:, 7] / 8760
    mags = zmap[:, 5]
    
    # Filter by magnitude and time (after mainshock)
    mask = (mags >= min_mag) & (years > ms_decimal)
    times_after = (years[mask] - ms_decimal) * 365  # Convert to days
    
    if len(times_after) == 0:
        print("No aftershocks found after mainshock time")
        return None
    
    # Bin into logarithmic time bins
    log_bins = np.logspace(np.log10(0.01), np.log10(times_after.max()), 30)
    counts, bin_edges = np.histogram(times_after, bins=log_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    
    # Rate = counts / bin_width
    rates = counts / bin_widths
    
    # Plot
    ax.loglog(bin_centers, rates, 'o-', color='steelblue', markersize=6)
    
    ax.set_xlabel('Time Since Mainshock (days)')
    ax.set_ylabel('Earthquake Rate (events/day)')
    ax.set_title('Aftershock Decay (Omori-Utsu)')
    ax.grid(True, alpha=0.3, which='both')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_catalog_summary(
    zmap: np.ndarray,
    title: str = "Earthquake Catalog Summary",
    use_cartopy: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Create ZMAP-style multi-panel summary figure.
    
    Panels:
    - Top left: Map view (with Cartopy coastlines if available)
    - Top right: Frequency-magnitude distribution
    - Bottom left: Magnitude vs time
    - Bottom right: Cumulative events
    
    Parameters
    ----------
    zmap : np.ndarray
        ZMAP format catalog
    title : str
        Super title
    use_cartopy : bool
        Use Cartopy for geographic map with coastlines (default: True)
    figsize : tuple
        Figure size
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (fig, axes)
    """
    import matplotlib.pyplot as plt
    from utils.data import estimate_mc, calculate_b_value
    
    lons = zmap[:, 0]
    lats = zmap[:, 1]
    mags = zmap[:, 5]
    
    # Try to use Cartopy for the map panel
    cartopy_available = False
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            cartopy_available = True
        except ImportError:
            cartopy_available = False
    
    if cartopy_available:
        # Create figure with mixed projections
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Panel 1: Map with Cartopy (takes special projection)
        ax_map = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
        
        # Set extent with padding
        lon_pad = (lons.max() - lons.min()) * 0.1
        lat_pad = (lats.max() - lats.min()) * 0.1
        ax_map.set_extent([lons.min() - lon_pad, lons.max() + lon_pad,
                           lats.min() - lat_pad, lats.max() + lat_pad],
                          crs=ccrs.PlateCarree())
        
        # Add geographic features
        ax_map.add_feature(cfeature.LAND, facecolor='#f0f0f0', alpha=0.8)
        ax_map.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.5)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#333333')
        ax_map.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='#666666')
        
        # Add gridlines
        gl = ax_map.gridlines(draw_labels=True, alpha=0.4, linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot earthquakes
        sizes = 10 * (mags - mags.min() + 1) ** 1.5
        sizes = np.clip(sizes, 3, 200)
        scatter = ax_map.scatter(lons, lats, c=mags, s=sizes, alpha=0.6,
                                  cmap='YlOrRd', edgecolors='k', linewidth=0.2,
                                  transform=ccrs.PlateCarree(), zorder=5)
        plt.colorbar(scatter, ax=ax_map, label='Magnitude', shrink=0.8, pad=0.02)
        ax_map.set_title('Epicenter Map')
        
        # Other panels (standard matplotlib)
        ax_fmd = fig.add_subplot(2, 2, 2)
        ax_time = fig.add_subplot(2, 2, 3)
        ax_cum = fig.add_subplot(2, 2, 4)
        
        axes = np.array([[ax_map, ax_fmd], [ax_time, ax_cum]])
    else:
        # Fallback: all standard matplotlib
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Panel 1: Simple map
        ax_map = axes[0, 0]
        sizes = 10 * (mags - mags.min() + 1) ** 1.5
        sizes = np.clip(sizes, 3, 200)
        scatter = ax_map.scatter(lons, lats, c=mags, s=sizes, alpha=0.5,
                                  cmap='YlOrRd', edgecolors='k', linewidth=0.2)
        plt.colorbar(scatter, ax=ax_map, label='Magnitude')
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        ax_map.set_title('Epicenter Map')
        ax_map.set_aspect('equal')
        ax_map.grid(True, alpha=0.3)
        
        ax_fmd = axes[0, 1]
        ax_time = axes[1, 0]
        ax_cum = axes[1, 1]
    
    # Panel 2: FMD
    mc = estimate_mc(mags)
    b, b_std, n = calculate_b_value(mags, mc)
    
    mag_min = np.floor(mags.min() * 10) / 10
    mag_max = np.ceil(mags.max() * 10) / 10
    bins = np.arange(mag_min, mag_max + 0.1, 0.1)
    counts, _ = np.histogram(mags, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    cum_counts = np.cumsum(counts[::-1])[::-1]
    
    ax_fmd.bar(bin_centers, counts, width=0.08, alpha=0.5, color='steelblue', label='Non-cumulative')
    ax_fmd.scatter(bin_centers, cum_counts, c='darkred', s=20, zorder=5, label='Cumulative')
    
    if not np.isnan(b):
        mask = bin_centers >= mc
        fit_mags = bin_centers[mask]
        if np.any(mask):
            fit_n = cum_counts[mask][0]
            fit_line = fit_n * 10 ** (-b * (fit_mags - mc))
            ax_fmd.plot(fit_mags, fit_line, 'k--', linewidth=1.5, label=f'b={b:.2f}')
    
    ax_fmd.axvline(mc, color='green', linestyle=':', label=f'Mc={mc:.1f}')
    ax_fmd.set_yscale('log')
    ax_fmd.set_xlabel('Magnitude')
    ax_fmd.set_ylabel('Count')
    ax_fmd.set_title('Frequency-Magnitude Distribution')
    ax_fmd.legend(fontsize=8)
    ax_fmd.grid(True, alpha=0.3)
    
    # Panel 3: Time series
    years = zmap[:, 2] + (zmap[:, 3] - 1) / 12 + (zmap[:, 4] - 1) / 365
    sizes = 5 * (mags - mags.min() + 1) ** 1.5
    ax_time.scatter(years, mags, s=sizes, alpha=0.4, c='steelblue', edgecolors='k', linewidth=0.1)
    ax_time.axhline(mc, color='red', linestyle='--', alpha=0.7, label=f'Mc={mc:.1f}')
    ax_time.set_xlabel('Year')
    ax_time.set_ylabel('Magnitude')
    ax_time.set_title('Magnitude vs Time')
    ax_time.legend(fontsize=8)
    ax_time.grid(True, alpha=0.3)
    
    # Panel 4: Cumulative
    sorted_years = np.sort(years)
    cumulative = np.arange(1, len(sorted_years) + 1)
    ax_cum.plot(sorted_years, cumulative, 'b-', linewidth=1)
    ax_cum.set_xlabel('Year')
    ax_cum.set_ylabel('Cumulative Events')
    ax_cum.set_title('Cumulative Number of Events')
    ax_cum.grid(True, alpha=0.3)
    
    # Add statistics text
    duration = years.max() - years.min()
    rate = len(zmap) / duration if duration > 0 else 0
    stats_text = f"N={len(zmap)}, Duration={duration:.1f}yr, Rate={rate:.0f}/yr"
    ax_cum.text(0.02, 0.98, stats_text, transform=ax_cum.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig, axes


# =============================================================================
# FORECAST VS OBSERVED COMPARISON
# =============================================================================

def plot_forecast_vs_observed(
    forecast_dir: str,
    observed_catalog: np.ndarray,
    min_mag: float = 0.0,
    use_cartopy: bool = True,
    figsize: Tuple[int, int] = (14, 14),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Compare forecast simulations with observed catalog using CSEP-style tests.
    
    Creates a 6-panel comparison figure (3 rows × 2 columns):
    - Row 1: N-test (event count), M-test (max magnitude)
    - Row 2: Magnitude distribution comparison, Combined spatial map
    - Row 3: Forecast-only spatial map, Observed-only spatial map
    
    Based on CSEP (Collaboratory for Study of Earthquake Predictability) 
    testing framework: Schorlemmer et al. (2007), Zechar et al. (2010).
    
    Parameters
    ----------
    forecast_dir : str
        Directory containing Simulation_*.txt files
    observed_catalog : np.ndarray
        Observed ZMAP catalog for comparison period
    min_mag : float
        Minimum magnitude to include (default: 0.0)
    use_cartopy : bool
        Use Cartopy for geographic map (default: True)
    figsize : tuple
        Figure size
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    dict
        CSEP test results including quantiles and pass/fail status
    """
    import matplotlib.pyplot as plt
    import glob
    import os
    
    # Load simulations
    sim_files = sorted(glob.glob(os.path.join(forecast_dir, 'Simulation_*.txt')))
    
    if not sim_files:
        print(f"No simulation files found in {forecast_dir}")
        return None
    
    sim_event_counts = []
    sim_max_mags = []
    sim_all_mags = []
    sim_all_lons = []
    sim_all_lats = []
    
    for sim_file in sim_files:
        try:
            data = np.loadtxt(sim_file)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            mags = data[:, 5]
            mask = mags >= min_mag
            sim_event_counts.append(mask.sum())
            if mask.sum() > 0:
                sim_max_mags.append(mags[mask].max())
                sim_all_mags.extend(mags[mask].tolist())
                sim_all_lons.extend(data[mask, 0].tolist())
                sim_all_lats.extend(data[mask, 1].tolist())
        except:
            continue
    
    # Observed statistics
    obs_mags = observed_catalog[:, 5]
    obs_mask = obs_mags >= min_mag
    obs_count = obs_mask.sum()
    obs_max_mag = obs_mags[obs_mask].max() if obs_mask.sum() > 0 else 0
    obs_lons = observed_catalog[obs_mask, 0]
    obs_lats = observed_catalog[obs_mask, 1]
    
    # CSEP quantile calculations
    # N-test: two-sided test, reject if quantile < 0.025 or > 0.975
    n_quantile = np.sum(np.array(sim_event_counts) <= obs_count) / len(sim_event_counts)
    n_test_pass = 0.025 <= n_quantile <= 0.975
    
    # M-test: one-sided, reject if observed max too high
    m_quantile = np.sum(np.array(sim_max_mags) <= obs_max_mag) / len(sim_max_mags)
    
    # Try Cartopy for map
    cartopy_available = False
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            cartopy_available = True
        except ImportError:
            cartopy_available = False
    
    # Create figure with 3 rows × 2 columns (6 panels)
    if cartopy_available:
        fig = plt.figure(figsize=figsize)
        fig.suptitle('CSEP-Style Forecast Evaluation', fontsize=14, fontweight='bold')
        
        # Row 1: Standard axes for N-test and M-test
        ax_n = fig.add_subplot(3, 2, 1)
        ax_m = fig.add_subplot(3, 2, 2)
        
        # Row 2: FMD + Combined spatial map
        ax_fmd = fig.add_subplot(3, 2, 3)
        ax_map = fig.add_subplot(3, 2, 4, projection=ccrs.PlateCarree())
        
        # Row 3: Separate spatial maps for forecast and observed
        ax_forecast = fig.add_subplot(3, 2, 5, projection=ccrs.PlateCarree())
        ax_observed = fig.add_subplot(3, 2, 6, projection=ccrs.PlateCarree())
    else:
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('CSEP-Style Forecast Evaluation', fontsize=14, fontweight='bold')
        ax_n = axes[0, 0]
        ax_m = axes[0, 1]
        ax_fmd = axes[1, 0]
        ax_map = axes[1, 1]
        ax_forecast = axes[2, 0]
        ax_observed = axes[2, 1]
    
    # Panel 1: N-test (Event count)
    ax_n.hist(sim_event_counts, bins=20, color='steelblue', 
              edgecolor='black', alpha=0.7, label='Forecast simulations')
    ax_n.axvline(obs_count, color='red', linewidth=2.5, linestyle='-', 
                 label=f'Observed: {obs_count}')
    ax_n.axvline(np.mean(sim_event_counts), color='blue', linewidth=1.5, 
                 linestyle='--', label=f'Mean: {np.mean(sim_event_counts):.0f}')
    
    ax_n.set_xlabel('Number of Events')
    ax_n.set_ylabel('Frequency')
    ax_n.set_title(f'N-test: quantile={n_quantile:.3f}')
    ax_n.legend(fontsize=8)
    ax_n.grid(True, alpha=0.3)
    
    # Panel 2: M-test (Max magnitude)
    ax_m.hist(sim_max_mags, bins=20, color='darkorange', edgecolor='black', 
              alpha=0.7, label='Forecast simulations')
    ax_m.axvline(obs_max_mag, color='blue', linewidth=2.5, linestyle='-', 
                 label=f'Observed: M{obs_max_mag:.1f}')
    ax_m.axvline(np.percentile(sim_max_mags, 95), color='red', linewidth=1.5, 
                 linestyle='--', label='95th percentile')
    
    ax_m.set_xlabel('Maximum Magnitude')
    ax_m.set_ylabel('Frequency')
    ax_m.set_title(f'M-test: δ={m_quantile:.3f}')
    ax_m.legend(fontsize=8)
    ax_m.grid(True, alpha=0.3)
    
    # Panel 3: Magnitude distribution comparison
    if len(sim_all_mags) > 0 and obs_mask.sum() > 0:
        mag_max = max(max(sim_all_mags), obs_mags[obs_mask].max()) + 0.2
        bins = np.arange(min_mag, mag_max, 0.2)
        
        ax_fmd.hist(sim_all_mags, bins=bins, alpha=0.5, color='steelblue', 
                    density=True, label=f'Forecast ({len(sim_files)} sims)')
        ax_fmd.hist(obs_mags[obs_mask], bins=bins, alpha=0.8, color='darkred', 
                    density=True, histtype='step', linewidth=2, label='Observed')
    
    ax_fmd.set_xlabel('Magnitude')
    ax_fmd.set_ylabel('Probability Density')
    ax_fmd.set_title('Magnitude Distribution Comparison')
    ax_fmd.legend(fontsize=9)
    ax_fmd.grid(True, alpha=0.3)
    
    # Panel 4: Spatial comparison
    # Calculate extent from observed data (more reliable than forecast sample)
    if len(obs_lons) > 0:
        lon_pad = max((obs_lons.max() - obs_lons.min()) * 0.15, 0.5)
        lat_pad = max((obs_lats.max() - obs_lats.min()) * 0.15, 0.5)
        extent = [obs_lons.min() - lon_pad, obs_lons.max() + lon_pad,
                  obs_lats.min() - lat_pad, obs_lats.max() + lat_pad]
    else:
        extent = None
    
    if cartopy_available:
        if extent:
            ax_map.set_extent(extent, crs=ccrs.PlateCarree())

        ax_map.add_feature(cfeature.LAND, facecolor='#f0f0f0', alpha=0.8)
        ax_map.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.5)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=1.0)
        gl = ax_map.gridlines(draw_labels=True, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot sample of simulated events (background)
        if len(sim_all_lons) > 0:
            ax_map.scatter(sim_all_lons[:500], sim_all_lats[:500], 
                           c='steelblue', s=8, alpha=0.2, 
                           transform=ccrs.PlateCarree(), label='Forecast sample')
        
        # Plot observed (foreground)
        if len(obs_lons) > 0:
            obs_sizes = np.clip(20 * (obs_mags[obs_mask] - min_mag + 1), 10, 200)
            ax_map.scatter(obs_lons, obs_lats, c='red', s=obs_sizes, alpha=0.8,
                           edgecolors='k', linewidth=0.3, transform=ccrs.PlateCarree(),
                           label=f'Observed ({obs_count})', zorder=10)
    else:
        if len(sim_all_lons) > 0:
            ax_map.scatter(sim_all_lons[:500], sim_all_lats[:500], 
                           c='steelblue', s=8, alpha=0.2, label='Forecast sample')
        if len(obs_lons) > 0:
            obs_sizes = np.clip(20 * (obs_mags[obs_mask] - min_mag + 1), 10, 200)
            ax_map.scatter(obs_lons, obs_lats, c='red', s=obs_sizes, alpha=0.8,
                           edgecolors='k', linewidth=0.3, label=f'Observed ({obs_count})')
        ax_map.set_aspect('equal')
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        ax_map.grid(True, alpha=0.3)
    
    ax_map.set_title('Combined Spatial Distribution')
    ax_map.legend(fontsize=8, loc='lower left')
    
    # Helper function to set up map axes
    def setup_map_axis(ax, title):
        if cartopy_available:
            if extent:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', alpha=0.8)
            ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.5)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            gl = ax.gridlines(draw_labels=True, alpha=0.4, linewidth=0.5)
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax.set_aspect('equal')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
        ax.set_title(title)
    
    # Panel 5: Forecast-only spatial distribution
    setup_map_axis(ax_forecast, f'Forecast Distribution (sample)')
    
    if len(sim_all_lons) > 0:
        n_plot = min(1000, len(sim_all_lons))
        sim_plot_mags = np.array(sim_all_mags[:n_plot])
        sim_sizes = np.clip(15 * (sim_plot_mags - min_mag + 1) ** 1.2, 5, 150)
        
        if cartopy_available:
            sc = ax_forecast.scatter(sim_all_lons[:n_plot], sim_all_lats[:n_plot], 
                                    c=sim_plot_mags, s=sim_sizes, alpha=0.6,
                                    cmap='Blues', edgecolors='steelblue', linewidth=0.3,
                                    transform=ccrs.PlateCarree())
        else:
            sc = ax_forecast.scatter(sim_all_lons[:n_plot], sim_all_lats[:n_plot], 
                                    c=sim_plot_mags, s=sim_sizes, alpha=0.6,
                                    cmap='Blues', edgecolors='steelblue', linewidth=0.3)
        plt.colorbar(sc, ax=ax_forecast, label='Magnitude', shrink=0.7, pad=0.02)
    
    # Add statistics text
    sim_stats = f"N sims: {len(sim_files)}\nMean: {np.mean(sim_event_counts):.0f} events"
    ax_forecast.text(0.02, 0.98, sim_stats, transform=ax_forecast.transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 6: Observed-only spatial distribution
    setup_map_axis(ax_observed, f'Observed Distribution ({obs_count} events)')
    
    if len(obs_lons) > 0:
        obs_sizes = np.clip(20 * (obs_mags[obs_mask] - min_mag + 1) ** 1.2, 10, 250)
        
        if cartopy_available:
            sc = ax_observed.scatter(obs_lons, obs_lats, c=obs_mags[obs_mask], s=obs_sizes, 
                                    alpha=0.7, cmap='Reds', edgecolors='k', linewidth=0.3,
                                    transform=ccrs.PlateCarree())
        else:
            sc = ax_observed.scatter(obs_lons, obs_lats, c=obs_mags[obs_mask], s=obs_sizes, 
                                    alpha=0.7, cmap='Reds', edgecolors='k', linewidth=0.3)
        plt.colorbar(sc, ax=ax_observed, label='Magnitude', shrink=0.7, pad=0.02)
        
        # Add observed stats text
        obs_stats = f"Max M: {obs_max_mag:.1f}\nMean M: {obs_mags[obs_mask].mean():.1f}"
        ax_observed.text(0.02, 0.98, obs_stats, transform=ax_observed.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    # Return CSEP test statistics
    stats = {
        'n_simulations': len(sim_files),
        'n_test': {
            'observed': obs_count,
            'forecast_mean': np.mean(sim_event_counts),
            'forecast_std': np.std(sim_event_counts),
            'quantile': n_quantile,
            'pass': n_test_pass,
        },
        'm_test': {
            'observed': obs_max_mag,
            'forecast_mean': np.mean(sim_max_mags),
            'quantile': m_quantile,
        },
    }
    
    return stats


def plot_forecast_consistency(
    forecast_dir: str,
    observed_catalog: np.ndarray,
    n_bins: int = 10,
    min_mag: float = 0.0,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot forecast consistency test (N-test style).
    
    Shows the probability integral transform (PIT) histogram 
    to assess if forecasts are well-calibrated.
    
    Parameters
    ----------
    forecast_dir : str
        Directory containing simulation files
    observed_catalog : np.ndarray
        Observed ZMAP catalog
    n_bins : int
        Number of bins for PIT histogram
    min_mag : float
        Minimum magnitude
    figsize : tuple
        Figure size
    show : bool
        Call plt.show()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    dict
        Consistency test statistics
    """
    import matplotlib.pyplot as plt
    import glob
    import os
    from scipy import stats as scipy_stats
    
    sim_files = sorted(glob.glob(os.path.join(forecast_dir, 'Simulation_*.txt')))
    
    if not sim_files:
        print(f"No simulation files found in {forecast_dir}")
        return None
    
    # Get simulated event counts
    sim_counts = []
    for sim_file in sim_files:
        try:
            data = np.loadtxt(sim_file)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            mask = data[:, 5] >= min_mag
            sim_counts.append(mask.sum())
        except:
            continue
    
    sim_counts = np.array(sim_counts)
    obs_count = np.sum(observed_catalog[:, 5] >= min_mag)
    
    # Calculate quantile of observation
    quantile = np.sum(sim_counts <= obs_count) / len(sim_counts)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: CDF comparison
    ax = axes[0]
    sorted_counts = np.sort(sim_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax.plot(sorted_counts, cdf, 'b-', linewidth=2, label='Forecast CDF')
    ax.axvline(obs_count, color='red', linewidth=2, linestyle='--', 
               label=f'Observed: {obs_count}')
    ax.axhline(quantile, color='green', linewidth=1, linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'N-test: Obs at quantile {quantile:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Summary statistics
    ax = axes[1]
    
    # Create box-and-whisker style summary
    percentiles = [5, 25, 50, 75, 95]
    perc_values = np.percentile(sim_counts, percentiles)
    
    ax.barh(0, perc_values[4] - perc_values[0], left=perc_values[0], 
            color='lightblue', height=0.3, label='5-95% range')
    ax.barh(0, perc_values[3] - perc_values[1], left=perc_values[1], 
            color='steelblue', height=0.3, label='25-75% range')
    ax.axvline(perc_values[2], color='blue', linewidth=2, label='Median')
    ax.axvline(obs_count, color='red', linewidth=2, linestyle='--', label='Observed')
    
    ax.set_yticks([])
    ax.set_xlabel('Number of Events')
    ax.set_title('Forecast Range vs Observed')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    # Return statistics
    return {
        'quantile': quantile,
        'observed': obs_count,
        'forecast_median': np.median(sim_counts),
        'forecast_5th': np.percentile(sim_counts, 5),
        'forecast_95th': np.percentile(sim_counts, 95),
        'within_90CI': (obs_count >= np.percentile(sim_counts, 5)) and 
                       (obs_count <= np.percentile(sim_counts, 95)),
    }


