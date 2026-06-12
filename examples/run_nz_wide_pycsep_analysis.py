"""
Comprehensive pyCSEP analysis for NZ-wide ETAS forecast runs.

This script reads the outputs from ``run_nz_wide_forecast.py`` and performs
catalog-based pyCSEP evaluations on each forecast horizon. It generates:

1. Per-horizon dashboards with pyCSEP N/M/S/PL tests, temporal diagnostics,
   and a spatial expected-rate view.
2. A cross-horizon consistency overview figure.
3. CSV/JSON/Markdown summaries for downstream reporting.

Usage:
    python run_nz_wide_pycsep_analysis.py \
        --metadata output_nz_wide/nz_wide_20180101_000000/experiment_config.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT_DIR, ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

OPEN_SHA_VENDOR_PYCSEP = os.path.abspath(
    os.path.join(ROOT_DIR, "..", "opensha-oaf", "vendor", "pycsep")
)
if os.path.isdir(OPEN_SHA_VENDOR_PYCSEP) and OPEN_SHA_VENDOR_PYCSEP not in sys.path:
    sys.path.append(OPEN_SHA_VENDOR_PYCSEP)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.patches import FancyBboxPatch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import csep.plots as csep_plots
    from csep.core import catalog_evaluations, catalogs, forecasts, regions
    from csep.models import Polygon
    from csep.utils.time_utils import datetime_to_utc_epoch
except ImportError as exc:
    raise ImportError(
        "pyCSEP is required for this analysis. Install the `csep` package or "
        "make sure a vendored pyCSEP checkout is importable."
    ) from exc


DEFAULT_METADATA_PATH = os.path.join(
    BASE_DIR, "output_nz_wide", "nz_wide_20180101_000000", "experiment_config.json"
)
DEFAULT_OUTPUT_SUBDIR = "pycsep_analysis"
DEFAULT_REGION_SOURCE = "forecast_domain"
DEFAULT_GRID_SPACING = 0.1
DEFAULT_MAG_BIN = 0.1

TEST_CONFIGS = [
    {
        "key": "number_test",
        "label": "Catalog N-Test",
        "one_sided_lower": False,
    },
    {
        "key": "magnitude_test",
        "label": "Catalog M-Test",
        "one_sided_lower": False,
    },
    {
        "key": "spatial_test",
        "label": "Catalog S-Test",
        "one_sided_lower": True,
    },
    {
        "key": "pseudolikelihood_test",
        "label": "Catalog PL-Test",
        "one_sided_lower": True,
    },
]

COLORS = {
    "observed": "#B33A3A",
    "simulated": "#2C6EAA",
    "band": "#A9C9E8",
    "median": "#1F4E79",
    "grid": "#E2E8F0",
    "accent": "#E8A24A",
    "good": "#1B7F4B",
    "bad": "#C0392B",
    "neutral": "#94A3B8",
    "muted": "#475569",
    "ink": "#1F2933",
    "hair": "#E2E8F0",
    "head_bg": "#1B2A3A",
    "head_tx": "#F4F7FA",
    "land": "#EEF1F4",
    "ocean": "#DCE7EE",
    "coast": "#3A4A57",
    "neat": "#9AA7B4",
    "cb_exp": "#2C6EAA",
    "cb_emp": "#E8A24A",
    "ob_sup": "#3F7F8C",
    "ob_zero": "#C0392B",
}
MONO = "DejaVu Sans Mono"


def apply_dashboard_style() -> None:
    """Global, reproducible matplotlib style for the dashboards."""
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#CBD5E1", "axes.linewidth": 0.8,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
        "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 8, "axes.titlecolor": COLORS["ink"],
        "axes.labelsize": 10.5, "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "grid.color": COLORS["hair"], "grid.linewidth": 0.7,
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "#CBD5E1",
        "legend.fontsize": 9, "text.color": COLORS["ink"],
        "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "dejavusans",
    })


def _style_panel(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.9)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    ax.tick_params(length=3, width=0.7)


@dataclass
class HorizonEvaluation:
    duration_days: float
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    forecast: forecasts.CatalogForecast
    observed_catalog: catalogs.CSEPCatalog
    filtered_simulations: pd.DataFrame
    filtered_observed: pd.DataFrame
    results: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass
class AnalysisOutputs:
    output_dir: str
    summary_csv_path: str
    results_json_path: str
    report_path: str
    overview_path: str
    dashboard_paths: list[str]
    metadata_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pyCSEP catalog-based analysis on NZ-wide ETAS outputs."
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help=f"Path to experiment_config.json. Default: {DEFAULT_METADATA_PATH}",
    )
    parser.add_argument(
        "--region-source",
        choices=["forecast_domain", "nz_csep_collection"],
        default=DEFAULT_REGION_SOURCE,
        help=(
            "Spatial region used for pyCSEP binning. "
            "`forecast_domain` matches the ETAS polygon. "
            "`nz_csep_collection` uses pyCSEP's NZ collection region."
        ),
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=DEFAULT_GRID_SPACING,
        help=f"Grid spacing in degrees for forecast_domain regions. Default: {DEFAULT_GRID_SPACING}",
    )
    parser.add_argument(
        "--mag-bin",
        type=float,
        default=DEFAULT_MAG_BIN,
        help=f"Magnitude bin width. Default: {DEFAULT_MAG_BIN}",
    )
    parser.add_argument(
        "--max-mag",
        type=float,
        default=None,
        help="Maximum magnitude bin edge. Default: inferred from the run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for pyCSEP outputs. Default: "
            "<run output dir>/pycsep_analysis"
        ),
    )
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_metadata(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def update_metadata_with_analysis(metadata_path: str, payload: dict[str, Any]) -> None:
    metadata = load_metadata(metadata_path)
    metadata["pycsep_analysis"] = payload
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def infer_max_magnitude(metadata: dict[str, Any], mag_bin: float, explicit: float | None) -> float:
    if explicit is not None:
        return explicit

    max_mag = float(metadata["mc"]) + mag_bin
    all_paths = list(metadata.get("simulation_files", {}).values()) + list(
        metadata.get("observed_files", {}).values()
    )
    for path in all_paths:
        if not os.path.exists(path):
            continue
        mags = pd.read_csv(path, usecols=["magnitude"])
        if len(mags) == 0:
            continue
        max_mag = max(max_mag, float(mags["magnitude"].max()))

    return round(math.ceil(max_mag / mag_bin) * mag_bin + mag_bin, 6)


def build_forecast_domain_region(
    polygon_path: str,
    magnitudes: np.ndarray,
    grid_spacing: float,
) -> regions.CartesianGrid2D:
    coords = np.load(polygon_path)
    lats = coords[:, 0]
    lons = coords[:, 1]
    polygon = Polygon(np.column_stack([lons, lats]))

    lon_origins = np.arange(lons.min(), lons.max(), grid_spacing)
    lat_origins = np.arange(lats.min(), lats.max(), grid_spacing)
    origins = np.array(list(itertools.product(lon_origins, lat_origins)))

    base_region = regions.CartesianGrid2D.from_origins(
        origins,
        dh=grid_spacing,
        magnitudes=magnitudes,
        name="etas-forecast-domain",
    )
    masked = regions.masked_region(base_region, polygon)
    return regions.create_space_magnitude_region(masked, magnitudes)


def build_region(
    metadata: dict[str, Any],
    region_source: str,
    grid_spacing: float,
    mag_bin: float,
    max_mag: float,
) -> regions.CartesianGrid2D:
    magnitudes = regions.magnitude_bins(float(metadata["mc"]), max_mag, mag_bin)
    if region_source == "nz_csep_collection":
        return regions.nz_csep_collection_region(magnitudes=magnitudes)
    return build_forecast_domain_region(metadata["polygon_path"], magnitudes, grid_spacing)


def to_csep_dataframe(df: pd.DataFrame, name_prefix: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(
            columns=["id", "origin_time", "latitude", "longitude", "depth", "magnitude"]
        )

    out = df.copy()
    if "time" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["time"]):
        out["time"] = pd.to_datetime(out["time"])

    if "id" not in out.columns:
        out["id"] = [f"{name_prefix}_{i}" for i in range(len(out))]
    else:
        out["id"] = out["id"].astype(str)

    out["origin_time"] = out["time"].map(datetime_to_utc_epoch).astype(np.int64)
    if "depth" not in out.columns:
        out["depth"] = 0.0

    return out[["id", "origin_time", "latitude", "longitude", "depth", "magnitude"]]


def build_csep_catalog(
    df: pd.DataFrame,
    region: regions.CartesianGrid2D,
    name: str,
    catalog_id: int | None = None,
) -> catalogs.CSEPCatalog:
    if len(df) == 0:
        return catalogs.CSEPCatalog(data=[], catalog_id=catalog_id, name=name, region=region)

    converted = to_csep_dataframe(df, name)
    catalog = catalogs.CSEPCatalog.from_dataframe(
        converted,
        name=name,
        region=region,
    )
    if catalog_id is not None:
        catalog.catalog_id = catalog_id
    return catalog.filter_spatial(region=region, in_place=False)


def build_catalog_forecast(
    simulations: pd.DataFrame,
    region: regions.CartesianGrid2D,
    n_catalogs: int,
    name: str,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> tuple[forecasts.CatalogForecast, pd.DataFrame]:
    sims = simulations.copy()
    if "time" in sims.columns and not pd.api.types.is_datetime64_any_dtype(sims["time"]):
        sims["time"] = pd.to_datetime(sims["time"])
    sims["catalog_id"] = sims["catalog_id"].astype(int)

    grouped = {
        int(catalog_id): group.drop(columns=["catalog_id"]).copy()
        for catalog_id, group in sims.groupby("catalog_id")
    }

    filtered_rows = []
    forecast_catalogs = []
    for catalog_id in range(n_catalogs):
        raw_catalog = grouped.get(catalog_id, pd.DataFrame())
        csep_catalog = build_csep_catalog(
            raw_catalog,
            region=region,
            name=f"{name}_{catalog_id}",
            catalog_id=catalog_id,
        )
        forecast_catalogs.append(csep_catalog)

        catalog_df = csep_catalog.to_dataframe()
        if len(catalog_df) > 0:
            catalog_df["catalog_id"] = catalog_id
            catalog_df["time"] = pd.to_datetime(catalog_df["origin_time"], unit="ms", utc=True)
            catalog_df["time"] = catalog_df["time"].dt.tz_localize(None)
            filtered_rows.append(
                catalog_df[["catalog_id", "longitude", "latitude", "magnitude", "time"]]
            )

    filtered_simulations = (
        pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame(
            columns=["catalog_id", "longitude", "latitude", "magnitude", "time"]
        )
    )

    forecast = forecasts.CatalogForecast(
        catalogs=forecast_catalogs,
        name=name,
        region=region,
        start_time=forecast_start.to_pydatetime(),
        end_time=forecast_end.to_pydatetime(),
        n_cat=n_catalogs,
    )
    return forecast, filtered_simulations


def summarize_result(result: Any, one_sided_lower: bool) -> dict[str, Any]:
    if result is None:
        return {
            "status": "not-run",
            "consistent": None,
            "quantile_lower": np.nan,
            "quantile_upper": np.nan,
            "observed_statistic": np.nan,
            "distribution_mean": np.nan,
            "distribution_std": np.nan,
            "distribution_n": 0,
        }

    lower = np.nan
    upper = np.nan
    if isinstance(result.quantile, (list, tuple)) and len(result.quantile) == 2:
        lower = float(result.quantile[0])
        upper = float(result.quantile[1])

    consistent = None
    if result.status != "not-valid" and not np.isnan(lower) and not np.isnan(upper):
        if one_sided_lower:
            consistent = bool(upper >= 0.025)
        else:
            # pyCSEP returns (P[T_sim >= T_obs], P[T_sim <= T_obs]).
            consistent = bool(lower >= 0.025 and upper >= 0.025)

    distribution = np.asarray(result.test_distribution, dtype=float)
    if distribution.size == 0:
        dist_mean = np.nan
        dist_std = np.nan
    else:
        dist_mean = float(np.nanmean(distribution))
        dist_std = float(np.nanstd(distribution))

    return {
        "status": result.status,
        "consistent": consistent,
        "quantile_lower": lower,
        "quantile_upper": upper,
        "observed_statistic": float(result.observed_statistic)
        if result.observed_statistic is not None
        else np.nan,
        "distribution_mean": dist_mean,
        "distribution_std": dist_std,
        "distribution_n": int(distribution.size),
    }


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def compute_temporal_envelope(
    simulations: pd.DataFrame,
    forecast_start: pd.Timestamp,
    duration_days: float,
    n_catalogs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_points = int(min(80, max(16, round(duration_days) + 1)))
    x_days = np.linspace(0.0, duration_days, n_points)
    curves = np.zeros((n_catalogs, x_days.size))

    if len(simulations) > 0:
        sims = simulations.copy()
        if not pd.api.types.is_datetime64_any_dtype(sims["time"]):
            sims["time"] = pd.to_datetime(sims["time"])
        sims["days_from_start"] = (
            sims["time"] - forecast_start
        ).dt.total_seconds() / 86400.0

        for catalog_id, group in sims.groupby("catalog_id"):
            event_days = np.sort(group["days_from_start"].to_numpy())
            curves[int(catalog_id)] = np.searchsorted(event_days, x_days, side="right")

    median = np.percentile(curves, 50, axis=0)
    p05 = np.percentile(curves, 5, axis=0)
    p95 = np.percentile(curves, 95, axis=0)
    return x_days, median, p05, p95


def _region_grid(region):
    """Build a regular 2D grid (+ cell edges) from a CSEP region's midpoints."""
    mid = np.asarray(region.midpoints(), dtype=float)
    dh = float(getattr(region, "dh", DEFAULT_GRID_SPACING))
    lons = np.unique(np.round(mid[:, 0], 5))
    lats = np.unique(np.round(mid[:, 1], 5))
    ix = np.searchsorted(lons, np.round(mid[:, 0], 5))
    iy = np.searchsorted(lats, np.round(mid[:, 1], 5))

    def to_grid(values):
        g = np.full((lats.size, lons.size), np.nan)
        g[iy, ix] = values
        return g

    lon_e = np.concatenate([lons - dh / 2, [lons[-1] + dh / 2]])
    lat_e = np.concatenate([lats - dh / 2, [lats[-1] + dh / 2]])
    extent = [float(lons.min() - dh / 2), float(lons.max() + dh / 2),
              float(lats.min() - dh / 2), float(lats.max() + dh / 2)]
    return lons, lats, lon_e, lat_e, to_grid, extent, dh


def _basemap(ax, extent, left_labels=True):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor=COLORS["ocean"], zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=COLORS["land"], zorder=0)
    # white casing under a dark coastline so it survives over light and dark cells
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=1.4, edgecolor="white", zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, edgecolor=COLORS["coast"], zorder=5)
    gl = ax.gridlines(draw_labels=True, alpha=0.45, linewidth=0.4, color="#AEB9C6", linestyle=(0, (2, 3)))
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left_labels
    gl.xlabel_style = {"size": 8, "color": COLORS["muted"]}
    gl.ylabel_style = {"size": 8, "color": COLORS["muted"]}
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(COLORS["neat"])
        sp.set_linewidth(0.9)


def _mag_sizes(mags, mc, base=9.0):
    return base * 2.0 ** ((np.asarray(mags) - mc) / 0.5)


def _map_colorbar(fig, ax, mappable, label, ticks=None, fmt=None):
    """Native colorbar attached to the map axes (auto-sized to the map height)."""
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02, extend="both", extendfrac=0.03)
    if ticks is not None:
        cb.set_ticks(ticks)
    if fmt is not None:
        cb.ax.yaxis.set_major_formatter(fmt)
    cb.set_label(label, fontsize=9.3)
    cb.ax.tick_params(labelsize=8.2)
    cb.outline.set_edgecolor(COLORS["neat"])
    cb.outline.set_linewidth(0.7)
    return cb


def plot_temporal_diagnostic(
    ax: plt.Axes,
    simulations: pd.DataFrame,
    observed: pd.DataFrame,
    forecast_start: pd.Timestamp,
    duration_days: float,
    n_catalogs: int,
) -> None:
    _style_panel(ax)
    x_days, median, p05, p95 = compute_temporal_envelope(
        simulations, forecast_start, duration_days, n_catalogs
    )
    ax.fill_between(x_days, p05, p95, color=COLORS["band"], alpha=0.6, label="Simulated 5–95%")
    ax.plot(x_days, median, color=COLORS["median"], linewidth=2.0, label="Simulated median")

    if len(observed) > 0:
        obs = observed.copy()
        if not pd.api.types.is_datetime64_any_dtype(obs["time"]):
            obs["time"] = pd.to_datetime(obs["time"])
        obs_days = np.sort(
            ((obs["time"] - forecast_start).dt.total_seconds() / 86400.0).to_numpy()
        )
        # exact cumulative step at true event times (prepend 0, extend to horizon)
        step_x = np.concatenate(([0.0], obs_days, [float(duration_days)]))
        step_y = np.concatenate((np.arange(obs_days.size + 1), [obs_days.size]))
        ax.step(step_x, step_y, where="post", color=COLORS["observed"], linewidth=2.2, label="Observed")

    ax.set_title("Temporal Accumulation")
    ax.set_xlabel("Days since forecast start")
    ax.set_ylabel("Cumulative events")
    ax.set_xlim(0, duration_days)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")


def plot_magnitude_distribution_diagnostic(
    ax: plt.Axes,
    simulations: pd.DataFrame,
    observed: pd.DataFrame,
    mc: float,
) -> None:
    _style_panel(ax)
    mag_max = mc + 1.0
    if len(simulations) > 0:
        mag_max = max(mag_max, float(simulations["magnitude"].max()) + 0.2)
    if len(observed) > 0:
        mag_max = max(mag_max, float(observed["magnitude"].max()) + 0.2)
    bins = np.arange(mc, mag_max + 0.2, 0.1)
    if len(simulations) > 0:
        ax.hist(simulations["magnitude"], bins=bins, density=True, color=COLORS["band"],
                edgecolor=COLORS["simulated"], linewidth=0.4, label="Simulated")
    if len(observed) > 0:
        ax.hist(observed["magnitude"], bins=bins, density=True, histtype="step",
                linewidth=2.2, color=COLORS["observed"], label="Observed")
    ax.set_yscale("log")
    ax.set_title("Magnitude Distribution")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Density (log)")
    ax.legend(loc="upper right")


def plot_spatial_diagnostic(ax, horizon_eval, mc):
    if horizon_eval.forecast.expected_rates is None:
        horizon_eval.forecast.get_expected_rates()
    region = horizon_eval.forecast.region
    rates = np.asarray(horizon_eval.forecast.expected_rates.spatial_counts(), dtype=float).ravel()
    lons, lats, lon_e, lat_e, to_grid, extent, dh = _region_grid(region)
    _basemap(ax, extent, left_labels=True)

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(alpha=0)
    pos = rates[rates > 0]
    if pos.size:
        vmin = max(np.percentile(pos, 2), 1e-3)
        vmax = np.percentile(pos, 99.5)
        # fill domain continuously (zero/very-low -> dark low end) and smooth
        # across cell centres so the field reads as a continuous heatmap
        rfill = np.where(rates > 0, rates, vmin)
        pm = ax.pcolormesh(lons, lats, to_grid(rfill), cmap=cmap,
                           norm=LogNorm(vmin=vmin, vmax=vmax, clip=True),
                           transform=ccrs.PlateCarree(), shading="gouraud", rasterized=True, zorder=2)
        _map_colorbar(ax.figure, ax, pm, "Expected count / cell (log)",
                      ticks=LogLocator(base=10), fmt=LogFormatterMathtext(base=10))

    observed = horizon_eval.filtered_observed
    if len(observed) > 0:
        sc = ax.scatter(observed["longitude"], observed["latitude"],
                        s=_mag_sizes(observed["magnitude"], mc),
                        facecolor="none", edgecolor="white", linewidth=1.0, alpha=0.95,
                        transform=ccrs.PlateCarree(), zorder=6)
        sc.set_path_effects([pe.withStroke(linewidth=2.0, foreground="#101418")])

    ax.set_title(f"Expected Spatial Rate & Observed Events   ·   {dh:g}° cells, "
                 f"{horizon_eval.duration_days:g} d")


def plot_spatial_residual_diagnostic(ax, horizon_eval):
    if horizon_eval.forecast.expected_rates is None:
        horizon_eval.forecast.get_expected_rates()
    region = horizon_eval.forecast.region
    rates = np.asarray(horizon_eval.forecast.expected_rates.spatial_counts(), dtype=float).ravel()
    obs_spatial = np.asarray(horizon_eval.observed_catalog.spatial_counts(), dtype=float).ravel()
    residual = obs_spatial - rates
    lons, lats, lon_e, lat_e, to_grid, extent, dh = _region_grid(region)
    _basemap(ax, extent, left_labels=True)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(alpha=0)
    active = (rates > 0) | (obs_spatial > 0)
    if active.any():
        # hide negligible over-forecast speckle (cells with no obs and tiny rate)
        show = active & ~((obs_spatial == 0) & (rates < 0.02))
        vlim = max(np.percentile(np.abs(residual[active]), 98), 1.0)
        grid = to_grid(np.where(show, residual, np.nan))
        pm = ax.pcolormesh(lon_e, lat_e, grid, cmap=cmap,
                           norm=TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim),
                           transform=ccrs.PlateCarree(), shading="flat", rasterized=True, zorder=2)
        _map_colorbar(ax.figure, ax, pm, "Observed − Expected count / cell")

    ax.set_title("Spatial Residuals   ·   red = under, blue = over-forecast")


def plot_count_bias_decomposition_panel(ax, diagnostics, show_legend=True):
    _style_panel(ax)
    ax.grid(True, axis="x", alpha=0.9)
    ax.grid(False, axis="y")
    es = diagnostics["expected_count_in_observed_cells"]
    ee = diagnostics["expected_count_in_empty_cells"]
    osup = diagnostics["observed_count_in_positive_rate_cells"]
    oz = diagnostics["observed_count_in_zero_rate_cells"]
    ax.barh([1], [es], height=0.5, color=COLORS["cb_exp"], label="Exp · observed")
    ax.barh([1], [ee], left=[es], height=0.5, color=COLORS["cb_emp"], label="Exp · empty")
    ax.barh([0], [osup], height=0.5, color=COLORS["ob_sup"], label="Obs · forecast")
    ax.barh([0], [oz], left=[osup], height=0.5, color=COLORS["ob_zero"], label="Obs · zero-rate")
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Simulated\nmean", "Observed"])
    ax.set_xlabel("Event count")
    ax.set_title("Count-Bias Decomp.")
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.4, ncol=2,
                  handlelength=1.1, columnspacing=1.0)


def plot_test_panel(ax, result, label, one_sided_lower=None, show_obs_caption=False):
    # Native pyCSEP test-distribution histogram -- kept in the original CSEP
    # style on purpose (the four N/M/S/PL panels are not restyled).
    if result is None or len(getattr(result, "test_distribution", [])) == 0:
        ax.axis("off")
        ax.set_title(label)
        status = "not-run" if result is None else getattr(result, "status", "no data")
        ax.text(0.5, 0.5, f"status: {status}", ha="center", va="center", color=COLORS["muted"])
        return
    csep_plots.plot_test_distribution(result, ax=ax, show=False, legend=True)
    ax.set_title(label)
    if getattr(result, "status", "normal") != "normal":
        ax.text(0.98, 0.96, f"status: {result.status}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=COLORS["bad"],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["bad"]))


def plot_dashboard_header(ax, metadata, horizon_eval):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
                 facecolor=COLORS["head_bg"], edgecolor="none", transform=ax.transAxes, clip_on=False))
    ax.text(0.016, 0.66, "pyCSEP Catalog Evaluation Dashboard", fontsize=18, fontweight="bold",
            color=COLORS["head_tx"], transform=ax.transAxes, va="center")
    start = horizon_eval.forecast_start
    end = (start + timedelta(days=float(horizon_eval.duration_days))).date()
    ax.text(0.016, 0.24,
            f"{metadata['run_label']}   ·   {horizon_eval.duration_days:g}-day horizon   ·   "
            f"M ≥ {float(metadata['mc']):g}   ·   {horizon_eval.diagnostics['n_catalogs']} catalogs   ·   "
            f"{start.date()} → {end}   ·   NZ testing region",
            fontsize=10, color="#C7D2DD", transform=ax.transAxes, va="center")


def plot_diagnostics_card(ax, horizon_eval):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Diagnostics")
    diag = horizon_eval.diagnostics
    labs = ["N", "M", "S", "PL"]
    n = len(TEST_CONFIGS)
    w = 1.0 / n
    for i, config in enumerate(TEST_CONFIGS):
        result = horizon_eval.results[config["key"]]
        if result is None:
            col, mark = COLORS["neutral"], "N/A"
        else:
            summary = summarize_result(result, one_sided_lower=config["one_sided_lower"])
            cons = summary["consistent"]
            col = COLORS["good"] if cons else (COLORS["bad"] if cons is not None else COLORS["neutral"])
            mark = "PASS" if cons else ("FAIL" if cons is not None else "N/A")
        x0 = i * w + 0.01
        ax.add_patch(FancyBboxPatch((x0, 0.85), w - 0.02, 0.13, boxstyle="round,pad=0.004,rounding_size=0.02",
                     facecolor=col, edgecolor="none", alpha=0.96, transform=ax.transAxes))
        ax.text(x0 + (w - 0.02) / 2, 0.94, labs[i], ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(x0 + (w - 0.02) / 2, 0.875, mark, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color="white", transform=ax.transAxes)
    kpis = [
        ("Observed events", f"{diag['observed_filtered_count']}"),
        ("Sim N mean / median", f"{diag['mean_simulated_filtered_count']:.0f} / {diag['median_simulated_filtered_count']:.0f}"),
        ("Obs / Sim ratio", f"{diag['observed_to_sim_mean_ratio']:.2f}"),
        ("Count bias (sim−obs)", f"{diag['sim_minus_obs_mean_bias']:.0f}"),
        ("Obs M mean / max", f"{diag['observed_mean_magnitude']:.2f} / {diag['observed_max_magnitude']:.2f}"),
        ("Obs in zero-rate cells", f"{diag['observed_count_in_zero_rate_cells']:.0f}"),
        ("Mean |resid| / cell", f"{diag['spatial_mean_abs_residual']:.3f}"),
        ("Exp. in empty cells", f"{diag['expected_count_in_empty_cells']:.0f} ({diag['expected_count_in_empty_cells_fraction']:.0%})"),
    ]
    y = 0.745
    dy = 0.092
    for k, v in kpis:
        ax.text(0.015, y, k, ha="left", va="center", fontsize=9.0, color=COLORS["muted"], transform=ax.transAxes)
        ax.text(0.985, y, v, ha="right", va="center", fontsize=10.3, fontweight="bold",
                color=COLORS["ink"], family=MONO, transform=ax.transAxes)
        ax.plot([0.015, 0.985], [y - dy / 2, y - dy / 2], color=COLORS["hair"], lw=0.7, transform=ax.transAxes)
        y -= dy


def plot_horizon_dashboard(
    horizon_eval: HorizonEvaluation,
    metadata: dict[str, Any],
    output_path: str,
) -> None:
    apply_dashboard_style()
    mc = float(metadata["mc"])

    # Manual layout -- constrained_layout collapses with cartopy GeoAxes.
    fig = plt.figure(figsize=(18, 15.0), dpi=200)
    gs = fig.add_gridspec(4, 12, height_ratios=[0.30, 0.92, 2.55, 1.18],
                          left=0.045, right=0.965, top=0.955, bottom=0.045,
                          hspace=0.45, wspace=0.85)

    plot_dashboard_header(fig.add_subplot(gs[0, :]), metadata, horizon_eval)

    for i, config in enumerate(TEST_CONFIGS):
        plot_test_panel(
            fig.add_subplot(gs[1, i * 3:i * 3 + 3]),
            horizon_eval.results[config["key"]],
            config["label"],
            config["one_sided_lower"],
            show_obs_caption=(i == 0),
        )

    axm1 = fig.add_subplot(gs[2, 0:6], projection=ccrs.Mercator())
    axm2 = fig.add_subplot(gs[2, 6:12], projection=ccrs.Mercator())
    plot_spatial_diagnostic(axm1, horizon_eval, mc)
    plot_spatial_residual_diagnostic(axm2, horizon_eval)

    plot_temporal_diagnostic(
        fig.add_subplot(gs[3, 0:4]),
        horizon_eval.filtered_simulations,
        horizon_eval.filtered_observed,
        horizon_eval.forecast_start,
        horizon_eval.duration_days,
        horizon_eval.diagnostics["n_catalogs"],
    )
    plot_magnitude_distribution_diagnostic(
        fig.add_subplot(gs[3, 4:7]),
        horizon_eval.filtered_simulations,
        horizon_eval.filtered_observed,
        mc,
    )
    plot_count_bias_decomposition_panel(fig.add_subplot(gs[3, 7:9]), horizon_eval.diagnostics)
    plot_diagnostics_card(fig.add_subplot(gs[3, 9:12]), horizon_eval)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_overview_summary_frame(horizon_evaluations: list[HorizonEvaluation]) -> pd.DataFrame:
    records = []
    for horizon_eval in horizon_evaluations:
        row = {
            "duration_days": float(horizon_eval.duration_days),
            "forecast_start": horizon_eval.forecast_start,
            "forecast_end": horizon_eval.forecast_end,
            **horizon_eval.diagnostics,
        }
        for config in TEST_CONFIGS:
            prefix = config["key"].replace("_test", "")
            summary = summarize_result(
                horizon_eval.results[config["key"]],
                one_sided_lower=config["one_sided_lower"],
            )
            row.update(
                {
                    f"{prefix}_status": summary["status"],
                    f"{prefix}_consistent": summary["consistent"],
                    f"{prefix}_q_lower": summary["quantile_lower"],
                    f"{prefix}_q_upper": summary["quantile_upper"],
                    f"{prefix}_observed_stat": summary["observed_statistic"],
                    f"{prefix}_distribution_mean": summary["distribution_mean"],
                    f"{prefix}_distribution_std": summary["distribution_std"],
                }
            )
        records.append(row)
    return pd.DataFrame.from_records(records).sort_values("duration_days")


def compute_calibration_results(
    horizon_evaluations: list[HorizonEvaluation],
) -> dict[str, Any]:
    calibration_results = {}
    for config in TEST_CONFIGS:
        eval_results = [
            horizon_eval.results[config["key"]]
            for horizon_eval in horizon_evaluations
            if horizon_eval.results[config["key"]] is not None
        ]
        if len(eval_results) < 2:
            calibration_results[config["key"]] = None
            continue
        calibration_results[config["key"]] = catalog_evaluations.calibration_test(eval_results)
    return calibration_results


def plot_consistency_overview(
    horizon_evaluations: list[HorizonEvaluation],
    summary_df: pd.DataFrame,
    calibration_results: dict[str, Any],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(17, 22))
    axes_flat = axes.flatten()

    for idx, config in enumerate(TEST_CONFIGS):
        eval_results = [
            horizon_eval.results[config["key"]]
            for horizon_eval in horizon_evaluations
            if horizon_eval.results[config["key"]] is not None
        ]
        ax = axes_flat[idx]
        if not eval_results:
            ax.axis("off")
            continue
        csep_plots.plot_consistency_test(
            eval_results,
            one_sided_lower=config["one_sided_lower"],
            ax=ax,
            show=False,
            legend=(idx == 0),
            title=config["label"],
        )
        calibration = calibration_results.get(config["key"])
        if calibration is not None:
            ax.set_title(
                f"{config['label']}\ncalibration p={float(calibration.quantile):.3f}"
            )

    ax_counts = axes_flat[4]
    durations = summary_df["duration_days"].to_numpy()
    observed = summary_df["observed_filtered_count"].to_numpy()
    simulated = summary_df["mean_simulated_filtered_count"].to_numpy()
    lower = summary_df["p05_simulated_filtered_count"].to_numpy()
    upper = summary_df["p95_simulated_filtered_count"].to_numpy()
    err = np.vstack([simulated - lower, upper - simulated])

    ax_counts.errorbar(
        durations,
        simulated,
        yerr=err,
        fmt="o-",
        color=COLORS["simulated"],
        linewidth=2,
        capsize=4,
        label="Simulated mean ± 5-95%",
    )
    ax_counts.plot(
        durations,
        observed,
        "s-",
        color=COLORS["observed"],
        linewidth=2,
        label="Observed",
    )
    ax_counts.set_title("Observed vs Simulated Count by Horizon")
    ax_counts.set_xlabel("Forecast Horizon (days)")
    ax_counts.set_ylabel("Event Count")
    ax_counts.grid(True, alpha=0.25)
    ax_counts.legend(loc="upper left")

    ax_ratio = axes_flat[5]
    ratio = summary_df["observed_to_sim_mean_ratio"].to_numpy()
    ax_ratio.plot(
        durations,
        ratio,
        "o-",
        color=COLORS["accent"],
        linewidth=2,
        label="Observed / simulated mean",
    )
    ax_ratio.axhline(1.0, color=COLORS["muted"], linestyle="--", linewidth=1)
    ax_ratio.set_title("Count Ratio by Horizon")
    ax_ratio.set_xlabel("Forecast Horizon (days)")
    ax_ratio.set_ylabel("Observed / Simulated Mean")
    ax_ratio.grid(True, alpha=0.25)
    ax_ratio.legend(loc="upper right")

    ax_undersample = axes_flat[6]
    width = 0.35
    ax_undersample.bar(
        durations - width / 2,
        summary_df["zero_rate_observed_cells"].to_numpy(),
        width=width,
        color=COLORS["observed"],
        alpha=0.8,
        label="Observed zero-rate cells",
    )
    ax_undersample.bar(
        durations + width / 2,
        100 * summary_df["simulation_outside_region_fraction"].to_numpy(),
        width=width,
        color=COLORS["accent"],
        alpha=0.75,
        label="Sim events outside region (%)",
    )
    ax_undersample.set_title("Undersampling Diagnostics")
    ax_undersample.set_xlabel("Forecast Horizon (days)")
    ax_undersample.set_ylabel("Count / Percent")
    ax_undersample.grid(True, axis="y", alpha=0.25)
    ax_undersample.legend(loc="upper left")

    ax_decomp = axes_flat[7]
    positions = np.arange(len(summary_df))
    width = 0.36
    ax_decomp.bar(
        positions - width / 2,
        summary_df["expected_count_in_observed_cells"].to_numpy(),
        width=width,
        color=COLORS["simulated"],
        alpha=0.9,
        label="Expected in observed cells",
    )
    ax_decomp.bar(
        positions - width / 2,
        summary_df["expected_count_in_empty_cells"].to_numpy(),
        width=width,
        bottom=summary_df["expected_count_in_observed_cells"].to_numpy(),
        color=COLORS["accent"],
        alpha=0.8,
        label="Expected in empty cells",
    )
    ax_decomp.bar(
        positions + width / 2,
        summary_df["observed_count_in_positive_rate_cells"].to_numpy(),
        width=width,
        color=COLORS["good"],
        alpha=0.9,
        label="Observed in forecast-supported cells",
    )
    ax_decomp.bar(
        positions + width / 2,
        summary_df["observed_count_in_zero_rate_cells"].to_numpy(),
        width=width,
        bottom=summary_df["observed_count_in_positive_rate_cells"].to_numpy(),
        color=COLORS["bad"],
        alpha=0.85,
        label="Observed in zero-rate cells",
    )
    ax_decomp.set_title("Count Bias Decomposition by Horizon")
    ax_decomp.set_xlabel("Forecast Horizon (days)")
    ax_decomp.set_ylabel("Event Count")
    ax_decomp.set_xticks(positions, labels=[f"{int(d)}" for d in durations])
    ax_decomp.grid(True, axis="y", alpha=0.25)
    ax_decomp.legend(loc="upper left", fontsize=9)

    ax_residual = axes_flat[8]
    ax_residual.plot(
        durations,
        100 * summary_df["expected_count_in_empty_cells_fraction"].to_numpy(),
        "o-",
        color=COLORS["accent"],
        linewidth=2,
        label="Expected rate in empty cells (%)",
    )
    ax_residual.plot(
        durations,
        100 * summary_df["observed_count_in_zero_rate_cells_fraction"].to_numpy(),
        "s-",
        color=COLORS["bad"],
        linewidth=2,
        label="Observed events in zero-rate cells (%)",
    )
    ax_residual.plot(
        durations,
        summary_df["spatial_mean_abs_residual"].to_numpy(),
        "^-",
        color=COLORS["muted"],
        linewidth=2,
        label="Mean |obs-exp| per cell",
    )
    ax_residual.set_title("Residual Structure by Horizon")
    ax_residual.set_xlabel("Forecast Horizon (days)")
    ax_residual.set_ylabel("Percent / Count")
    ax_residual.grid(True, alpha=0.25)
    ax_residual.legend(loc="upper right", fontsize=9)

    ax_text = axes_flat[9]
    ax_text.axis("off")
    report_lines = [
        "Summary diagnostics",
        "",
        f"Horizons evaluated: {len(summary_df)}",
        f"Catalog simulations per horizon: {int(summary_df['n_catalogs'].iloc[0])}",
        f"Mean zero-rate observed cells: {summary_df['zero_rate_observed_cells'].mean():.2f}",
        (
            "Mean observed / simulated ratio: "
            f"{summary_df['observed_to_sim_mean_ratio'].mean():.3f}"
        ),
        (
            "Mean expected count in empty cells: "
            f"{summary_df['expected_count_in_empty_cells_fraction'].mean():.1%}"
        ),
        (
            "Mean observed events in zero-rate cells: "
            f"{summary_df['observed_count_in_zero_rate_cells_fraction'].mean():.1%}"
        ),
        "",
    ]
    for config in TEST_CONFIGS:
        prefix = config["key"].replace("_test", "")
        if f"{prefix}_consistent" not in summary_df.columns:
            continue
        valid = summary_df[f"{prefix}_consistent"].dropna()
        consistent = int(valid.sum()) if len(valid) else 0
        report_lines.append(
            f"{config['label']}: {consistent}/{len(valid)} consistent windows"
        )

    ax_text.text(
        0.0,
        1.0,
        "\n".join(report_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    fig.suptitle("pyCSEP Cross-Horizon Consistency Overview", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    summary_df: pd.DataFrame,
    calibration_results: dict[str, Any],
    metadata: dict[str, Any],
    region_source: str,
    output_path: str,
) -> None:
    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NA"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value)

    lines = [
        "# pyCSEP NZ-Wide Analysis",
        "",
        f"- Run label: `{metadata['run_label']}`",
        f"- Forecast start: `{metadata['forecast_start']}`",
        f"- Region source: `{region_source}`",
        f"- Forecast horizons: `{metadata['forecast_durations_days']}` days",
        f"- Simulations per horizon: `{metadata['n_simulations']}`",
        "",
        "## Key Findings",
        "",
    ]

    ratio_mean = float(summary_df["observed_to_sim_mean_ratio"].mean())
    if ratio_mean < 0.9:
        lines.append(
            f"- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `{ratio_mean:.3f}`."
        )
    elif ratio_mean > 1.1:
        lines.append(
            f"- The ensemble underpredicts event counts on average; observed / simulated mean ratio is `{ratio_mean:.3f}`."
        )
    else:
        lines.append(
            f"- Event-count performance is roughly balanced on average; observed / simulated mean ratio is `{ratio_mean:.3f}`."
        )

    for config in TEST_CONFIGS:
        prefix = config["key"].replace("_test", "")
        valid = summary_df[f"{prefix}_consistent"].dropna()
        if len(valid) == 0:
            continue
        lines.append(
            f"- {config['label']} is consistent in `{int(valid.sum())}/{len(valid)}` horizons."
        )

    undersampled = summary_df["spatial_status"].astype(str).str.contains("undersampled").any()
    if undersampled:
        lines.append(
            "- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon."
        )
    expected_empty_fraction_mean = float(summary_df["expected_count_in_empty_cells_fraction"].mean())
    if expected_empty_fraction_mean > 0.5:
        lines.append(
            f"- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `{expected_empty_fraction_mean:.3f}`."
        )
    observed_zero_rate_fraction_mean = float(
        summary_df["observed_count_in_zero_rate_cells_fraction"].mean()
    )
    if observed_zero_rate_fraction_mean > 0:
        lines.append(
            f"- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `{observed_zero_rate_fraction_mean:.3f}`."
        )

    lines.extend([
        "",
        "## Horizon Summary",
        "",
        "| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |",
        "| --- | ---: | ---: | --- | --- | --- | --- | ---: |",
    ])

    for row in summary_df.itertuples(index=False):
        lines.append(
            "| "
            f"{fmt(row.duration_days)} | "
            f"{fmt(row.observed_filtered_count)} | "
            f"{fmt(row.mean_simulated_filtered_count)} | "
            f"{fmt(row.number_status)} ({fmt(row.number_consistent)}) | "
            f"{fmt(row.magnitude_status)} ({fmt(row.magnitude_consistent)}) | "
            f"{fmt(row.spatial_status)} ({fmt(row.spatial_consistent)}) | "
            f"{fmt(row.pseudolikelihood_status)} ({fmt(row.pseudolikelihood_consistent)}) | "
            f"{fmt(row.zero_rate_observed_cells)} |"
        )

    lines.extend(["", "## Calibration Summary", ""])
    for config in TEST_CONFIGS:
        calibration = calibration_results.get(config["key"])
        if calibration is None:
            lines.append(f"- {config['label']}: insufficient valid horizons for calibration test")
        else:
            lines.append(
                f"- {config['label']}: calibration p-value = `{float(calibration.quantile):.3f}`"
            )

    lines.extend(
        [
            "",
            "## Additional Diagnostics",
            "",
            "| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| "
            f"{fmt(row.duration_days)} | "
            f"{fmt(row.observed_to_sim_mean_ratio)} | "
            f"{fmt(row.sim_minus_obs_mean_bias)} | "
            f"{fmt(100 * row.simulation_outside_region_fraction)} | "
            f"{fmt(100 * row.observed_outside_region_fraction)} |"
        )

    lines.extend(
        [
            "",
            "## Spatial Residual Diagnostics",
            "",
            "| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| "
            f"{fmt(row.duration_days)} | "
            f"{fmt(row.expected_count_in_observed_cells)} | "
            f"{fmt(row.expected_count_in_empty_cells)} | "
            f"{fmt(row.observed_count_in_zero_rate_cells)} | "
            f"{fmt(row.spatial_mean_abs_residual)} |"
        )

    lines.extend(
        [
            "",
            "## Peak Residual Cells",
            "",
            "| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |",
            "| --- | --- | --- |",
        ]
    )
    for row in summary_df.itertuples(index=False):
        under_cell = "NA"
        if not (np.isnan(row.max_positive_residual_lon) or np.isnan(row.max_positive_residual_lat) or np.isnan(row.max_positive_residual_value)):
            under_cell = (
                f"({fmt(row.max_positive_residual_lon)}, {fmt(row.max_positive_residual_lat)}, "
                f"{fmt(row.max_positive_residual_value)})"
            )
        over_cell = "NA"
        if not (np.isnan(row.max_negative_residual_lon) or np.isnan(row.max_negative_residual_lat) or np.isnan(row.max_negative_residual_value)):
            over_cell = (
                f"({fmt(row.max_negative_residual_lon)}, {fmt(row.max_negative_residual_lat)}, "
                f"{fmt(row.max_negative_residual_value)})"
            )
        lines.append(
            f"| {fmt(row.duration_days)} | {under_cell} | {over_cell} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.",
            "- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.",
            "- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.",
            "- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.",
            "- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def evaluate_horizon(
    metadata: dict[str, Any],
    duration_days: float,
    region: regions.CartesianGrid2D,
) -> HorizonEvaluation:
    duration_key = str(int(duration_days)) if float(duration_days).is_integer() else str(duration_days)
    simulations = pd.read_csv(metadata["simulation_files"][duration_key], parse_dates=["time"])
    observed = pd.read_csv(metadata["observed_files"][duration_key], parse_dates=["time"])

    forecast_start = pd.Timestamp(metadata["forecast_start"])
    forecast_end = forecast_start + timedelta(days=float(duration_days))

    forecast, filtered_simulations = build_catalog_forecast(
        simulations,
        region=region,
        n_catalogs=int(metadata["n_simulations"]),
        name=f"{metadata['run_label']}_{duration_key}d",
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )
    observed_catalog = build_csep_catalog(
        observed,
        region=region,
        name=f"{metadata['run_label']}_observed_{duration_key}d",
    )
    filtered_observed = observed_catalog.to_dataframe()
    if len(filtered_observed) > 0:
        filtered_observed["time"] = pd.to_datetime(filtered_observed["origin_time"], unit="ms", utc=True)
        filtered_observed["time"] = filtered_observed["time"].dt.tz_localize(None)
        filtered_observed = filtered_observed[["longitude", "latitude", "magnitude", "time"]]
    else:
        filtered_observed = pd.DataFrame(columns=["longitude", "latitude", "magnitude", "time"])

    results = {
        "number_test": catalog_evaluations.number_test(forecast, observed_catalog),
        "magnitude_test": (
            catalog_evaluations.resampled_magnitude_test(forecast, observed_catalog)
            if hasattr(catalog_evaluations, "resampled_magnitude_test")
            else catalog_evaluations.magnitude_test(forecast, observed_catalog)
        ),
        "spatial_test": catalog_evaluations.spatial_test(forecast, observed_catalog),
        "pseudolikelihood_test": catalog_evaluations.pseudolikelihood_test(forecast, observed_catalog),
    }

    for result in results.values():
        if result is not None:
            result.sim_name = f"{duration_days:g} d"

    if forecast.expected_rates is None:
        forecast.get_expected_rates()
    spatial_rates = np.asarray(forecast.expected_rates.spatial_counts(), dtype=float).ravel()
    obs_spatial = np.asarray(observed_catalog.spatial_counts(), dtype=float).ravel()
    midpoints = forecast.region.midpoints()
    residual = obs_spatial - spatial_rates
    zero_rate_obs_cells = int(np.sum((obs_spatial > 0) & (spatial_rates == 0)))
    expected_in_observed_cells = float(spatial_rates[obs_spatial > 0].sum())
    expected_in_empty_cells = float(spatial_rates[obs_spatial == 0].sum())
    observed_in_positive_rate_cells = float(obs_spatial[spatial_rates > 0].sum())
    observed_in_zero_rate_cells = float(obs_spatial[spatial_rates == 0].sum())
    max_positive_idx = int(np.argmax(residual)) if residual.size else 0
    max_negative_idx = int(np.argmin(residual)) if residual.size else 0
    max_positive_value = float(residual[max_positive_idx]) if residual.size else np.nan
    max_negative_value = float(residual[max_negative_idx]) if residual.size else np.nan

    event_counts = np.array([catalog.event_count for catalog in forecast.catalogs], dtype=float)

    diagnostics = {
        "n_catalogs": int(metadata["n_simulations"]),
        "observed_raw_count": int(len(observed)),
        "observed_filtered_count": int(observed_catalog.event_count),
        "observed_outside_region_count": int(len(observed) - observed_catalog.event_count),
        "observed_outside_region_fraction": float(
            (len(observed) - observed_catalog.event_count) / max(len(observed), 1)
        ),
        "simulation_raw_event_count": int(len(simulations)),
        "simulation_filtered_event_count": int(len(filtered_simulations)),
        "simulation_outside_region_count": int(len(simulations) - len(filtered_simulations)),
        "simulation_outside_region_fraction": float(
            (len(simulations) - len(filtered_simulations)) / max(len(simulations), 1)
        ),
        "mean_simulated_filtered_count": float(np.mean(event_counts)),
        "median_simulated_filtered_count": float(np.median(event_counts)),
        "p25_simulated_filtered_count": float(np.percentile(event_counts, 25)),
        "p05_simulated_filtered_count": float(np.percentile(event_counts, 5)),
        "p75_simulated_filtered_count": float(np.percentile(event_counts, 75)),
        "p95_simulated_filtered_count": float(np.percentile(event_counts, 95)),
        "sim_minus_obs_mean_bias": float(np.mean(event_counts) - observed_catalog.event_count),
        "observed_to_sim_mean_ratio": float(
            observed_catalog.event_count / max(np.mean(event_counts), 1e-12)
        ),
        "zero_rate_observed_cells": zero_rate_obs_cells,
        "expected_count_in_observed_cells": expected_in_observed_cells,
        "expected_count_in_empty_cells": expected_in_empty_cells,
        "expected_count_in_empty_cells_fraction": float(
            expected_in_empty_cells / max(spatial_rates.sum(), 1e-12)
        ),
        "observed_count_in_positive_rate_cells": observed_in_positive_rate_cells,
        "observed_count_in_zero_rate_cells": observed_in_zero_rate_cells,
        "observed_count_in_zero_rate_cells_fraction": float(
            observed_in_zero_rate_cells / max(observed_catalog.event_count, 1e-12)
        ),
        "spatial_l1_residual": float(np.abs(residual).sum()),
        "spatial_mean_abs_residual": float(np.mean(np.abs(residual))),
        "spatial_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "max_positive_residual_value": max_positive_value if max_positive_value > 0 else np.nan,
        "max_positive_residual_lon": float(midpoints[max_positive_idx, 0])
        if residual.size and max_positive_value > 0
        else np.nan,
        "max_positive_residual_lat": float(midpoints[max_positive_idx, 1])
        if residual.size and max_positive_value > 0
        else np.nan,
        "max_negative_residual_value": max_negative_value if max_negative_value < 0 else np.nan,
        "max_negative_residual_lon": float(midpoints[max_negative_idx, 0])
        if residual.size and max_negative_value < 0
        else np.nan,
        "max_negative_residual_lat": float(midpoints[max_negative_idx, 1])
        if residual.size and max_negative_value < 0
        else np.nan,
        "observed_mean_magnitude": float(observed["magnitude"].mean()) if len(observed) > 0 else np.nan,
        "observed_max_magnitude": float(observed["magnitude"].max()) if len(observed) > 0 else np.nan,
        "simulated_mean_magnitude": float(filtered_simulations["magnitude"].mean()) if len(filtered_simulations) > 0 else np.nan,
        "simulated_max_magnitude": float(filtered_simulations["magnitude"].max()) if len(filtered_simulations) > 0 else np.nan,
    }

    return HorizonEvaluation(
        duration_days=float(duration_days),
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        forecast=forecast,
        observed_catalog=observed_catalog,
        filtered_simulations=filtered_simulations,
        filtered_observed=filtered_observed,
        results=results,
        diagnostics=diagnostics,
    )


def run_analysis_from_metadata(
    metadata_path: str,
    region_source: str = DEFAULT_REGION_SOURCE,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    mag_bin: float = DEFAULT_MAG_BIN,
    max_mag: float | None = None,
    output_dir: str | None = None,
) -> AnalysisOutputs:
    metadata = load_metadata(metadata_path)

    inferred_max_mag = infer_max_magnitude(metadata, mag_bin, max_mag)
    region = build_region(
        metadata,
        region_source=region_source,
        grid_spacing=grid_spacing,
        mag_bin=mag_bin,
        max_mag=inferred_max_mag,
    )

    output_dir = ensure_dir(
        output_dir or os.path.join(os.path.dirname(metadata_path), DEFAULT_OUTPUT_SUBDIR)
    )

    horizon_evaluations = []
    dashboard_paths = []
    for duration_days in metadata["forecast_durations_days"]:
        horizon_eval = evaluate_horizon(metadata, float(duration_days), region)
        horizon_evaluations.append(horizon_eval)

        dashboard_path = os.path.join(
            output_dir, f"pycsep_dashboard_{int(float(duration_days))}days.png"
        )
        plot_horizon_dashboard(horizon_eval, metadata, dashboard_path)
        dashboard_paths.append(dashboard_path)

    summary_df = build_overview_summary_frame(horizon_evaluations)
    calibration_results = compute_calibration_results(horizon_evaluations)

    overview_path = os.path.join(output_dir, "pycsep_consistency_overview.png")
    plot_consistency_overview(
        horizon_evaluations,
        summary_df,
        calibration_results,
        overview_path,
    )

    summary_csv_path = os.path.join(output_dir, "pycsep_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    results_json_path = os.path.join(output_dir, "pycsep_results.json")
    results_payload = {
        "run_label": metadata["run_label"],
        "region_source": region_source,
        "max_mag": inferred_max_mag,
        "grid_spacing": grid_spacing,
        "horizons": [
            {
                "duration_days": horizon_eval.duration_days,
                "diagnostics": to_serializable(horizon_eval.diagnostics),
                "results": {
                    key: None if value is None else to_serializable(value.to_dict())
                    for key, value in horizon_eval.results.items()
                },
            }
            for horizon_eval in horizon_evaluations
        ],
        "calibration": {
            key: None if value is None else to_serializable(value.to_dict())
            for key, value in calibration_results.items()
        },
    }
    with open(results_json_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    report_path = os.path.join(output_dir, "pycsep_report.md")
    write_markdown_report(
        summary_df,
        calibration_results,
        metadata,
        region_source,
        report_path,
    )

    outputs = AnalysisOutputs(
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        results_json_path=results_json_path,
        report_path=report_path,
        overview_path=overview_path,
        dashboard_paths=dashboard_paths,
        metadata_path=metadata_path,
    )
    update_metadata_with_analysis(
        metadata_path,
        {
            "status": "completed",
            "region_source": region_source,
            "grid_spacing": grid_spacing,
            "mag_bin": mag_bin,
            "max_mag": inferred_max_mag,
            "output_dir": output_dir,
            "summary_csv": summary_csv_path,
            "results_json": results_json_path,
            "report_md": report_path,
            "overview_png": overview_path,
            "dashboard_pngs": dashboard_paths,
        },
    )
    return outputs


def main() -> None:
    args = parse_args()
    outputs = run_analysis_from_metadata(
        metadata_path=args.metadata,
        region_source=args.region_source,
        grid_spacing=args.grid_spacing,
        mag_bin=args.mag_bin,
        max_mag=args.max_mag,
        output_dir=args.output_dir,
    )

    metadata = load_metadata(args.metadata)
    print(f"pyCSEP analysis complete for {metadata['run_label']}")
    print(f"Summary CSV: {outputs.summary_csv_path}")
    print(f"Results JSON: {outputs.results_json_path}")
    print(f"Report: {outputs.report_path}")
    print(f"Overview figure: {outputs.overview_path}")


if __name__ == "__main__":
    main()
