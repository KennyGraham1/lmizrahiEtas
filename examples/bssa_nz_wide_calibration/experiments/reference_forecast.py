#!/usr/bin/env python3
"""Recommendation 4: a smoothed-seismicity Poisson reference forecast.

The published sweep scored no reference model, so it tests consistency but not
skill. Here we build a time-stationary, spatially smoothed Poisson reference for
the 2021 origin, generate a 2,000-catalog ensemble at the same five horizons,
score it with the identical pyCSEP catalog tests used for the ETAS runs, and
compute the spatial Poisson information gain of the admissible 2000-window ETAS
forecast relative to it.

The reference is deliberately simple and standard: the training catalog
(2000--2021, M>=Mc, in polygon) is smoothed with a fixed isotropic Gaussian
kernel onto the evaluation grid, mixed with a small uniform floor so no in-domain
cell has zero rate, normalized to a spatial probability map, and combined with a
stationary Poisson temporal rate equal to the historical mean. Magnitudes are
drawn from the same exponential Gutenberg--Richter law (fitted beta) as the ETAS
baseline.

Outputs:
  simulations_nz_wide/nzref_smoothed_*/forecasts_<d>days.csv  (ensemble)
  tables/reference_comparison.csv                              (scores + IG)
"""

from __future__ import annotations

import json
import os
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

TABLE_DIR = BSSA / "tables"
OUTPUT_ROOT = EXAMPLES / "output_nz_wide"
SIM_ROOT = EXAMPLES / "simulations_nz_wide"
CATALOG_PATH = ROOT / "input_data" / "nzcat.csv"
POLYGON_PATH = ROOT / "input_data" / "nz_polygon.npy"

ETAS_LABEL = "nz_wide_calibration_window_2000_20210101_000000"
ORIGIN = dt.datetime(2021, 1, 1)
TW_START = dt.datetime(2000, 1, 1)
MC = 4.1
DURATIONS = [365, 730, 1095, 1461, 1826]
GRID_SPACING = 0.1


def load_polygon_path():
    from matplotlib.path import Path as MplPath
    coords = np.load(POLYGON_PATH)  # [lat, lon]
    return MplPath(np.column_stack([coords[:, 1], coords[:, 0]])), coords


def build_smoothed_density(sigma_deg: float, floor_frac: float):
    """Return grid edges, in-polygon cell centres, and a normalized probability map."""
    from scipy.ndimage import gaussian_filter

    poly_path, coords = load_polygon_path()
    lats, lons = coords[:, 0], coords[:, 1]

    catalog = pd.read_csv(CATALOG_PATH, index_col=0, parse_dates=["time"])
    train = catalog[(catalog["time"] >= TW_START) & (catalog["time"] < ORIGIN)
                    & (catalog["magnitude"] >= MC)].copy()
    # Restrict to polygon.
    pts = np.column_stack([train["longitude"], train["latitude"]])
    train = train[poly_path.contains_points(pts)]
    n_train = len(train)
    t_train_days = (ORIGIN - TW_START).days
    rate_per_day = n_train / t_train_days

    lon_edges = np.arange(lons.min(), lons.max() + GRID_SPACING, GRID_SPACING)
    lat_edges = np.arange(lats.min(), lats.max() + GRID_SPACING, GRID_SPACING)
    counts, _, _ = np.histogram2d(train["longitude"], train["latitude"],
                                  bins=[lon_edges, lat_edges])

    sigma_cells = sigma_deg / GRID_SPACING
    smoothed = gaussian_filter(counts, sigma=sigma_cells, mode="constant")

    # Mask to in-polygon cells.
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lon_centers, lat_centers, indexing="ij")
    cell_pts = np.column_stack([LON.ravel(), LAT.ravel()])
    in_poly = poly_path.contains_points(cell_pts).reshape(LON.shape)

    density = np.where(in_poly, smoothed, 0.0)
    if density.sum() <= 0:
        raise RuntimeError("Smoothed density is empty; check polygon/catalog.")
    density = density / density.sum()
    # Uniform floor over in-polygon cells to remove zero-rate cells.
    uniform = in_poly.astype(float)
    uniform = uniform / uniform.sum()
    prob = (1 - floor_frac) * density + floor_frac * uniform
    prob = prob / prob.sum()

    return {
        "lon_edges": lon_edges, "lat_edges": lat_edges,
        "lon_centers": lon_centers, "lat_centers": lat_centers,
        "prob": prob, "in_poly": in_poly,
        "rate_per_day": rate_per_day, "n_train": n_train,
        "t_train_days": t_train_days,
    }


def generate_reference_ensemble(density, beta, n_sims, m_max, sim_label, seed=12345):
    rng = np.random.default_rng(seed)
    prob_flat = density["prob"].ravel()
    cell_idx = np.arange(prob_flat.size)
    lon_centers = density["lon_centers"]
    lat_centers = density["lat_centers"]
    n_lon = lon_centers.size
    rate_per_day = density["rate_per_day"]

    sim_dir = SIM_ROOT / sim_label
    sim_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    def draw_mags(n):
        u = rng.uniform(size=n)
        if m_max is not None:
            norm = 1 - np.exp(-beta * (m_max - MC))
        else:
            norm = 1.0
        return (-np.log(1 - norm * u) / beta) + MC

    for duration in DURATIONS:
        expected = rate_per_day * duration
        rows = []
        for cat_id in range(n_sims):
            n_ev = rng.poisson(expected)
            if n_ev == 0:
                continue
            chosen = rng.choice(cell_idx, size=n_ev, p=prob_flat)
            ci = chosen // lat_centers.size  # lon index (prob is [lon, lat])
            cj = chosen % lat_centers.size   # lat index
            lon = lon_centers[ci] + rng.uniform(-GRID_SPACING / 2, GRID_SPACING / 2, n_ev)
            lat = lat_centers[cj] + rng.uniform(-GRID_SPACING / 2, GRID_SPACING / 2, n_ev)
            mags = np.round(draw_mags(n_ev) / 0.1) * 0.1
            secs = rng.uniform(0, duration * 86400.0, n_ev)
            times = [ORIGIN + dt.timedelta(seconds=float(s)) for s in secs]
            for k in range(n_ev):
                rows.append((lat[k], lon[k], mags[k], times[k], cat_id))
        df = pd.DataFrame(rows, columns=["latitude", "longitude", "magnitude",
                                         "time", "catalog_id"])
        df.index.name = "id"
        path = sim_dir / f"forecasts_{duration}days.csv"
        df.to_csv(path)
        paths[duration] = str(path)
        print(f"  reference {duration}d: {len(df)} events across "
              f"{df['catalog_id'].nunique()} non-empty catalogs "
              f"(expected {expected:.0f}/catalog)", flush=True)
    return paths


def score_and_compare(ref_paths, n_sims, m_max):
    """Score the reference and ETAS window_2000 on a common region; compute IG."""
    import warnings
    warnings.filterwarnings("ignore")
    from csep.core import regions
    from csep.utils import time_utils  # noqa: F401
    import run_nz_wide_pycsep_analysis as pa
    from csep.core import catalog_evaluations

    etas_sim_dir = SIM_ROOT / ETAS_LABEL
    obs_dir = OUTPUT_ROOT / ETAS_LABEL

    # Common magnitude region spanning both ensembles + observations.
    max_mag = MC
    for d in DURATIONS:
        for p in [ref_paths[d], str(etas_sim_dir / f"forecasts_{d}days.csv"),
                  str(obs_dir / f"observed_{d}days.csv")]:
            m = pd.read_csv(p, usecols=["magnitude"])["magnitude"]
            if len(m):
                max_mag = max(max_mag, float(m.max()))
    max_mag = float(np.ceil(max_mag * 10) / 10 + 0.1)
    magnitudes = regions.magnitude_bins(MC, max_mag, 0.1)
    region = pa.build_forecast_domain_region(str(POLYGON_PATH), magnitudes, GRID_SPACING)

    def score_one(sim_path, obs_path, name, d):
        sims = pd.read_csv(sim_path, parse_dates=["time"])
        obs = pd.read_csv(obs_path, parse_dates=["time"])
        fstart = pd.Timestamp(ORIGIN)
        fend = fstart + dt.timedelta(days=d)
        forecast, _ = pa.build_catalog_forecast(
            sims, region=region, n_catalogs=n_sims, name=name,
            forecast_start=fstart, forecast_end=fend)
        observed = pa.build_csep_catalog(obs, region=region, name=f"{name}_obs")
        res = {
            "number": catalog_evaluations.number_test(forecast, observed),
            "magnitude": (catalog_evaluations.resampled_magnitude_test(forecast, observed)
                          if hasattr(catalog_evaluations, "resampled_magnitude_test")
                          else catalog_evaluations.magnitude_test(forecast, observed)),
            "spatial": catalog_evaluations.spatial_test(forecast, observed),
            "pseudolikelihood": catalog_evaluations.pseudolikelihood_test(forecast, observed),
        }
        if forecast.expected_rates is None:
            forecast.get_expected_rates()
        rates = np.asarray(forecast.expected_rates.spatial_counts(), dtype=float).ravel()
        obs_spatial = np.asarray(observed.spatial_counts(), dtype=float).ravel()
        return res, rates, obs_spatial

    def consistent(result, one_sided_lower):
        summ = pa.summarize_result(result, one_sided_lower)
        return summ["consistent"], summ

    rows = []
    for d in DURATIONS:
        etas_res, etas_rates, obs_spatial = score_one(
            str(etas_sim_dir / f"forecasts_{d}days.csv"),
            str(obs_dir / f"observed_{d}days.csv"), f"etas_{d}", d)
        ref_res, ref_rates, _ = score_one(
            ref_paths[d], str(obs_dir / f"observed_{d}days.csv"), f"ref_{d}", d)

        # Poisson information gain per earthquake (ETAS vs reference).
        # Guard zero-rate cells with a tiny floor to keep the log finite.
        eps = 1e-8
        er = np.clip(etas_rates, eps, None)
        rr = np.clip(ref_rates, eps, None)
        n_obs = obs_spatial.sum()
        # (i) Rate-based IG: uses the raw expected-rate fields, so it mixes the
        # spatial allocation with the total-count difference between models.
        ll_etas = float(np.sum(obs_spatial * np.log(er) - er))
        ll_ref = float(np.sum(obs_spatial * np.log(rr) - rr))
        igpe = (ll_etas - ll_ref) / n_obs if n_obs > 0 else np.nan
        # (ii) Normalized SPATIAL IG: rescale each rate field to the observed
        # total count so the Poisson count term cancels, isolating spatial
        # allocation from the total-rate difference (the reference deliberately
        # carries an inflated stationary count, which would otherwise confound
        # the rate-based metric).
        er_n = er * (n_obs / er.sum())
        rr_n = rr * (n_obs / rr.sum())
        ll_etas_sp = float(np.sum(obs_spatial * np.log(er_n) - er_n))
        ll_ref_sp = float(np.sum(obs_spatial * np.log(rr_n) - rr_n))
        igpe_spatial = (ll_etas_sp - ll_ref_sp) / n_obs if n_obs > 0 else np.nan

        e_n, e_ns = consistent(etas_res["number"], False)
        r_n, r_ns = consistent(ref_res["number"], False)
        e_s, _ = consistent(etas_res["spatial"], True)
        r_s, _ = consistent(ref_res["spatial"], True)
        e_m, _ = consistent(etas_res["magnitude"], False)
        r_m, _ = consistent(ref_res["magnitude"], False)
        e_pl, _ = consistent(etas_res["pseudolikelihood"], True)
        r_pl, _ = consistent(ref_res["pseudolikelihood"], True)

        rows.append({
            "duration_days": d,
            "observed_count": int(n_obs),
            "etas_mean_count": float(etas_rates.sum()),
            "ref_mean_count": float(ref_rates.sum()),
            "etas_n_consistent": e_n, "ref_n_consistent": r_n,
            "etas_m_consistent": e_m, "ref_m_consistent": r_m,
            "etas_s_consistent": e_s, "ref_s_consistent": r_s,
            "etas_pl_consistent": e_pl, "ref_pl_consistent": r_pl,
            "spatial_ll_etas": ll_etas, "spatial_ll_ref": ll_ref,
            "info_gain_per_eq": igpe,
            "info_gain_per_eq_spatial": igpe_spatial,
            "info_gain_per_eq_count": igpe - igpe_spatial,
        })
        print(f"  {d}d: IG/eq rate={igpe:+.3f} spatial={igpe_spatial:+.3f}  "
              f"ETAS N={e_n} S={e_s}  REF N={r_n} S={r_s}", flush=True)

    frame = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "reference_comparison.csv"
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(frame.to_string(index=False))
    return frame


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma-deg", type=float, default=0.5)
    parser.add_argument("--floor-frac", type=float, default=0.05)
    parser.add_argument("--n-simulations", type=int, default=2000)
    parser.add_argument("--m-max", type=float, default=None,
                        help="Truncation for reference magnitudes (default: unbounded, "
                             "matching the published ETAS baseline).")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    # beta from the window_2000 fit, so the reference shares the ETAS magnitude law.
    with open(OUTPUT_ROOT / ETAS_LABEL / f"parameters_{ETAS_LABEL}.json") as fh:
        beta = float(json.load(fh)["beta"])

    sim_label = "nzref_smoothed_20210101_000000"
    density = build_smoothed_density(args.sigma_deg, args.floor_frac)
    print(f"Reference: {density['n_train']} training events, "
          f"rate {density['rate_per_day']*365.25:.1f}/yr, sigma {args.sigma_deg} deg, "
          f"beta {beta:.4f}", flush=True)

    if args.skip_generate:
        ref_paths = {d: str(SIM_ROOT / sim_label / f"forecasts_{d}days.csv")
                     for d in DURATIONS}
    else:
        ref_paths = generate_reference_ensemble(
            density, beta, args.n_simulations, args.m_max, sim_label)

    score_and_compare(ref_paths, args.n_simulations, args.m_max)


if __name__ == "__main__":
    main()
