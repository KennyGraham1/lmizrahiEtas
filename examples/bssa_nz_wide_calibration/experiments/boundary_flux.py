#!/usr/bin/env python3
"""Experiment H5: open-boundary triggering-flux correction.

Hypothesis
----------
Closed-domain branching-ratio (eta) estimates at an open plate boundary are
biased *low* because triggering flux that an ETAS parent would deposit east of
the artificial boundary is simply discarded: the offspring that would have
landed outside the source/target domain are never counted, so the fitted
productivity is suppressed. The bias should be predictable from the fitted
spatial kernel, and an estimate that *adds back* the leaked flux ought to be
roughly constant as the eastern boundary L moves seaward.

Known anchors (from the published sweep and the dateline experiment):
    eta(L = 180 E, published bounded domain) = 0.969
    eta(L = 184 E, dateline-extended domain) = 1.000

Method
------
PART A -- empirical eta(L)  [PRIMARY RESULT]
    Refit the 2000-window model (mc = 4.1, auxiliary 1990, training start 2000)
    on the *dateline* catalog (``nzcat_dateline.csv``, unwrapped longitudes so
    great-circle distances stay correct) restricted to a sequence of eastern
    longitude cutoffs L. For each L we build a truncated polygon by clipping the
    eastern edge of the 165--184 E lat/lon box to L (every vertex with lon > L is
    moved to lon = L), validate the ring with shapely, save it to a temp .npy of
    [lat, lon] pairs, and pass it as ``shape_coords``. Each refit records the
    branching ratio n, the number of target events, and an asymptotic 95% CI
    (the Hessian + delta-method machinery reused verbatim from
    ``branching_uncertainty.py``). n is expected to rise monotonically with L.

PART B -- kernel-leakage model  [FIRST-ORDER MODEL, secondary]
    Take the fitted kernel from the WIDEST fit (L = 184). The 2-D spatial
    offspring PDF of a parent of magnitude m is, in the formulation used by this
    codebase (see ``triggering_kernel`` / ``branching_ratio`` in
    ``etas/inversion.py``):

        f(x, y) propto 1 / (x^2 + y^2 + sigma2(m))^(1 + rho),
        sigma2(m) = d * exp(gamma * (m - mc)),   d = 10**log10_d

    (x, y in km; the kernel's spatial argument is squared distance in km^2.)
    Its full-plane integral is  Z = pi / rho * sigma2^(-rho), matching the
    ``k_factor = k0 * pi/rho * d^(-rho)`` term in ``branching_ratio``.

    For a parent at perpendicular distance s WEST of a straight N--S boundary,
    the leaked fraction (offspring landing east of the boundary, x > s) is

        leak(s, m) = [ int_{x>s} int_{-inf}^{inf} f dy dx ] / Z.

    We evaluate this by reducing the inner y-integral analytically (a Beta-type
    integral) and integrating the remaining 1-D x-integral numerically with
    scipy. For the truncation at L we then average leak over the actual catalog
    sources within ~100 km west of the boundary, weighted by ETAS productivity
    exp(a * (m - mc)) (the productivity factor that appears in the kernel and in
    ``branching_ratio``; note this codebase uses exp(a*.), not 10**(a*.)), to get
    expected leakage(L). The boundary-corrected estimate is

        n_corr(L) = n_measured(L) / (1 - leakage(L)).

    PART B is explicitly a first-order model; PART A is the primary result.

Output
------
``tables/boundary_flux.csv`` with one row per L:
    L, n_measured, n_target, ci_low, ci_high, leakage_frac, n_corrected
plus a printed comparison of the spread (max - min) of n_measured vs n_corrected.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

import numpy as np  # noqa: E402

# Reuse the CI machinery verbatim from the branching-uncertainty experiment.
from branching_uncertainty import _numerical_hessian  # noqa: E402

TABLE_DIR = BSSA / "tables"

# Dateline catalog (unwrapped longitudes, 165--184 E) and the 165--184 E box.
DATELINE_CATALOG = str(ROOT / "input_data" / "nzcat_dateline.csv")
DATELINE_POLYGON = str(ROOT / "input_data" / "nz_polygon_dateline.npy")

# 2000-window training configuration (matches the published admissible scenario).
TIMEWINDOW_START = "2000-01-01 00:00:00"
AUXILIARY_START = "1990-01-01 00:00:00"
MC = 4.1
ORIGIN = dt.datetime(2021, 1, 1)

# Eastern-longitude cutoffs (deg E). 180 = published boundary, 184 = dateline.
DEFAULT_CUTOFFS = [178.0, 179.0, 180.0, 181.0, 182.0, 183.0, 184.0]

EARTH_RADIUS_KM = 6.3781e3  # matches ETASParameterCalculation default
BOUNDARY_BAND_KM = 100.0    # sources within this band feed the leakage average


# --------------------------------------------------------------------------- #
# Truncated-polygon construction
# --------------------------------------------------------------------------- #
def build_truncated_polygon(cutoff_lon: float,
                            source_polygon: str = DATELINE_POLYGON) -> np.ndarray:
    """Clip the eastern edge of the lat/lon box to ``cutoff_lon``.

    The source polygon is the 165--184 E rectangle (a simple lon/lat box), so
    clipping is exact: every vertex with lon > cutoff is moved to lon = cutoff.
    The result is validated as a simple closed ring with shapely.

    Returns an array of [lat, lon] vertex pairs (closed ring), as expected by
    ``shape_coords``/``np.load`` consumers in this codebase.
    """
    from shapely.geometry import Polygon

    coords = np.load(source_polygon).astype(float)  # [lat, lon]
    clipped = coords.copy()
    clipped[:, 1] = np.minimum(clipped[:, 1], float(cutoff_lon))

    # shapely wants (x=lon, y=lat); drop the duplicate closing vertex for the
    # validity check, shapely re-closes the ring itself.
    ring = clipped[:, ::-1]  # [lon, lat]
    poly = Polygon(ring[:-1] if np.allclose(ring[0], ring[-1]) else ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.area <= 0:
        raise ValueError(
            f"Truncated polygon at L={cutoff_lon} is degenerate "
            f"(valid={poly.is_valid}, area={poly.area})."
        )
    return clipped


def write_temp_polygon(coords: np.ndarray) -> str:
    """Persist a [lat, lon] polygon to a temp .npy and return its path."""
    fd, path = tempfile.mkstemp(prefix="boundary_flux_poly_", suffix=".npy")
    os.close(fd)
    np.save(path, coords)
    return path


# --------------------------------------------------------------------------- #
# PART A worker: refit at a single cutoff L and attach an asymptotic CI
# --------------------------------------------------------------------------- #
def _fit_one_cutoff(task: dict) -> dict:
    """Refit the 2000-window model on the dateline catalog truncated at L."""
    import numba
    numba.set_num_threads(task["numba_threads"])

    import numpy as np
    import run_nz_wide_forecast as fc
    from etas.inversion import (
        ETASParameterCalculation,
        branching_ratio,
        neg_log_likelihood,
        parameter_dict2array,
    )

    cutoff = float(task["cutoff"])

    # Build + persist the truncated polygon inside the worker.
    coords = build_truncated_polygon(cutoff, task["source_polygon"])
    poly_path = write_temp_polygon(coords)

    try:
        label = f"nzbf_L{int(round(cutoff))}"
        paths = fc.build_run_paths(label, ORIGIN)
        config = fc.build_inversion_config(
            paths["run_label"],
            task["catalog"],
            ORIGIN,
            AUXILIARY_START,
            TIMEWINDOW_START,
            MC,
            fc.build_initial_theta(),
            bg_term=None,
            shape_coords=poly_path,
        )
        calc = ETASParameterCalculation(config)
        calc.prepare()
        calc.invert()

        beta = float(calc.beta)
        mc_min = calc.m_ref - calc.delta_m / 2
        th = calc.theta
        theta8 = np.array([
            th["log10_k0"], th["a"], th["log10_c"], th["omega"],
            th["log10_tau"], th["log10_d"], th["gamma"], th["rho"],
        ], dtype=float)

        pij, src = calc.pij, calc.source_events

        def nll(v):
            return float(neg_log_likelihood(v, pij, src.copy(), mc_min))

        # ----- asymptotic Hessian + delta-method CI (verbatim recipe) -----
        H = _numerical_hessian(nll, theta8)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)

        def n_of(v, b):
            return float(branching_ratio(np.array([0.0, 0.0, *v]), b))

        n_point = n_of(theta8, beta)
        grad = np.zeros(8)
        step = np.maximum(np.abs(theta8) * 1e-5, 1e-7)
        for i in range(8):
            tp = theta8.copy(); tp[i] += step[i]
            tm = theta8.copy(); tm[i] -= step[i]
            grad[i] = (n_of(tp, beta) - n_of(tm, beta)) / (2 * step[i])
        var_params = float(grad @ cov @ grad)

        n_target = int(len(calc.target_events))
        se_beta = beta / np.sqrt(max(n_target, 1))
        dn_db = (n_of(theta8, beta + 1e-4) - n_of(theta8, beta - 1e-4)) / 2e-4
        se_n = float(np.sqrt(max(var_params, 0.0) + (dn_db * se_beta) ** 2))
        # ------------------------------------------------------------------

        # Full kernel dict for the leakage model (needed only at the widest L,
        # but cheap to return everywhere).
        kernel = {k: float(th[k]) for k in (
            "log10_k0", "a", "log10_c", "omega", "log10_tau",
            "log10_d", "gamma", "rho",
        )}

        return {
            "L": cutoff,
            "n_measured": n_point,
            "n_target": n_target,
            "beta": beta,
            "mc": float(calc.m_ref),
            "se_n": se_n,
            "ci_low": n_point - 1.96 * se_n,
            "ci_high": n_point + 1.96 * se_n,
            "kernel": kernel,
        }
    finally:
        try:
            os.remove(poly_path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# PART B: kernel-leakage model (first-order)
# --------------------------------------------------------------------------- #
def leak_fraction(s_km: float, m: float, kernel: dict, mc: float) -> float:
    """Fraction of a parent's offspring that land east of a N--S boundary.

    The parent sits at perpendicular distance ``s_km`` WEST of the boundary
    (s >= 0). Offspring density f(x, y) propto 1 / (x^2 + y^2 + sigma2)^(1+rho)
    in km, with sigma2 = d * exp(gamma*(m-mc)). leak = P(x > s).

    The inner y-integral of 1/(x^2 + sigma2 + y^2)^(1+rho) over the whole line is
        B / (x^2 + sigma2)^(rho + 1/2),
    where B = sqrt(pi) * Gamma(rho + 1/2) / Gamma(rho + 1) is constant in x, so it
    cancels against the full-plane normalisation. The leaked fraction reduces to

        leak(s) = [ int_{x>s} (x^2 + sigma2)^(-(rho+1/2)) dx ]
                  / [ int_{-inf}^{inf} (x^2 + sigma2)^(-(rho+1/2)) dx ].

    Both 1-D integrals are evaluated numerically with scipy.
    """
    from scipy import integrate

    d = 10.0 ** kernel["log10_d"]
    rho = kernel["rho"]
    gamma = kernel["gamma"]
    sigma2 = d * np.exp(gamma * (m - mc))
    p = rho + 0.5

    def g(x):
        return (x * x + sigma2) ** (-p)

    # Full marginal over x (closed form: sqrt(pi) Gamma(rho)/Gamma(rho+1/2) *
    # sigma2^(-rho)), but integrate numerically for transparency / robustness.
    total, _ = integrate.quad(g, -np.inf, np.inf, limit=200)
    tail, _ = integrate.quad(g, s_km, np.inf, limit=200)
    if total <= 0:
        return 0.0
    return float(np.clip(tail / total, 0.0, 1.0))


def perpendicular_distance_km(lat_deg: np.ndarray, lon_deg: np.ndarray,
                              boundary_lon: float) -> np.ndarray:
    """Great-circle E--W distance from each point to the meridian at boundary_lon.

    A small east-west separation at latitude phi spans
        R * cos(phi) * delta_lon(rad)
    on the sphere, which is the perpendicular distance to a N--S boundary.
    Positive when the point is west of the boundary (lon < boundary_lon).
    """
    dlon = np.radians(boundary_lon - lon_deg)
    return EARTH_RADIUS_KM * np.cos(np.radians(lat_deg)) * dlon


def expected_leakage(cutoff_lon: float, kernel: dict, mc: float,
                     catalog, band_km: float = BOUNDARY_BAND_KM) -> dict:
    """Productivity-weighted mean leak over sources near the boundary at L.

    Sources = dateline-catalog events in the 2000-window training span, west of
    the cutoff and within ``band_km`` of it. Each source's leak(s, m) is weighted
    by ETAS productivity exp(a * (m - mc)).
    """
    a = kernel["a"]

    train = catalog[
        (catalog["time"] >= TIMEWINDOW_START)
        & (catalog["time"] < ORIGIN)
        & (catalog["magnitude"] >= mc)
        & (catalog["longitude"] <= cutoff_lon)
    ].copy()

    s = perpendicular_distance_km(
        train["latitude"].to_numpy(),
        train["longitude"].to_numpy(),
        cutoff_lon,
    )
    near = (s >= 0.0) & (s <= band_km)
    train = train.loc[near]
    s = s[near]

    if len(train) == 0:
        return {"leakage_frac": 0.0, "n_sources_in_band": 0}

    mags = train["magnitude"].to_numpy()
    weights = np.exp(a * (mags - mc))
    leaks = np.array([leak_fraction(si, mi, kernel, mc)
                      for si, mi in zip(s, mags)])
    leakage = float(np.average(leaks, weights=weights))
    return {
        "leakage_frac": leakage,
        "n_sources_in_band": int(len(train)),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(cutoffs, workers, numba_threads, catalog_path, source_polygon,
        out_name, band_km=BOUNDARY_BAND_KM):
    import multiprocessing as mp
    import pandas as pd

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [{
        "cutoff": L,
        "catalog": catalog_path,
        "source_polygon": source_polygon,
        "numba_threads": numba_threads,
    } for L in cutoffs]

    print(f"PART A: refitting 2000-window model at {len(tasks)} eastern cutoffs "
          f"{[float(c) for c in cutoffs]} "
          f"({workers} workers x {numba_threads} numba threads)", flush=True)

    rows = []
    ctx = mp.get_context("spawn")
    if workers == 1:
        # Serial path keeps the smoke test single-process and easy to debug.
        for t in tasks:
            r = _fit_one_cutoff(t)
            rows.append(r)
            print(f"  L={r['L']:.0f}  n={r['n_measured']:.4f} "
                  f"CI[{r['ci_low']:.4f},{r['ci_high']:.4f}]  "
                  f"({r['n_target']} targets)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(_fit_one_cutoff, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    r = fut.result()
                    rows.append(r)
                    print(f"  L={r['L']:.0f}  n={r['n_measured']:.4f} "
                          f"CI[{r['ci_low']:.4f},{r['ci_high']:.4f}]  "
                          f"({r['n_target']} targets)", flush=True)
                except Exception as exc:  # pragma: no cover
                    print(f"  FAILED L={t['cutoff']}: {exc}", flush=True)

    rows.sort(key=lambda r: r["L"])

    # ---- PART B: kernel-leakage model using the WIDEST fit's kernel ----
    catalog = pd.read_csv(catalog_path, index_col=0, parse_dates=["time"])
    widest = max(rows, key=lambda r: r["L"])
    kernel = widest["kernel"]
    mc = widest["mc"]
    print(f"\nPART B (first-order model): leakage from L={widest['L']:.0f} "
          f"kernel  log10_d={kernel['log10_d']:.3f} gamma={kernel['gamma']:.3f} "
          f"rho={kernel['rho']:.3f} a={kernel['a']:.3f}", flush=True)

    out_rows = []
    for r in rows:
        leak_info = expected_leakage(r["L"], kernel, mc, catalog, band_km)
        leakage = leak_info["leakage_frac"]
        denom = max(1.0 - leakage, 1e-6)
        n_corr = r["n_measured"] / denom
        out_rows.append({
            "L": r["L"],
            "n_measured": r["n_measured"],
            "n_target": r["n_target"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "leakage_frac": leakage,
            "n_sources_in_band": leak_info["n_sources_in_band"],
            "n_corrected": n_corr,
        })
        print(f"  L={r['L']:.0f}  leakage={leakage:.4f} "
              f"({leak_info['n_sources_in_band']} sources within {band_km:.0f} km)"
              f"  n_meas={r['n_measured']:.4f} -> n_corr={n_corr:.4f}", flush=True)

    frame = pd.DataFrame(out_rows)
    out = TABLE_DIR / out_name
    frame.to_csv(out, index=False)

    spread_meas = float(frame["n_measured"].max() - frame["n_measured"].min())
    spread_corr = float(frame["n_corrected"].max() - frame["n_corrected"].min())
    print(f"\nWrote {out}")
    print(frame.to_string(index=False))
    print(f"\nSpread (max-min) of n_measured  = {spread_meas:.4f}")
    print(f"Spread (max-min) of n_corrected = {spread_corr:.4f}")
    flatter = spread_corr < spread_meas
    print(f"Boundary-corrected n is {'FLATTER' if flatter else 'NOT flatter'} "
          f"across L than measured n "
          f"({'reduced' if flatter else 'did not reduce'} spread by "
          f"{spread_meas - spread_corr:+.4f}).")
    return frame


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoffs", type=float, nargs="*", default=DEFAULT_CUTOFFS,
        help="Eastern-longitude cutoffs L (deg E). Default: 178..184.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--numba-threads", type=int, default=12)
    parser.add_argument("--catalog", default=DATELINE_CATALOG)
    parser.add_argument("--source-polygon", default=DATELINE_POLYGON)
    parser.add_argument("--band-km", type=float, default=BOUNDARY_BAND_KM)
    parser.add_argument(
        "--out", default="boundary_flux.csv",
        help="Output CSV name under tables/. Use a *_smoke.csv name for smoke runs.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: ONE inversion at L=180, serial, *_smoke.csv output.",
    )
    args = parser.parse_args()

    if args.smoke:
        cutoffs = [180.0]
        workers = 1
        out_name = "boundary_flux_smoke.csv"
    else:
        cutoffs = args.cutoffs
        workers = args.workers
        out_name = args.out

    run(
        cutoffs=cutoffs,
        workers=workers,
        numba_threads=args.numba_threads,
        catalog_path=args.catalog,
        source_polygon=args.source_polygon,
        out_name=out_name,
        band_km=args.band_km,
    )


if __name__ == "__main__":
    main()
