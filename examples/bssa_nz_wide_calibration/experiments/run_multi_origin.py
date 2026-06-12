#!/usr/bin/env python3
"""Recommendation 1: branching-ratio admissibility across multiple forecast origins.

The published sweep used a single origin (1 January 2021), so it cannot separate
"long training windows are supercritical" from "this particular 1960--2021
endpoint is supercritical." Here we re-invert the three training-window-start
scenarios (1960, 1980, 2000 starts, all M_c=4.1) at four additional origins and
record the fitted branching ratio at each. Only the inversion is needed for the
admissibility question, so no forecast simulation is run.

Outputs ``tables/multi_origin_branching.csv`` with one row per (origin, window).
"""

from __future__ import annotations

import json
import os
import sys
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

TABLE_DIR = BSSA / "tables"

# (training_start, auxiliary_start) pairs, mirroring the published window scenarios.
WINDOWS = {
    "1960": ("1960-01-01 00:00:00", "1950-01-01 00:00:00"),
    "1980": ("1980-01-01 00:00:00", "1970-01-01 00:00:00"),
    "2000": ("2000-01-01 00:00:00", "1990-01-01 00:00:00"),
}
# Origins with at least one full post-origin year in the catalog (ends 2026-04-30).
# 2021 is the published origin and is read from existing parameter files.
NEW_ORIGINS = ["2013-01-01 00:00:00", "2015-01-01 00:00:00",
               "2017-01-01 00:00:00", "2019-01-01 00:00:00"]
PUBLISHED_ORIGIN = "2021-01-01 00:00:00"
MC = 4.1

# Map published window scenarios to their existing run labels (origin 2021).
PUBLISHED_LABELS = {
    "1960": "nz_wide_calibration_baseline_20210101_000000",
    "1980": "nz_wide_calibration_window_1980_20210101_000000",
    "2000": "nz_wide_calibration_window_2000_20210101_000000",
}


def _invert_one(task: dict) -> dict:
    """Run a single inversion in a worker process and return its branching ratio."""
    import numba
    numba.set_num_threads(task["numba_threads"])

    # Imported inside the worker so the spawn start method re-imports cleanly.
    import run_nz_wide_forecast as fc

    forecast_start = dt.datetime.strptime(task["origin"], "%Y-%m-%d %H:%M:%S")
    paths = fc.build_run_paths(task["experiment_name"], forecast_start)
    config = fc.build_inversion_config(
        paths["run_label"],
        fc.CATALOG_PATH,
        forecast_start,
        task["auxiliary_start"],
        task["timewindow_start"],
        MC,
        fc.build_initial_theta(),
        bg_term=None,
    )
    calculation, parameter_path = fc.load_or_run_inversion(
        config, paths["output_dir"], force_reinvert=task["force"]
    )
    with open(parameter_path) as handle:
        payload = json.load(handle)
    return {
        "origin": task["origin"][:10],
        "origin_year": task["origin"][:4],
        "window_start": task["window"],
        "n_target_events": int(payload["n_target_events"]),
        "branching_ratio": float(payload["branching_ratio"]),
        "inversion_degenerate": bool(payload["inversion_degenerate"]),
    }


def _published_rows() -> list[dict]:
    rows = []
    for window, label in PUBLISHED_LABELS.items():
        path = EXAMPLES / "output_nz_wide" / label / f"parameters_{label}.json"
        with open(path) as handle:
            payload = json.load(handle)
        rows.append({
            "origin": PUBLISHED_ORIGIN[:10],
            "origin_year": PUBLISHED_ORIGIN[:4],
            "window_start": window,
            "n_target_events": int(payload["n_target_events"]),
            "branching_ratio": float(payload["branching_ratio"]),
            "inversion_degenerate": bool(payload["inversion_degenerate"]),
        })
    return rows


def main() -> None:
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--numba-threads", type=int, default=8)
    parser.add_argument("--force", action="store_true",
                        help="Force re-inversion even if cached parameters exist.")
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for origin in NEW_ORIGINS:
        year = origin[:4]
        for window, (tw_start, aux_start) in WINDOWS.items():
            tasks.append({
                "origin": origin,
                "window": window,
                "timewindow_start": tw_start,
                "auxiliary_start": aux_start,
                "experiment_name": f"nzmo_{window}_{year}",
                "numba_threads": args.numba_threads,
                "force": args.force,
            })

    print(f"Running {len(tasks)} inversions across {len(NEW_ORIGINS)} new origins "
          f"with {args.workers} workers x {args.numba_threads} numba threads",
          flush=True)

    rows = _published_rows()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_invert_one, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"  done origin {row['origin']} window {row['window_start']}: "
                      f"n={row['branching_ratio']:.4f} "
                      f"({'supercritical' if row['branching_ratio'] >= 1 else 'subcritical'}, "
                      f"{row['n_target_events']} events)", flush=True)
            except Exception as exc:  # pragma: no cover
                print(f"  FAILED origin {t['origin']} window {t['window']}: {exc}",
                      flush=True)

    frame = pd.DataFrame(rows).sort_values(["window_start", "origin"]).reset_index(drop=True)
    out = TABLE_DIR / "multi_origin_branching.csv"
    frame.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
