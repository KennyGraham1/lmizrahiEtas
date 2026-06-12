#!/usr/bin/env python3
"""Recommendation 3: physically constrained (bounded) magnitude distribution.

The published simulations used an unbounded exponential Gutenberg--Richter law,
which produced an M=11.2 ensemble maximum. The branching ratio also depends on
the magnitude integral, so an unbounded law inflates n as well as the tail. This
script has two parts.

Part A (analytic, instant): recompute the branching ratio for every scenario
under a finite maximum magnitude m_max, using the closed-form
``branching_ratio(theta, beta, dm_max)``. This shows how much of each scenario's
supercriticality is an artifact of the unbounded law.

Part B (simulation): re-simulate the admissible 2000-window forecast with a
truncated GR law (m_max=8.5 by default) and the same pyCSEP scoring, so the
bounded tail and its effect on count calibration can be compared with the
unbounded run. The cached inversion is reused (m_max does not affect the fit), so
only the simulation and evaluation are recomputed.

Outputs ``tables/bounded_branching.csv`` (Part A) and a full forecast run under
``output_nz_wide/nzbnd_window_2000_mmax<MM>_*`` (Part B).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import datetime as dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
EXAMPLES = BSSA.parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT / "SeismoStats"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

TABLE_DIR = BSSA / "tables"
OUTPUT_ROOT = EXAMPLES / "output_nz_wide"

SCENARIO_ORDER = [
    "baseline", "low_mu_k0", "high_mu_k0", "mc_4p3", "mc_4p5",
    "window_1980", "window_2000",
]


def published_label(scenario: str) -> str:
    return f"nz_wide_calibration_{scenario}_20210101_000000"


def part_a_analytic(m_max_values: list[float]) -> None:
    import numpy as np
    import pandas as pd
    from etas.inversion import branching_ratio, parameter_dict2array

    rows = []
    for scenario in SCENARIO_ORDER:
        label = published_label(scenario)
        path = OUTPUT_ROOT / label / f"parameters_{label}.json"
        with open(path) as handle:
            payload = json.load(handle)
        theta = parameter_dict2array(payload["final_parameters"])
        beta = float(payload["beta"])
        m_ref = float(payload["m_ref"])
        row = {
            "scenario": scenario,
            "beta": beta,
            "b_value": beta / np.log(10),
            "m_ref": m_ref,
            "n_unbounded": float(branching_ratio(theta, beta, dm_max=None)),
        }
        for m_max in m_max_values:
            dm_max = m_max - m_ref
            row[f"n_mmax_{m_max:g}"] = float(branching_ratio(theta, beta, dm_max=dm_max))
        rows.append(row)
    frame = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "bounded_branching.csv"
    frame.to_csv(out, index=False)
    print(f"Part A: wrote {out}")
    print(frame.to_string(index=False))
    return frame


def part_b_simulate(m_max: float, n_sims: int, sim_workers: int) -> None:
    """Re-simulate the 2000-window forecast with a truncated GR law."""
    src_label = published_label("window_2000")
    forecast_start = dt.datetime(2021, 1, 1)
    mm_tag = f"{m_max:g}".replace(".", "p")
    experiment_name = f"nzbnd_window_2000_mmax{mm_tag}"

    # Pre-seed the cached inversion so only the simulation/evaluation re-runs.
    import run_nz_wide_forecast as fc
    paths = fc.build_run_paths(experiment_name, forecast_start)
    new_label = paths["run_label"]
    src_params = OUTPUT_ROOT / src_label / f"parameters_{src_label}.json"
    dst_params = Path(paths["output_dir"]) / f"parameters_{new_label}.json"
    # The parameters JSON references sources_/trig_and_bg_probs_ by absolute path
    # in the source run directory, which load_calculation reads directly, so only
    # the parameter file itself needs copying.
    shutil.copy(src_params, dst_params)
    print(f"Part B: seeded cached inversion {dst_params.name} from {src_label}")

    cmd = [
        sys.executable, str(EXAMPLES / "run_nz_wide_forecast.py"),
        "--forecast-start", "2021-01-01 00:00:00",
        "--durations", "365,730,1095,1461,1826",
        "--n-simulations", str(n_sims),
        "--experiment-name", experiment_name,
        "--timewindow-start", "2000-01-01 00:00:00",
        "--auxiliary-start", "1990-01-01 00:00:00",
        "--mc", "4.1",
        "--m-max", str(m_max),
        "--simulation-workers", str(sim_workers),
        "--pycsep-region-source", "forecast_domain",
        "--pycsep-grid-spacing", "0.1",
        "--pycsep-mag-bin", "0.1",
    ]
    print("Part B: running bounded-magnitude forecast:\n  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(EXAMPLES))
    print(f"Part B: completed bounded forecast {new_label}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-max", type=float, default=8.5)
    parser.add_argument("--m-max-grid", nargs="*", type=float,
                        default=[7.5, 8.0, 8.5, 9.0])
    parser.add_argument("--n-simulations", type=int, default=2000)
    parser.add_argument("--simulation-workers", type=int, default=12)
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Only recompute the analytic Part A table.")
    args = parser.parse_args()

    part_a_analytic(args.m_max_grid)
    if not args.skip_simulation:
        part_b_simulate(args.m_max, args.n_simulations, args.simulation_workers)


if __name__ == "__main__":
    main()
