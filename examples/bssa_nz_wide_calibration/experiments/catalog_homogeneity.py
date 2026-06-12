#!/usr/bin/env python3
"""Catalog magnitude provenance, completeness, and dateline-aware inputs.

The GeoNet FDSN catalog contains several reported magnitude types. This script
does not pretend that they are interchangeable with Mw. It preserves the
reported value and applies only the published historical correction of Zuniga
et al. (2005) to deep (depth > 40 km) events:

* before 1968: M_h = 1.23 M - 0.64
* 1968 through 1986: M_h = M + 0.2
* otherwise: unchanged

The first expression combines their 1940--1968 to 1968--1987 conversion with
their recommended +0.2 shift onto the post-1987 scale. Completeness is then
estimated separately by era and broad geographic sector with the maximum-
curvature and Gutenberg--Richter KS methods.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BSSA = HERE.parent
ROOT = BSSA.parents[1]
sys.path.insert(0, str(ROOT / "SeismoStats"))

from seismostats.analysis import estimate_mc_ks, estimate_mc_maxc
from seismostats.utils import bin_to_precision

INPUT = ROOT / "input_data" / "nzcat_buffered_typed.csv"
OUTPUT = ROOT / "input_data" / "nzcat_buffered_homogeneous.csv"
OUTER_POLYGON = ROOT / "input_data" / "nz_polygon_buffered_dateline.npy"
INNER_POLYGON = ROOT / "input_data" / "nz_polygon_target_unwrapped.npy"
TABLE_DIR = BSSA / "tables"

ERAS = [
    ("1950--1967", "1950-01-01", "1968-01-01"),
    ("1968--1986", "1968-01-01", "1987-01-01"),
    ("1987--1999", "1987-01-01", "2000-01-01"),
    ("2000--2010", "2000-01-01", "2011-01-01"),
    ("2011--2020", "2011-01-01", "2021-01-01"),
]


def sector(frame: pd.DataFrame) -> pd.Series:
    """Return reproducible broad sectors, not tectonic-zone classifications."""
    result = pd.Series("buffer", index=frame.index, dtype=object)
    target = frame["longitude"].between(165.0, 180.0) & frame["latitude"].between(
        -48.0, -34.0
    )
    result[target & (frame["latitude"] >= -38.5)] = "target_north"
    result[
        target
        & (frame["latitude"] < -38.5)
        & (frame["latitude"] >= -42.5)
    ] = "target_central"
    result[target & (frame["latitude"] < -42.5)] = "target_south"
    return result


def homogenize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["reported_magnitude"] = out["magnitude"].astype(float)
    out["magnitude_correction"] = 0.0
    deep = out["depth"].fillna(0) > 40.0
    pre_1968 = out["time"] < pd.Timestamp("1968-01-01")
    era_1968 = out["time"].between(
        pd.Timestamp("1968-01-01"), pd.Timestamp("1987-01-01"), inclusive="left"
    )
    out.loc[deep & pre_1968, "magnitude"] = (
        1.23 * out.loc[deep & pre_1968, "reported_magnitude"] - 0.64
    )
    out.loc[deep & era_1968, "magnitude"] = (
        out.loc[deep & era_1968, "reported_magnitude"] + 0.2
    )
    out["magnitude"] = np.round(out["magnitude"] / 0.1) * 0.1
    out["magnitude_correction"] = out["magnitude"] - out["reported_magnitude"]
    out["historical_correction_applied"] = np.abs(out["magnitude_correction"]) > 1e-9
    out["sector"] = sector(out)
    return out


def estimate_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    np.random.seed(20260611)
    groups = [("all_target", frame[frame["sector"].str.startswith("target_")])]
    groups.extend((name, frame[frame["sector"] == name]) for name in sorted(
        s for s in frame["sector"].unique() if s.startswith("target_")
    ))
    groups.append(("buffer_only", frame[frame["sector"] == "buffer"]))

    for era, start, end in ERAS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        for region, group in groups:
            sample = group[group["time"].between(start_ts, end_ts, inclusive="left")]
            mags = bin_to_precision(sample["magnitude"].to_numpy(dtype=float), 0.1)
            if len(mags) < 100:
                rows.append({
                    "era": era, "region": region, "n_events_mge3p5": len(mags),
                    "mc_maxc": np.nan, "mc_ks": np.nan, "ks_p_value": np.nan,
                    "b_value_at_mc_ks": np.nan,
                })
                continue
            mc_maxc, _ = estimate_mc_maxc(mags, fmd_bin=0.1, correction_factor=0.2)
            candidates = bin_to_precision(
                np.arange(3.5, min(5.6, np.max(mags)), 0.1), 0.1
            )
            mc_ks, info = estimate_mc_ks(
                mags,
                delta_m=0.1,
                mcs_test=candidates,
                p_value_pass=0.1,
                stop_when_passed=True,
                n=1000,
            )
            if mc_ks is None:
                ks_p, b_value = np.nan, np.nan
            else:
                index = info["mcs_tested"].index(mc_ks)
                ks_p = info["p_values"][index]
                b_value = info["b_values_tested"][index]
            rows.append({
                "era": era,
                "region": region,
                "n_events_mge3p5": len(mags),
                "mc_maxc": mc_maxc,
                "mc_ks": mc_ks,
                "ks_p_value": ks_p,
                "b_value_at_mc_ks": b_value,
            })
    return pd.DataFrame(rows)


def assign_variable_completeness(
    catalog: pd.DataFrame,
    completeness: pd.DataFrame,
) -> pd.DataFrame:
    """Assign conservative era-sector Mc values for ETAS variable-Mc mode."""
    estimates = completeness.copy()
    estimates["mc_selected"] = estimates[["mc_maxc", "mc_ks"]].max(axis=1)
    lookup = estimates.set_index(["era", "region"])["mc_selected"].to_dict()
    all_target = {
        era: lookup[(era, "all_target")]
        for era, _, _ in ERAS
    }
    era_edges = [pd.Timestamp(item[1]) for item in ERAS] + [pd.Timestamp(ERAS[-1][2])]
    era_names = [item[0] for item in ERAS]

    out = catalog.copy()
    out["completeness_era"] = pd.cut(
        out["time"], bins=era_edges, labels=era_names, right=False
    ).astype(object)
    out.loc[out["time"] >= era_edges[-1], "completeness_era"] = era_names[-1]
    out["mc_current"] = np.nan
    for index, event in out.iterrows():
        era = event["completeness_era"]
        region = event["sector"] if event["sector"] != "buffer" else "buffer_only"
        value = lookup.get((era, region), np.nan)
        if pd.isna(value):
            value = all_target[era]
        out.at[index, "mc_current"] = value
    return out


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(INPUT, index_col=0, parse_dates=["time"])
    catalog = homogenize(catalog).sort_values("time")

    np.save(OUTER_POLYGON, np.array([
        [-29.0, 160.0], [-29.0, 190.0], [-53.0, 190.0],
        [-53.0, 160.0], [-29.0, 160.0],
    ]))
    np.save(INNER_POLYGON, np.array([
        [-34.0, 165.0], [-34.0, 180.0], [-48.0, 180.0],
        [-48.0, 165.0], [-34.0, 165.0],
    ]))

    type_counts = (
        catalog.assign(era=pd.cut(
            catalog["time"],
            bins=pd.to_datetime([e[1] for e in ERAS] + [ERAS[-1][2]]),
            labels=[e[0] for e in ERAS],
            right=False,
        ))
        .groupby(["era", "sector", "mag_type"], observed=True)
        .size()
        .rename("n_events")
        .reset_index()
    )
    type_counts.to_csv(TABLE_DIR / "catalog_magnitude_types.csv", index=False)

    completeness = estimate_completeness(catalog)
    completeness.to_csv(TABLE_DIR / "catalog_completeness.csv", index=False)
    catalog = assign_variable_completeness(catalog, completeness)
    catalog.to_csv(OUTPUT)

    summary = pd.DataFrame([{
        "catalog_events": len(catalog),
        "events_with_historical_correction": int(
            catalog["historical_correction_applied"].sum()
        ),
        "events_east_of_dateline": int((catalog["longitude"] > 180.0).sum()),
        "target_events": int(catalog["sector"].str.startswith("target_").sum()),
        "buffer_events": int((catalog["sector"] == "buffer").sum()),
        "catalog_start": catalog["time"].min(),
        "catalog_end": catalog["time"].max(),
    }])
    summary.to_csv(TABLE_DIR / "catalog_homogeneity_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nCompleteness estimates:")
    print(completeness.to_string(index=False))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
