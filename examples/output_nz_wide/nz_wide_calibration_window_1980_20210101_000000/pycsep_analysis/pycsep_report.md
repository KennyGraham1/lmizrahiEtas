# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_window_1980_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.735`.
- Catalog N-Test is consistent in `1/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `0/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.963`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 357 | 324.573 | normal (1) | normal (1) | normal (0) | normal (0) | 0 |
| 730.000 | 495 | 649.095 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1095.000 | 637 | 973.542 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1461.000 | 767 | 1299.180 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1826.000 | 921 | 1623.249 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.001`
- Catalog M-Test: calibration p-value = `0.477`
- Catalog S-Test: calibration p-value = `0.000`
- Catalog PL-Test: calibration p-value = `0.001`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.100 | -32.427 | 0.000 | 0.000 |
| 730.000 | 0.763 | 154.095 | 0.000 | 0.000 |
| 1095.000 | 0.654 | 336.542 | 0.000 | 0.000 |
| 1461.000 | 0.590 | 532.180 | 0.000 | 0.000 |
| 1826.000 | 0.567 | 702.249 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 5.490 | 319.083 | 0.000 | 0.032 |
| 730.000 | 18.136 | 630.959 | 0.000 | 0.053 |
| 1095.000 | 37.158 | 936.384 | 0.000 | 0.073 |
| 1461.000 | 61.313 | 1237.867 | 0.000 | 0.093 |
| 1826.000 | 89.890 | 1533.360 | 0.000 | 0.113 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.987) | (167.350, -44.650, -0.587) |
| 730.000 | (179.850, -37.450, 35.974) | (167.350, -44.650, -1.246) |
| 1095.000 | (179.850, -37.450, 35.971) | (167.350, -44.650, -1.876) |
| 1461.000 | (179.850, -37.450, 35.968) | (167.350, -44.650, -2.422) |
| 1826.000 | (179.850, -37.450, 35.948) | (167.350, -44.650, -3.067) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
