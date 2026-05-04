# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_window_2000_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.873`.
- Catalog N-Test is consistent in `1/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `0/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.963`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 357 | 273.216 | normal (1) | normal (1) | normal (0) | normal (0) | 0 |
| 730.000 | 495 | 546.424 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1095.000 | 637 | 820.273 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1461.000 | 767 | 1095.129 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1826.000 | 921 | 1368.120 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.001`
- Catalog M-Test: calibration p-value = `0.699`
- Catalog S-Test: calibration p-value = `0.000`
- Catalog PL-Test: calibration p-value = `0.030`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.307 | -83.784 | 0.000 | 0.000 |
| 730.000 | 0.906 | 51.424 | 0.000 | 0.000 |
| 1095.000 | 0.777 | 183.273 | 0.000 | 0.000 |
| 1461.000 | 0.700 | 328.129 | 0.000 | 0.000 |
| 1826.000 | 0.673 | 447.120 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 4.629 | 268.587 | 0.000 | 0.030 |
| 730.000 | 15.210 | 531.214 | 0.000 | 0.048 |
| 1095.000 | 31.126 | 789.148 | 0.000 | 0.066 |
| 1461.000 | 51.697 | 1043.432 | 0.000 | 0.084 |
| 1826.000 | 76.018 | 1292.103 | 0.000 | 0.102 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.993) | (167.350, -44.650, -0.501) |
| 730.000 | (179.850, -37.450, 35.983) | (167.350, -44.650, -1.022) |
| 1095.000 | (179.850, -37.450, 35.974) | (167.350, -44.650, -1.548) |
| 1461.000 | (179.850, -37.450, 35.967) | (167.350, -44.650, -2.002) |
| 1826.000 | (179.850, -37.450, 35.952) | (167.350, -44.650, -2.611) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
