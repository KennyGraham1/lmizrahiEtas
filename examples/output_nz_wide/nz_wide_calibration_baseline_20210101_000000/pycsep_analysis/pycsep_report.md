# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_baseline_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.832`.
- Catalog N-Test is consistent in `1/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `0/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.963`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 357 | 286.956 | normal (1) | normal (1) | normal (0) | normal (0) | 0 |
| 730.000 | 495 | 573.328 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1095.000 | 637 | 860.172 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1461.000 | 767 | 1147.822 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1826.000 | 921 | 1434.165 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.001`
- Catalog M-Test: calibration p-value = `0.682`
- Catalog S-Test: calibration p-value = `0.000`
- Catalog PL-Test: calibration p-value = `0.009`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.244 | -70.044 | 0.000 | 0.000 |
| 730.000 | 0.863 | 78.328 | 0.000 | 0.000 |
| 1095.000 | 0.741 | 223.172 | 0.000 | 0.000 |
| 1461.000 | 0.668 | 380.822 | 0.000 | 0.000 |
| 1826.000 | 0.642 | 513.165 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 4.860 | 282.096 | 0.000 | 0.030 |
| 730.000 | 15.992 | 557.337 | 0.000 | 0.049 |
| 1095.000 | 32.805 | 827.368 | 0.000 | 0.068 |
| 1461.000 | 54.082 | 1093.739 | 0.000 | 0.086 |
| 1826.000 | 79.610 | 1354.555 | 0.000 | 0.105 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.989) | (167.350, -44.650, -0.559) |
| 730.000 | (179.850, -37.450, 35.977) | (167.350, -44.650, -1.119) |
| 1095.000 | (179.850, -37.450, 35.976) | (167.350, -44.650, -1.623) |
| 1461.000 | (179.850, -37.450, 35.965) | (167.350, -44.650, -2.152) |
| 1826.000 | (179.850, -37.450, 35.954) | (167.350, -44.650, -2.781) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
