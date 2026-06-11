# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_window_2000_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble underpredicts event counts on average; observed / simulated mean ratio is `1.112`.
- Catalog N-Test is consistent in `4/5` horizons.
- Catalog M-Test is consistent in `4/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.859`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.002`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 357 | 211.507 | normal (1) | normal (1) | undersampled (1) | undersampled (0) | 1 |
| 730.000 | 495 | 427.447 | normal (1) | normal (0) | undersampled (1) | undersampled (1) | 1 |
| 1095.000 | 637 | 647.413 | normal (1) | normal (1) | normal (1) | normal (1) | 0 |
| 1461.000 | 767 | 865.494 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 2 |
| 1826.000 | 921 | 1088.986 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.475`
- Catalog M-Test: calibration p-value = `0.402`
- Catalog S-Test: calibration p-value = `0.605`
- Catalog PL-Test: calibration p-value = `0.814`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.688 | -145.493 | 0.000 | 0.000 |
| 730.000 | 1.158 | -67.553 | 0.000 | 0.000 |
| 1095.000 | 0.984 | 10.413 | 0.000 | 0.000 |
| 1461.000 | 0.886 | 98.494 | 0.000 | 0.000 |
| 1826.000 | 0.846 | 167.986 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 15.235 | 196.272 | 1.000 | 0.026 |
| 730.000 | 48.156 | 379.292 | 1.000 | 0.040 |
| 1095.000 | 93.126 | 554.287 | 0.000 | 0.053 |
| 1461.000 | 152.172 | 713.322 | 2.000 | 0.064 |
| 1826.000 | 220.441 | 868.545 | 1.000 | 0.077 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.862) | (174.250, -41.750, -1.216) |
| 730.000 | (179.850, -37.450, 35.712) | (174.350, -41.650, -1.770) |
| 1095.000 | (179.850, -37.450, 35.565) | (174.350, -41.650, -2.660) |
| 1461.000 | (179.850, -37.450, 35.449) | (174.350, -41.650, -3.357) |
| 1826.000 | (179.850, -37.450, 35.189) | (174.350, -41.650, -4.148) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
