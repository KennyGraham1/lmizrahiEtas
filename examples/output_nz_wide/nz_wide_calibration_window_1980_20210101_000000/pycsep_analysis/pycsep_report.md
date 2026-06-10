# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_window_1980_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- Event-count performance is roughly balanced on average; observed / simulated mean ratio is `0.906`.
- Catalog N-Test is consistent in `2/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.865`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.002`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 357 | 253.716 | normal (1) | normal (1) | undersampled (1) | undersampled (0) | 1 |
| 730.000 | 495 | 524.611 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1095.000 | 637 | 800.448 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1461.000 | 767 | 1082.792 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 3 |
| 1826.000 | 921 | 1365.971 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.032`
- Catalog M-Test: calibration p-value = `0.591`
- Catalog S-Test: calibration p-value = `0.107`
- Catalog PL-Test: calibration p-value = `0.020`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.407 | -103.284 | 0.000 | 0.000 |
| 730.000 | 0.944 | 29.611 | 0.000 | 0.000 |
| 1095.000 | 0.796 | 163.448 | 0.000 | 0.000 |
| 1461.000 | 0.708 | 315.792 | 0.000 | 0.000 |
| 1826.000 | 0.674 | 444.971 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 17.183 | 236.534 | 1.000 | 0.027 |
| 730.000 | 55.598 | 469.013 | 1.000 | 0.043 |
| 1095.000 | 108.691 | 691.757 | 1.000 | 0.058 |
| 1461.000 | 182.035 | 900.757 | 3.000 | 0.072 |
| 1826.000 | 266.421 | 1099.550 | 1.000 | 0.086 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.860) | (174.250, -41.750, -1.124) |
| 730.000 | (179.850, -37.450, 35.704) | (174.350, -41.650, -1.809) |
| 1095.000 | (179.850, -37.450, 35.566) | (174.350, -41.650, -2.687) |
| 1461.000 | (179.850, -37.450, 35.342) | (174.350, -41.650, -3.533) |
| 1826.000 | (179.850, -37.450, 35.186) | (174.350, -41.650, -4.249) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
