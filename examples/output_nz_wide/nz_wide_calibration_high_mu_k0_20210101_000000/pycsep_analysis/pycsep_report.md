# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_high_mu_k0_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.876`.
- Catalog N-Test is consistent in `2/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `5/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.870`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.001`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 357 | 261.115 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 730.000 | 495 | 543.207 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1095.000 | 637 | 828.469 | normal (0) | normal (1) | normal (1) | normal (1) | 0 |
| 1461.000 | 767 | 1119.544 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1826.000 | 921 | 1420.862 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.027`
- Catalog M-Test: calibration p-value = `0.701`
- Catalog S-Test: calibration p-value = `0.054`
- Catalog PL-Test: calibration p-value = `0.007`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.367 | -95.885 | 0.000 | 0.000 |
| 730.000 | 0.911 | 48.207 | 0.000 | 0.000 |
| 1095.000 | 0.769 | 191.469 | 0.000 | 0.000 |
| 1461.000 | 0.685 | 352.544 | 0.000 | 0.000 |
| 1826.000 | 0.648 | 499.862 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 17.068 | 244.046 | 1.000 | 0.028 |
| 730.000 | 55.309 | 487.898 | 1.000 | 0.044 |
| 1095.000 | 109.031 | 719.438 | 0.000 | 0.060 |
| 1461.000 | 183.365 | 936.179 | 1.000 | 0.073 |
| 1826.000 | 268.726 | 1152.137 | 1.000 | 0.088 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.846) | (174.250, -41.750, -1.072) |
| 730.000 | (179.850, -37.450, 35.710) | (174.350, -41.650, -1.717) |
| 1095.000 | (179.850, -37.450, 35.543) | (174.350, -41.650, -2.567) |
| 1461.000 | (179.850, -37.450, 35.362) | (174.350, -41.650, -3.280) |
| 1826.000 | (179.850, -37.450, 35.243) | (174.350, -41.650, -4.197) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
