# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_mc_4p3_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.856`.
- Catalog N-Test is consistent in `2/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `5/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.907`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.001`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 208 | 162.954 | normal (1) | normal (1) | normal (1) | normal (1) | 0 |
| 730.000 | 301 | 334.861 | normal (1) | normal (1) | normal (1) | normal (1) | 0 |
| 1095.000 | 391 | 511.050 | normal (0) | normal (1) | normal (1) | normal (1) | 0 |
| 1461.000 | 473 | 690.611 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 2 |
| 1826.000 | 575 | 875.489 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.023`
- Catalog M-Test: calibration p-value = `0.726`
- Catalog S-Test: calibration p-value = `0.267`
- Catalog PL-Test: calibration p-value = `0.028`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.276 | -45.046 | 0.000 | 0.000 |
| 730.000 | 0.899 | 33.861 | 0.000 | 0.000 |
| 1095.000 | 0.765 | 120.050 | 0.000 | 0.000 |
| 1461.000 | 0.685 | 217.611 | 0.000 | 0.000 |
| 1826.000 | 0.657 | 300.489 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 7.277 | 155.678 | 0.000 | 0.017 |
| 730.000 | 23.774 | 311.087 | 0.000 | 0.028 |
| 1095.000 | 47.964 | 463.085 | 0.000 | 0.038 |
| 1461.000 | 80.951 | 609.660 | 2.000 | 0.048 |
| 1826.000 | 119.655 | 755.833 | 1.000 | 0.058 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 18.934) | (174.250, -41.750, -0.701) |
| 730.000 | (179.850, -37.450, 19.842) | (174.350, -41.650, -1.052) |
| 1095.000 | (179.850, -37.450, 19.761) | (174.350, -41.650, -1.595) |
| 1461.000 | (179.850, -37.450, 19.686) | (174.350, -41.650, -2.077) |
| 1826.000 | (179.850, -37.450, 19.567) | (174.350, -41.650, -2.566) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
