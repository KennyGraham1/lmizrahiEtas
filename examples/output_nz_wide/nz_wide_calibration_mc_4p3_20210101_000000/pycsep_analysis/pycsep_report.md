# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_mc_4p3_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.858`.
- Catalog N-Test is consistent in `2/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `5/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.907`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.002`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 208 | 162.470 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 730.000 | 301 | 334.428 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1095.000 | 391 | 511.525 | normal (0) | normal (1) | normal (1) | normal (1) | 0 |
| 1461.000 | 473 | 690.783 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1826.000 | 575 | 871.470 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.026`
- Catalog M-Test: calibration p-value = `0.722`
- Catalog S-Test: calibration p-value = `0.289`
- Catalog PL-Test: calibration p-value = `0.013`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.280 | -45.530 | 0.000 | 0.000 |
| 730.000 | 0.900 | 33.428 | 0.000 | 0.000 |
| 1095.000 | 0.764 | 120.525 | 0.000 | 0.000 |
| 1461.000 | 0.685 | 217.783 | 0.000 | 0.000 |
| 1826.000 | 0.660 | 296.470 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 7.422 | 155.048 | 1.000 | 0.017 |
| 730.000 | 23.853 | 310.575 | 1.000 | 0.028 |
| 1095.000 | 47.845 | 463.680 | 0.000 | 0.039 |
| 1461.000 | 80.013 | 610.770 | 1.000 | 0.048 |
| 1826.000 | 119.245 | 752.225 | 1.000 | 0.058 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 18.921) | (174.250, -41.750, -0.711) |
| 730.000 | (179.850, -37.450, 19.834) | (174.350, -41.650, -1.003) |
| 1095.000 | (179.850, -37.450, 19.765) | (174.350, -41.650, -1.544) |
| 1461.000 | (179.850, -37.450, 19.674) | (174.350, -41.650, -2.046) |
| 1826.000 | (179.850, -37.450, 19.617) | (174.350, -41.650, -2.561) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
