# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_mc_4p5_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.860`.
- Catalog N-Test is consistent in `2/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `5/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.935`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.007`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 124 | 100.652 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 730.000 | 187 | 207.417 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1095.000 | 240 | 314.812 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 2 |
| 1461.000 | 299 | 423.682 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 2 |
| 1826.000 | 371 | 533.047 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 3 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.037`
- Catalog M-Test: calibration p-value = `0.843`
- Catalog S-Test: calibration p-value = `0.540`
- Catalog PL-Test: calibration p-value = `0.042`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.232 | -23.348 | 0.000 | 0.000 |
| 730.000 | 0.902 | 20.417 | 0.000 | 0.000 |
| 1095.000 | 0.762 | 74.812 | 0.000 | 0.000 |
| 1461.000 | 0.706 | 124.682 | 0.000 | 0.000 |
| 1826.000 | 0.696 | 162.047 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 2.597 | 98.055 | 1.000 | 0.010 |
| 730.000 | 9.820 | 197.597 | 1.000 | 0.018 |
| 1095.000 | 20.249 | 294.564 | 2.000 | 0.024 |
| 1461.000 | 36.575 | 387.106 | 2.000 | 0.031 |
| 1826.000 | 54.112 | 478.935 | 3.000 | 0.038 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 9.966) | (174.250, -41.750, -0.424) |
| 730.000 | (179.850, -37.450, 10.919) | (174.250, -41.750, -0.840) |
| 1095.000 | (179.850, -37.450, 10.855) | (174.250, -41.750, -1.188) |
| 1461.000 | (179.850, -37.450, 10.796) | (174.350, -41.650, -1.152) |
| 1826.000 | (179.850, -37.450, 10.764) | (174.350, -41.650, -1.503) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
