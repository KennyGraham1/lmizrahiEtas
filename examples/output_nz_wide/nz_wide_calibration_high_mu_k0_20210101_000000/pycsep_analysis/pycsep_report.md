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
| 365.000 | 357 | 262.634 | normal (1) | normal (1) | normal (1) | normal (1) | 0 |
| 730.000 | 495 | 540.328 | normal (1) | normal (1) | normal (1) | normal (1) | 0 |
| 1095.000 | 637 | 827.729 | normal (0) | normal (1) | normal (1) | normal (1) | 0 |
| 1461.000 | 767 | 1119.633 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 2 |
| 1826.000 | 921 | 1417.878 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.028`
- Catalog M-Test: calibration p-value = `0.636`
- Catalog S-Test: calibration p-value = `0.053`
- Catalog PL-Test: calibration p-value = `0.009`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.359 | -94.366 | 0.000 | 0.000 |
| 730.000 | 0.916 | 45.328 | 0.000 | 0.000 |
| 1095.000 | 0.770 | 190.729 | 0.000 | 0.000 |
| 1461.000 | 0.685 | 352.633 | 0.000 | 0.000 |
| 1826.000 | 0.650 | 496.878 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 16.928 | 245.706 | 0.000 | 0.028 |
| 730.000 | 55.235 | 485.093 | 0.000 | 0.044 |
| 1095.000 | 109.083 | 718.646 | 0.000 | 0.060 |
| 1461.000 | 181.846 | 937.788 | 2.000 | 0.074 |
| 1826.000 | 268.383 | 1149.494 | 1.000 | 0.088 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.850) | (174.250, -41.750, -1.135) |
| 730.000 | (179.850, -37.450, 35.684) | (174.350, -41.650, -1.803) |
| 1095.000 | (179.850, -37.450, 35.509) | (174.350, -41.650, -2.560) |
| 1461.000 | (179.850, -37.450, 35.418) | (174.350, -41.650, -3.370) |
| 1826.000 | (179.850, -37.450, 35.245) | (174.350, -41.650, -4.209) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
