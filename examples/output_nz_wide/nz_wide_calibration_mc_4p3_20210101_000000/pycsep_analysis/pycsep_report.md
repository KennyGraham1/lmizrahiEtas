# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_mc_4p3_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.859`.
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
| 365.000 | 208 | 162.331 | normal (1) | normal (1) | normal (1) | normal (1) | 0 |
| 730.000 | 301 | 334.667 | normal (1) | normal (1) | normal (1) | normal (1) | 0 |
| 1095.000 | 391 | 511.478 | normal (0) | normal (1) | normal (1) | normal (1) | 0 |
| 1461.000 | 473 | 687.430 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1826.000 | 575 | 868.665 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 2 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.027`
- Catalog M-Test: calibration p-value = `0.749`
- Catalog S-Test: calibration p-value = `0.252`
- Catalog PL-Test: calibration p-value = `0.020`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.281 | -45.669 | 0.000 | 0.000 |
| 730.000 | 0.899 | 33.667 | 0.000 | 0.000 |
| 1095.000 | 0.764 | 120.478 | 0.000 | 0.000 |
| 1461.000 | 0.688 | 214.430 | 0.000 | 0.000 |
| 1826.000 | 0.662 | 293.665 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 7.399 | 154.932 | 0.000 | 0.017 |
| 730.000 | 23.956 | 310.711 | 0.000 | 0.028 |
| 1095.000 | 47.793 | 463.685 | 0.000 | 0.039 |
| 1461.000 | 80.081 | 607.350 | 1.000 | 0.048 |
| 1826.000 | 118.301 | 750.365 | 2.000 | 0.058 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 18.931) | (174.250, -41.750, -0.668) |
| 730.000 | (179.850, -37.450, 19.855) | (174.350, -41.650, -1.067) |
| 1095.000 | (179.850, -37.450, 19.753) | (174.350, -41.650, -1.591) |
| 1461.000 | (179.850, -37.450, 19.676) | (174.350, -41.650, -2.076) |
| 1826.000 | (179.850, -37.450, 19.580) | (174.350, -41.650, -2.478) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
