# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_mc_4p5_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.857`.
- Catalog N-Test is consistent in `2/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `5/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.935`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.005`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 124 | 101.028 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 730.000 | 187 | 207.288 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1095.000 | 240 | 316.223 | normal (0) | normal (1) | normal (1) | normal (1) | 0 |
| 1461.000 | 299 | 425.174 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 2 |
| 1826.000 | 371 | 534.843 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 2 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.036`
- Catalog M-Test: calibration p-value = `0.880`
- Catalog S-Test: calibration p-value = `0.530`
- Catalog PL-Test: calibration p-value = `0.041`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.227 | -22.972 | 0.000 | 0.000 |
| 730.000 | 0.902 | 20.288 | 0.000 | 0.000 |
| 1095.000 | 0.759 | 76.223 | 0.000 | 0.000 |
| 1461.000 | 0.703 | 126.174 | 0.000 | 0.000 |
| 1826.000 | 0.694 | 163.843 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 2.631 | 98.396 | 1.000 | 0.010 |
| 730.000 | 9.946 | 197.342 | 1.000 | 0.018 |
| 1095.000 | 20.561 | 295.662 | 0.000 | 0.025 |
| 1461.000 | 36.910 | 388.264 | 2.000 | 0.031 |
| 1826.000 | 53.501 | 481.342 | 2.000 | 0.038 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 9.953) | (174.250, -41.750, -0.380) |
| 730.000 | (179.850, -37.450, 10.910) | (174.250, -41.750, -0.812) |
| 1095.000 | (179.850, -37.450, 10.862) | (174.250, -41.750, -1.140) |
| 1461.000 | (179.850, -37.450, 10.822) | (174.350, -41.650, -1.241) |
| 1826.000 | (179.850, -37.450, 10.771) | (174.350, -41.650, -1.442) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
