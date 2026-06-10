# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_baseline_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.875`.
- Catalog N-Test is consistent in `2/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.870`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.001`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 357 | 261.652 | normal (1) | normal (1) | undersampled (1) | undersampled (0) | 1 |
| 730.000 | 495 | 543.452 | normal (1) | normal (1) | normal (1) | normal (1) | 0 |
| 1095.000 | 637 | 829.242 | normal (0) | normal (1) | normal (1) | normal (1) | 0 |
| 1461.000 | 767 | 1122.078 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1826.000 | 921 | 1421.172 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.017`
- Catalog M-Test: calibration p-value = `0.664`
- Catalog S-Test: calibration p-value = `0.041`
- Catalog PL-Test: calibration p-value = `0.005`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.364 | -95.348 | 0.000 | 0.000 |
| 730.000 | 0.911 | 48.452 | 0.000 | 0.000 |
| 1095.000 | 0.768 | 192.242 | 0.000 | 0.000 |
| 1461.000 | 0.684 | 355.078 | 0.000 | 0.000 |
| 1826.000 | 0.648 | 500.172 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 16.811 | 244.841 | 1.000 | 0.028 |
| 730.000 | 55.522 | 487.929 | 0.000 | 0.044 |
| 1095.000 | 109.317 | 719.925 | 0.000 | 0.060 |
| 1461.000 | 183.228 | 938.849 | 1.000 | 0.074 |
| 1826.000 | 270.377 | 1150.794 | 1.000 | 0.088 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.868) | (174.250, -41.750, -1.075) |
| 730.000 | (179.850, -37.450, 35.684) | (174.350, -41.650, -1.788) |
| 1095.000 | (179.850, -37.450, 35.562) | (174.350, -41.650, -2.575) |
| 1461.000 | (179.850, -37.450, 35.351) | (174.350, -41.650, -3.323) |
| 1826.000 | (179.850, -37.450, 35.213) | (174.350, -41.650, -4.136) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
