# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_window_2000_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble underpredicts event counts on average; observed / simulated mean ratio is `1.114`.
- Catalog N-Test is consistent in `4/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.859`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.002`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 357 | 209.875 | normal (1) | normal (1) | undersampled (1) | undersampled (0) | 1 |
| 730.000 | 495 | 427.308 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1095.000 | 637 | 647.493 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1461.000 | 767 | 868.129 | normal (1) | normal (1) | normal (1) | normal (1) | 0 |
| 1826.000 | 921 | 1089.078 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 2 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.498`
- Catalog M-Test: calibration p-value = `0.411`
- Catalog S-Test: calibration p-value = `0.958`
- Catalog PL-Test: calibration p-value = `0.848`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.701 | -147.125 | 0.000 | 0.000 |
| 730.000 | 1.158 | -67.692 | 0.000 | 0.000 |
| 1095.000 | 0.984 | 10.493 | 0.000 | 0.000 |
| 1461.000 | 0.884 | 101.129 | 0.000 | 0.000 |
| 1826.000 | 0.846 | 168.078 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 15.248 | 194.628 | 1.000 | 0.026 |
| 730.000 | 48.272 | 379.035 | 1.000 | 0.039 |
| 1095.000 | 92.196 | 555.297 | 1.000 | 0.053 |
| 1461.000 | 153.125 | 715.005 | 0.000 | 0.064 |
| 1826.000 | 220.406 | 868.671 | 2.000 | 0.077 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.862) | (174.250, -41.750, -1.199) |
| 730.000 | (179.850, -37.450, 35.704) | (174.350, -41.650, -1.791) |
| 1095.000 | (179.850, -37.450, 35.562) | (174.350, -41.650, -2.635) |
| 1461.000 | (179.850, -37.450, 35.406) | (174.350, -41.650, -3.361) |
| 1826.000 | (179.850, -37.450, 35.255) | (174.350, -41.650, -4.125) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
