# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_20180101_000000`
- Forecast start: `2018-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[30, 90, 365]` days
- Simulations per horizon: `250`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.278`.
- Catalog N-Test is consistent in `0/3` horizons.
- Catalog M-Test is consistent in `3/3` horizons.
- Catalog S-Test is consistent in `3/3` horizons.
- Catalog PL-Test is consistent in `3/3` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.968`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.194`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 30.000 | 7 | 25.708 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 2 |
| 90.000 | 21 | 80.832 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 6 |
| 365.000 | 102 | 337.988 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 1 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.000`
- Catalog M-Test: calibration p-value = `0.000`
- Catalog S-Test: calibration p-value = `0.327`
- Catalog PL-Test: calibration p-value = `0.000`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 30.000 | 0.272 | 18.708 | 0.000 | 0.000 |
| 90.000 | 0.260 | 59.832 | 0.000 | 0.000 |
| 365.000 | 0.302 | 235.988 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 30.000 | 0.216 | 25.492 | 2.000 | 0.002 |
| 90.000 | 1.120 | 79.712 | 6.000 | 0.005 |
| 365.000 | 24.936 | 313.052 | 1.000 | 0.019 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 30.000 | (173.950, -40.350, 1.000) | (174.250, -41.750, -0.256) |
| 90.000 | (167.250, -45.550, 1.000) | (174.250, -41.750, -0.760) |
| 365.000 | (165.550, -47.850, 1.952) | (174.250, -41.650, -2.244) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
