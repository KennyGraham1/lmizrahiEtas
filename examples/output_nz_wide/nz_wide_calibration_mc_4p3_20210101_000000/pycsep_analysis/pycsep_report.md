# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_mc_4p3_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.826`.
- Catalog N-Test is consistent in `1/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `0/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.971`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 208 | 174.727 | normal (1) | normal (1) | normal (0) | normal (0) | 0 |
| 730.000 | 301 | 349.529 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1095.000 | 391 | 523.770 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1461.000 | 473 | 699.457 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1826.000 | 575 | 874.597 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.001`
- Catalog M-Test: calibration p-value = `0.703`
- Catalog S-Test: calibration p-value = `0.000`
- Catalog PL-Test: calibration p-value = `0.005`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.190 | -33.273 | 0.000 | 0.000 |
| 730.000 | 0.861 | 48.529 | 0.000 | 0.000 |
| 1095.000 | 0.747 | 132.770 | 0.000 | 0.000 |
| 1461.000 | 0.676 | 226.457 | 0.000 | 0.000 |
| 1826.000 | 0.657 | 299.597 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 2.331 | 172.397 | 0.000 | 0.018 |
| 730.000 | 7.227 | 342.303 | 0.000 | 0.030 |
| 1095.000 | 15.491 | 508.279 | 0.000 | 0.042 |
| 1461.000 | 25.087 | 674.370 | 0.000 | 0.053 |
| 1826.000 | 37.864 | 836.733 | 0.000 | 0.065 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 18.994) | (167.350, -44.650, -0.350) |
| 730.000 | (179.850, -37.450, 19.990) | (167.350, -44.650, -0.654) |
| 1095.000 | (179.850, -37.450, 19.982) | (167.350, -44.650, -0.979) |
| 1461.000 | (179.850, -37.450, 19.978) | (167.350, -44.650, -1.349) |
| 1826.000 | (179.850, -37.450, 19.973) | (167.350, -44.650, -1.636) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
