# pyCSEP NZ-Wide Analysis

- Run label: `nz_wide_calibration_mc_4p5_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble overpredicts event counts on average; observed / simulated mean ratio is `0.845`.
- Catalog N-Test is consistent in `2/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `0/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.980`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 124 | 105.748 | normal (1) | normal (1) | normal (0) | normal (0) | 0 |
| 730.000 | 187 | 210.784 | normal (1) | normal (1) | normal (0) | normal (1) | 0 |
| 1095.000 | 240 | 317.353 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1461.000 | 299 | 422.343 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |
| 1826.000 | 371 | 528.890 | normal (0) | normal (1) | normal (0) | normal (1) | 0 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.002`
- Catalog M-Test: calibration p-value = `0.814`
- Catalog S-Test: calibration p-value = `0.000`
- Catalog PL-Test: calibration p-value = `0.030`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.173 | -18.252 | 0.000 | 0.000 |
| 730.000 | 0.887 | 23.784 | 0.000 | 0.000 |
| 1095.000 | 0.756 | 77.353 | 0.000 | 0.000 |
| 1461.000 | 0.708 | 123.343 | 0.000 | 0.000 |
| 1826.000 | 0.701 | 157.890 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 0.912 | 104.837 | 0.000 | 0.011 |
| 730.000 | 2.991 | 207.793 | 0.000 | 0.019 |
| 1095.000 | 6.482 | 310.871 | 0.000 | 0.026 |
| 1461.000 | 10.956 | 411.388 | 0.000 | 0.033 |
| 1826.000 | 16.657 | 512.234 | 0.000 | 0.041 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 9.998) | (167.350, -44.650, -0.192) |
| 730.000 | (179.850, -37.450, 10.995) | (167.350, -44.650, -0.393) |
| 1095.000 | (179.850, -37.450, 10.988) | (167.350, -44.650, -0.599) |
| 1461.000 | (179.850, -37.450, 10.992) | (167.350, -44.650, -0.800) |
| 1826.000 | (179.850, -37.450, 10.982) | (167.350, -44.650, -0.976) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
