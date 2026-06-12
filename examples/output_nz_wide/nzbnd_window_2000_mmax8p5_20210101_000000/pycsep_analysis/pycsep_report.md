# pyCSEP NZ-Wide Analysis

- Run label: `nzbnd_window_2000_mmax8p5_20210101_000000`
- Forecast start: `2021-01-01 00:00:00`
- Region source: `forecast_domain`
- Forecast horizons: `[365, 730, 1095, 1461, 1826]` days
- Simulations per horizon: `2000`

## Key Findings

- The ensemble underpredicts event counts on average; observed / simulated mean ratio is `1.114`.
- Catalog N-Test is consistent in `3/5` horizons.
- Catalog M-Test is consistent in `5/5` horizons.
- Catalog S-Test is consistent in `5/5` horizons.
- Catalog PL-Test is consistent in `4/5` horizons.
- Spatial and pseudo-likelihood diagnostics indicate undersampled forecast cells for at least one horizon.
- A large share of the forecasted rate falls in cells with no observed events; the mean empty-cell share is `0.859`.
- Some observed activity lands in zero-rate cells; the mean unsupported observed-event share is `0.003`.

## Horizon Summary

| Horizon (days) | Observed | Sim mean | N | M | S | PL | Zero-rate obs cells |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 365.000 | 357 | 210.601 | normal (0) | normal (1) | undersampled (1) | undersampled (0) | 1 |
| 730.000 | 495 | 428.560 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 1 |
| 1095.000 | 637 | 646.905 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 2 |
| 1461.000 | 767 | 866.042 | normal (1) | normal (1) | undersampled (1) | undersampled (1) | 3 |
| 1826.000 | 921 | 1086.965 | normal (0) | normal (1) | undersampled (1) | undersampled (1) | 3 |

## Calibration Summary

- Catalog N-Test: calibration p-value = `0.507`
- Catalog M-Test: calibration p-value = `0.086`
- Catalog S-Test: calibration p-value = `0.829`
- Catalog PL-Test: calibration p-value = `0.788`

## Additional Diagnostics

| Horizon (days) | Obs/Sim mean ratio | Sim-Obs mean bias | Sim outside region (%) | Obs outside region (%) |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 1.695 | -146.399 | 0.000 | 0.000 |
| 730.000 | 1.155 | -66.440 | 0.000 | 0.000 |
| 1095.000 | 0.985 | 9.905 | 0.000 | 0.000 |
| 1461.000 | 0.886 | 99.042 | 0.000 | 0.000 |
| 1826.000 | 0.847 | 165.965 | 0.000 | 0.000 |

## Spatial Residual Diagnostics

| Horizon (days) | Expected in observed cells | Expected in empty cells | Observed in zero-rate cells | Mean abs(obs-exp) per cell |
| --- | ---: | ---: | ---: | ---: |
| 365.000 | 15.203 | 195.398 | 1.000 | 0.026 |
| 730.000 | 47.889 | 380.672 | 1.000 | 0.040 |
| 1095.000 | 92.635 | 554.269 | 2.000 | 0.053 |
| 1461.000 | 152.085 | 713.957 | 3.000 | 0.064 |
| 1826.000 | 220.118 | 866.846 | 3.000 | 0.076 |

## Peak Residual Cells

| Horizon (days) | Largest underforecast cell (lon, lat, obs-exp) | Largest overforecast cell (lon, lat, obs-exp) |
| --- | --- | --- |
| 365.000 | (179.850, -37.450, 34.867) | (174.250, -41.750, -1.224) |
| 730.000 | (179.850, -37.450, 35.703) | (174.350, -41.650, -1.749) |
| 1095.000 | (179.850, -37.450, 35.542) | (174.350, -41.650, -2.603) |
| 1461.000 | (179.850, -37.450, 35.422) | (174.350, -41.650, -3.442) |
| 1826.000 | (179.850, -37.450, 35.245) | (174.350, -41.650, -4.139) |

## Notes

- The pyCSEP catalog evaluations use the simulated catalog ensemble directly.
- `S` and `PL` tests are treated as one-sided lower-tail consistency checks.
- `zero-rate obs cells` counts how many occupied observed spatial bins had zero forecast mean rate.
- `expected in empty cells` is the forecast mean count allocated to cells without observed events in that horizon.
- `expected count in empty cells fraction` is tracked in the CSV/JSON outputs and the overview figure.
