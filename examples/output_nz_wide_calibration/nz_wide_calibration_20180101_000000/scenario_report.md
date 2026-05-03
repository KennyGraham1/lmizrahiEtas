# NZ-Wide Calibration Sweep

- Forecast start: `2018-01-01 00:00:00`
- Forecast horizons: `30,90,365` days
- Simulations per scenario: `250`
- Scenario count: `1`

## Ranking Rule

- `calibration_score = mean_abs_log_count_ratio + consistency_penalty + spatial_penalty`, where consistency penalizes failed `N/M/S/PL` windows and spatial penalizes empty-cell rate allocation, unsupported observed events, and mean absolute spatial residuals.

## Best Scenario

- Best scenario: `baseline`
- Score: `3.917`
- Mean observed/simulated ratio: `0.286`
- Mean empty-cell forecast share: `0.968`
- Mean unsupported observed-event share: `0.114`

## Scenario Comparison

| Scenario | Score | Obs/Sim ratio | Empty-cell share | Unsupported obs share | N frac | S frac | PL frac | final log10_mu | final log10_k0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.917 | 0.286 | 0.968 | 0.114 | 0.000 | 1.000 | 1.000 | -8.437 | -0.554 |

## Horizon Ratios

| Scenario | Horizon (days) | Obs/Sim ratio | Empty-cell share | Unsupported obs share |
| --- | ---: | ---: | ---: | ---: |
| baseline | 30 | 0.289 | 0.992 | 0.286 |
| baseline | 90 | 0.267 | 0.986 | 0.048 |
| baseline | 365 | 0.304 | 0.928 | 0.010 |
