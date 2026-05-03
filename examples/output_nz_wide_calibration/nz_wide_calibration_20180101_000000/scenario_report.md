# NZ-Wide ETAS Calibration Sweep

## Data & Configuration

| Parameter | Value |
| --- | --- |
| **Catalog** | GeoNet NZ FDSN (`nzcat.csv`) |
| **Mc** | 4.1 |
| **Training window** | 1960-01-01 → 2018-01-01 |
| **Auxiliary start** | 1950-01-01 |
| **Region** | 34°S–48°S, 164°E–180°E (rectangular) |
| **Forecast origin** | `2018-01-01 00:00:00` |
| **Forecast horizons** | 30,90,365 days |
| **Simulations / scenario** | 250 |
| **Scenario count** | 1 |

## Ranking Rule

`calibration_score = mean_abs_log_count_ratio + consistency_penalty + spatial_penalty`

- **Consistency penalty** penalizes failed N/M/S/PL CSEP windows
- **Spatial penalty** penalizes empty-cell rate allocation, unsupported observed events, and mean absolute spatial residuals

## Best Scenario

| Metric | Value |
| --- | --- |
| Best scenario | **baseline** |
| Score | 3.917 |
| Mean obs/sim ratio | 0.286 |
| Empty-cell share | 96.8% |
| Unsupported obs share | 11.4% |

## CSEP Consistency Tests

| Test | Fraction Passed | Status |
| --- | ---: | --- |
| N-test | 0.000 | ⚠️ Fail |
| M-test | 1.000 | ✅ Pass |
| S-test | 1.000 | ✅ Pass |
| PL-test | 1.000 | ✅ Pass |

> **Interpretation**: A test passes if ≥ 50% of forecast horizons produce
> consistency at the 95% significance level (quantile between 0.025 and 0.975).

## Scenario Comparison

| Scenario | Score | Obs/Sim ratio | Empty-cell share | Unsupported obs share | N frac | S frac | PL frac | log₁₀(μ) | log₁₀(k₀) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.917 | 0.286 | 0.968 | 0.114 | 0.000 | 1.000 | 1.000 | -8.437 | -0.554 |

## Horizon Ratios

| Scenario | Horizon (days) | Obs/Sim ratio | Empty-cell share | Unsupported obs share |
| --- | ---: | ---: | ---: | ---: |
| baseline | 30 | 0.289 | 0.992 | 0.286 |
| baseline | 90 | 0.267 | 0.986 | 0.048 |
| baseline | 365 | 0.304 | 0.928 | 0.010 |

## Diagnostic Notes

- ⚠️ **Over-prediction**: Obs/Sim ratio = 0.286. The model forecasts ~3× more events than observed. Consider lowering the background rate (log₁₀μ) or narrowing the study region.
- ⚠️ **High empty-cell share** (96.8%): The rectangular bounding box includes large oceanic areas with no real seismicity. A tighter NZ seismogenic-zone polygon would reduce this penalty and improve spatial diagnostics.
- ⚠️ **N-test failure**: The event-count forecast is inconsistent with observations across most horizons. This is the primary driver of the calibration score.
