# NZ-Wide ETAS Calibration Sweep

## Data & Configuration

| Parameter | Value |
| --- | --- |
| **Catalog** | GeoNet NZ FDSN (`nzcat.csv`) |
| **Mc** | 4.5 |
| **Training window** | 1960-01-01 → 2021-01-01 |
| **Auxiliary start** | 1950-01-01 |
| **Region** | 34°S–48°S, 164°E–180°E (rectangular) |
| **Background rate grid** | /home/kennyg/projects/ETASModels/lmizrahiEtas/input_data/hftlongtermmodel005.txt |
| **Background rate magnitude slice** | 5.0 |
| **Forecast origin** | `2021-01-01 00:00:00` |
| **Forecast horizons** | 365,730,1095,1461,1826 days |
| **Simulations / scenario** | 2000 |
| **Scenario count** | 7 |

## Ranking Rule

`calibration_score = mean_abs_log_count_ratio + consistency_penalty + spatial_penalty`

- **Consistency penalty** penalizes failed N/M/S/PL CSEP windows
- **Spatial penalty** penalizes empty-cell rate allocation, unsupported observed events, and mean absolute spatial residuals

## Best Scenario

| Metric | Value |
| --- | --- |
| Best scenario | **mc_4p5** |
| Score | 2.992 |
| Mean obs/sim ratio | 0.845 |
| Empty-cell share | 98.0% |
| Unsupported obs share | 0.0% |

## CSEP Consistency Tests

| Test | Fraction Passed | Status |
| --- | ---: | --- |
| N-test | 0.400 | ⚠️ Fail |
| M-test | 1.000 | ✅ Pass |
| S-test | 0.000 | ⚠️ Fail |
| PL-test | 0.800 | ✅ Pass |

> **Interpretation**: A test passes if ≥ 50% of forecast horizons produce
> consistency at the 95% significance level (quantile between 0.025 and 0.975).

## Scenario Comparison

| Scenario | Score | Obs/Sim ratio | Empty-cell share | Unsupported obs share | N frac | S frac | PL frac | log₁₀(μ) | log₁₀(k₀) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mc_4p5 | 2.992 | 0.845 | 0.980 | 0.000 | 0.400 | 0.000 | 0.800 | -inf | -20.000 |
| mc_4p3 | 3.475 | 0.826 | 0.971 | 0.000 | 0.200 | 0.000 | 0.800 | -inf | -20.000 |
| window_2000 | 3.697 | 0.873 | 0.963 | 0.000 | 0.200 | 0.000 | 0.800 | -inf | -20.000 |
| high_mu_k0 | 3.742 | 0.832 | 0.963 | 0.000 | 0.200 | 0.000 | 0.800 | -inf | -20.000 |
| low_mu_k0 | 3.743 | 0.832 | 0.963 | 0.000 | 0.200 | 0.000 | 0.800 | -inf | -20.000 |
| baseline | 3.743 | 0.832 | 0.963 | 0.000 | 0.200 | 0.000 | 0.800 | -inf | -20.000 |
| window_1980 | 3.867 | 0.735 | 0.963 | 0.000 | 0.200 | 0.000 | 0.800 | -inf | -20.000 |

## Horizon Ratios

| Scenario | Horizon (days) | Obs/Sim ratio | Empty-cell share | Unsupported obs share |
| --- | ---: | ---: | ---: | ---: |
| mc_4p5 | 365 | 1.173 | 0.991 | 0.000 |
| mc_4p5 | 730 | 0.887 | 0.986 | 0.000 |
| mc_4p5 | 1095 | 0.756 | 0.980 | 0.000 |
| mc_4p5 | 1461 | 0.708 | 0.974 | 0.000 |
| mc_4p5 | 1826 | 0.701 | 0.969 | 0.000 |
| mc_4p3 | 365 | 1.190 | 0.987 | 0.000 |
| mc_4p3 | 730 | 0.861 | 0.979 | 0.000 |
| mc_4p3 | 1095 | 0.747 | 0.970 | 0.000 |
| mc_4p3 | 1461 | 0.676 | 0.964 | 0.000 |
| mc_4p3 | 1826 | 0.657 | 0.957 | 0.000 |
| window_2000 | 365 | 1.307 | 0.983 | 0.000 |
| window_2000 | 730 | 0.906 | 0.972 | 0.000 |
| window_2000 | 1095 | 0.777 | 0.962 | 0.000 |
| window_2000 | 1461 | 0.700 | 0.953 | 0.000 |
| window_2000 | 1826 | 0.673 | 0.944 | 0.000 |
| high_mu_k0 | 365 | 1.245 | 0.983 | 0.000 |
| high_mu_k0 | 730 | 0.864 | 0.972 | 0.000 |
| high_mu_k0 | 1095 | 0.740 | 0.962 | 0.000 |
| high_mu_k0 | 1461 | 0.669 | 0.953 | 0.000 |
| high_mu_k0 | 1826 | 0.643 | 0.944 | 0.000 |
| low_mu_k0 | 365 | 1.245 | 0.983 | 0.000 |
| low_mu_k0 | 730 | 0.864 | 0.972 | 0.000 |
| low_mu_k0 | 1095 | 0.741 | 0.962 | 0.000 |
| low_mu_k0 | 1461 | 0.668 | 0.953 | 0.000 |
| low_mu_k0 | 1826 | 0.643 | 0.944 | 0.000 |
| baseline | 365 | 1.244 | 0.983 | 0.000 |
| baseline | 730 | 0.863 | 0.972 | 0.000 |
| baseline | 1095 | 0.741 | 0.962 | 0.000 |
| baseline | 1461 | 0.668 | 0.953 | 0.000 |
| baseline | 1826 | 0.642 | 0.944 | 0.000 |
| window_1980 | 365 | 1.100 | 0.983 | 0.000 |
| window_1980 | 730 | 0.763 | 0.972 | 0.000 |
| window_1980 | 1095 | 0.654 | 0.962 | 0.000 |
| window_1980 | 1461 | 0.590 | 0.953 | 0.000 |
| window_1980 | 1826 | 0.567 | 0.945 | 0.000 |

## Diagnostic Notes

- ⚠️ **High empty-cell share** (98.0%): The rectangular bounding box includes large oceanic areas with no real seismicity. A tighter NZ seismogenic-zone polygon would reduce this penalty and improve spatial diagnostics.
- ⚠️ **N-test failure**: The event-count forecast is inconsistent with observations across most horizons. This is the primary driver of the calibration score.
