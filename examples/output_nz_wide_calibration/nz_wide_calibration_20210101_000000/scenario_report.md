# NZ-Wide ETAS Calibration Sweep

## Data & Configuration

| Parameter | Value |
| --- | --- |
| **Catalog** | GeoNet NZ FDSN (`nzcat.csv`) |
| **Mc** | 4.1 |
| **Training window** | 2000-01-01 → 2021-01-01 |
| **Auxiliary start** | 1990-01-01 |
| **Region** | 34°S–48°S, 164°E–180°E (rectangular) |
| **Background rate grid** | disabled |
| **Background rate magnitude slice** | n/a |
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
| Best scenario | **window_2000** |
| Score | 2.022 |
| Mean obs/sim ratio | 1.112 |
| Empty-cell share | 85.9% |
| Unsupported obs share | 0.2% |

## CSEP Consistency Tests

| Test | Fraction Passed | Status |
| --- | ---: | --- |
| N-test | 0.800 | ✅ Pass |
| M-test | 0.800 | ✅ Pass |
| S-test | 1.000 | ✅ Pass |
| PL-test | 0.800 | ✅ Pass |

> **Interpretation**: A test passes if ≥ 50% of forecast horizons produce
> consistency at the 95% significance level (quantile between 0.025 and 0.975).

## Scenario Comparison

| Scenario | Score | Obs/Sim ratio | Empty-cell share | Unsupported obs share | N frac | S frac | PL frac | log₁₀(μ) | log₁₀(k₀) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_2000 | 2.022 | 1.112 | 0.859 | 0.002 | 0.800 | 1.000 | 0.800 | -8.484 | -1.185 |
| mc_4p5 | 2.345 | 0.857 | 0.935 | 0.005 | 0.400 | 1.000 | 1.000 | -8.845 | -0.802 |
| mc_4p3 | 2.471 | 0.858 | 0.907 | 0.002 | 0.400 | 1.000 | 1.000 | -8.667 | -0.711 |
| high_mu_k0 | 2.653 | 0.875 | 0.870 | 0.001 | 0.400 | 1.000 | 1.000 | -8.441 | -0.557 |
| low_mu_k0 | 2.654 | 0.873 | 0.870 | 0.001 | 0.400 | 1.000 | 1.000 | -8.441 | -0.557 |
| window_1980 | 2.716 | 0.905 | 0.866 | 0.001 | 0.400 | 1.000 | 0.800 | -8.406 | -0.750 |
| baseline | 2.754 | 0.875 | 0.870 | 0.000 | 0.400 | 1.000 | 0.800 | -8.441 | -0.557 |

## Horizon Ratios

| Scenario | Horizon (days) | Obs/Sim ratio | Empty-cell share | Unsupported obs share |
| --- | ---: | ---: | ---: | ---: |
| window_2000 | 365 | 1.688 | 0.928 | 0.003 |
| window_2000 | 730 | 1.158 | 0.887 | 0.002 |
| window_2000 | 1095 | 0.984 | 0.856 | 0.000 |
| window_2000 | 1461 | 0.886 | 0.824 | 0.003 |
| window_2000 | 1826 | 0.846 | 0.798 | 0.001 |
| mc_4p5 | 365 | 1.227 | 0.974 | 0.008 |
| mc_4p5 | 730 | 0.902 | 0.952 | 0.005 |
| mc_4p5 | 1095 | 0.759 | 0.935 | 0.000 |
| mc_4p5 | 1461 | 0.703 | 0.913 | 0.007 |
| mc_4p5 | 1826 | 0.694 | 0.900 | 0.005 |
| mc_4p3 | 365 | 1.280 | 0.954 | 0.005 |
| mc_4p3 | 730 | 0.900 | 0.929 | 0.003 |
| mc_4p3 | 1095 | 0.764 | 0.906 | 0.000 |
| mc_4p3 | 1461 | 0.685 | 0.884 | 0.002 |
| mc_4p3 | 1826 | 0.660 | 0.863 | 0.002 |
| high_mu_k0 | 365 | 1.359 | 0.935 | 0.003 |
| high_mu_k0 | 730 | 0.915 | 0.898 | 0.000 |
| high_mu_k0 | 1095 | 0.769 | 0.869 | 0.000 |
| high_mu_k0 | 1461 | 0.682 | 0.837 | 0.001 |
| high_mu_k0 | 1826 | 0.649 | 0.810 | 0.001 |
| low_mu_k0 | 365 | 1.355 | 0.935 | 0.003 |
| low_mu_k0 | 730 | 0.911 | 0.897 | 0.000 |
| low_mu_k0 | 1095 | 0.768 | 0.869 | 0.000 |
| low_mu_k0 | 1461 | 0.684 | 0.837 | 0.001 |
| low_mu_k0 | 1826 | 0.647 | 0.811 | 0.001 |
| window_1980 | 365 | 1.407 | 0.933 | 0.003 |
| window_1980 | 730 | 0.942 | 0.894 | 0.000 |
| window_1980 | 1095 | 0.795 | 0.865 | 0.000 |
| window_1980 | 1461 | 0.708 | 0.832 | 0.001 |
| window_1980 | 1826 | 0.673 | 0.805 | 0.001 |
| baseline | 365 | 1.363 | 0.936 | 0.000 |
| baseline | 730 | 0.912 | 0.898 | 0.000 |
| baseline | 1095 | 0.767 | 0.868 | 0.000 |
| baseline | 1461 | 0.682 | 0.838 | 0.001 |
| baseline | 1826 | 0.649 | 0.810 | 0.001 |

## Diagnostic Notes

