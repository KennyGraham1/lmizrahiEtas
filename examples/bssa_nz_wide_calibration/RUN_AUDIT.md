# Scientific Audit of the June 11, 2026 Sweep

## Run Identification

The analyzed artifacts correspond to:

```text
python examples/run_nz_wide_calibration_sweep.py \
  --scenario-set default \
  --background-rate-file "" \
  --allow-degenerate-inversion \
  --force-rerun
```

The manifest reports seven completed scenarios, no subprocess failures, 2,000
catalogs per scenario and horizon, and five horizons from 365 to 1,826 days.
The forecast origin was 1 January 2021. The external background-rate grid was
disabled.

## Findings That Support the Paper

1. **Only one branching-ratio point estimate is below unity.** The 2000-window
   estimate is 0.9691. The other six estimates range from 1.0198 to 1.0329 and
   were simulated only because the command overrode the guard. No parameter- or
   catalog-uncertainty interval was calculated.

2. **The initialization test is informative.** The low, baseline, and high
   initial-rate scenarios converge to nearly identical final parameters and
   branching ratios. The supercritical result is not explained by the tested
   starting values.

3. **Threshold changes do not solve the long-window problem.** Raising
   `Mc` from 4.1 to 4.3 or 4.5 reduces the training sample but leaves the
   branching ratio near 1.033.

4. **The 2000-window model changes bias sign with horizon.** Its observed to
   mean-simulated count ratio is 1.688, 1.158, 0.984, 0.886, and 0.846 from
   one through five years. A single average hides first-year underprediction
   and longer-term overprediction.

5. **The main spatial miss is interpretable.** The largest positive residual
   is repeatedly near 179.85 degrees E, 37.45 degrees S, corresponding to the
   March 2021 East Cape sequence (the M8.1 Kermadec Islands earthquake of 5
   March 2021 occurred north of the study domain and is a different event).

6. **The magnitude tail is not operationally credible.** With no `m_max`,
   4.25% of five-year 2000-window catalogs contain an `M >= 8` event and the
   ensemble maximum is 11.2.

## Methodological Qualifications

1. The five horizons are nested and are not independent forecast experiments.
   Cross-horizon pass fractions and pyCSEP calibration-test p-values must be
   interpreted descriptively.

2. The composite calibration score is a project-specific, post-hoc weighted
   sum. Its weights have no sampling-theory basis. It should not be presented
   as a standard CSEP score or formal model-selection criterion.

3. The score's empty-cell penalty is strongly grid dependent. On a 0.1-degree
   grid with about 21,000 cells and fewer than 1,000 observations, a high
   expected-rate fraction in unoccupied cells is not by itself evidence of
   poor spatial calibration.

4. pyCSEP reports undersampled spatial or pseudolikelihood evaluations at
   several horizons. When an observation falls in a zero-rate cell, the
   implementation reports the issue and recomputes after removing unsupported
   events. Pass indicators therefore need the accompanying zero-rate-event
   diagnostic.

5. The original wrapper used an incorrect two-sided quantile rule and labeled
   the 2000-window first-year N-test as consistent even though all 2,000
   simulated counts were below the observation. The corrected N-test is
   consistent at three of five horizons. The manuscript uses pyCSEP's
   resampled magnitude test, which is consistent at all five horizons.

6. Scenarios with different `Mc` evaluate different target catalogs. Their
   within-scenario calibration is valid, but their composite scores are not
   comparisons of exactly the same predictand.

7. There is no reference model. The run tests consistency and sensitivity, not
   comparative skill.

8. Random seeds are reset from system entropy in simulation workers. The
   archived realizations are analyzable, but rerunning the command will not
   reproduce identical catalogs.

9. The metadata records longitude bounds starting at 164 degrees E, whereas
   the polygon used by inversion, simulation, and pyCSEP starts at 165 degrees
   E. The paper reports the effective 165--180 degree E domain.

10. The GeoNet input catalog extends to 30 April 2026 and can be revised by the
   provider. The inversion ends at the 2021 forecast origin, and observed
   windows end on 1 January 2026, so no direct temporal leakage was found.
   Reproducibility still requires archiving the exact catalog snapshot.

11. The generated scenario report labels its data configuration using the
    highest-ranked scenario (`Mc=4.1`, 2000 start), even though the sweep
    contains multiple thresholds and windows. The manuscript uses a scenario
    table instead.

## Publication Boundary

The current evidence supports a retrospective diagnostic report: the tested
catalog windows are associated with different branching-ratio point estimates
and finite-horizon calibration. It does not identify a causal training-window
effect or demonstrate prospective New Zealand forecasting skill. A
submission-ready study requires branching-ratio uncertainty, multiple
independent origins, a frozen and magnitude-homogenized catalog, a buffered
dateline-aware source domain, reference models, deterministic seeds, and a
physically constrained magnitude distribution.
