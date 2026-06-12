# Bibliography and Citation Audit

Audit date: 11 June 2026

## Method

Each DOI in the original `references.bib` was resolved through Crossref.
Titles, authors, journal, year, volume, issue, pages or article number, and DOI
were compared with the registered record. OpenAlex was used as a second
registry where Crossref omitted page ranges. Citation contexts in
`manuscript.tex` were then checked against article titles and registered
abstracts.

## Critical Findings

### Fabricated or DOI-Mismatched Entries

1. **Former `Gerstenberger2018`: invalid as written.**

   The title *Recent Developments in Operational Aftershock Forecasting in New
   Zealand* was not found in Crossref under that author combination or as an
   exact title. DOI
   [10.1785/0220180092](https://doi.org/10.1785/0220180092) resolves to Niu and
   Yamaoka, *Preface to the Focus Section on Nonexplosive Source Monitoring and
   Imaging*, SRL 89(3), 972--973. It is unrelated to New Zealand aftershock
   forecasting.

   **Action:** replaced with Gerstenberger, Christophersen, and Rhoades (2024),
   *A Review of 15 Years of Public Earthquake Forecasting in Aotearoa New
   Zealand*, [10.1785/0220240207](https://doi.org/10.1785/0220240207), SRL
   95(6), 3416--3432. Its abstract directly documents public forecasts issued
   in New Zealand and their operational models.

2. **Former `Marzocchi2020`: invalid as written.**

   The title *The Dramatic Bias-Variance Compensation Undermines Some ETAS
   Models* was not found as a Crossref record or among indexed works of
   Marzocchi and Lombardi. DOI
   [10.1029/2020JB019396](https://doi.org/10.1029/2020JB019396) resolves to Yu
   et al., *New Insights Into Crustal and Mantle Flow Beneath the Red River
   Fault Zone and Adjacent Areas on the Southern Margin of the Tibetan Plateau
   Revealed by a 3-D Magnetotelluric Study*, JGR Solid Earth 125(10). It is
   unrelated to ETAS or forecast calibration.

   **Action:** deleted. The manuscript's unsupported attribution of a specific
   bias--variance result was removed.

3. **Former `Zechar2013`: real title combined with an unrelated DOI and wrong
   publication data.**

   The intended paper is Zechar, Gerstenberger, and Rhoades (2010), BSSA
   100(3), 1184--1195,
   [10.1785/0120090192](https://doi.org/10.1785/0120090192). The supplied DOI
   [10.1785/0120120081](https://doi.org/10.1785/0120120081) resolves to Bonner
   and Russell (2012), *Effects of Delay Firing on Surface Waves*.

   **Action:** corrected year, volume, issue, DOI, and citation key
   (`Zechar2010`).

## Real Entries with Material Metadata Errors

4. **`Mizrahi2024`: title and DOI real; author list and issue were wrong.**

   DOI [10.1029/2023RG000823](https://doi.org/10.1029/2023RG000823) confirms
   the title, journal, volume 62, and article number. The paper is issue 3, not
   issue 1. The original five-person author list was not a valid abbreviation:
   Nandan, Savran, and Lomax are not authors. The registered article has 25
   authors beginning Mizrahi, Dallo, van der Elst, Christophersen, and
   Spassiani.

   **Action:** replaced the author list and corrected the issue. This review
   supports the OEF overview and explicitly discusses New Zealand.

5. **`Savran2022`: real, with an incorrect author name.**

   DOI [10.1785/0220220033](https://doi.org/10.1785/0220220033) resolves to the
   stated pyCSEP paper. The fourth author is Khawaja M. Asim, not “M. A.
   Khawaja”; the third author is registered as Pablo Iturrieta.

   **Action:** corrected the names. This is the appropriate software and
   methods citation for pyCSEP and catalog-based evaluations.

## Real but Misapplied Citation

6. **Former `Mizrahi2021a`: real paper, wrong support for the EM method.**

   DOI [10.1785/0220200231](https://doi.org/10.1785/0220200231) resolves
   correctly to *The Effect of Declustering on the Size Distribution of
   Mainshocks*. Its subject is declustering-induced changes in the magnitude
   distribution of mainshocks. It does not document the
   expectation--maximization inversion used here.

   **Action:** removed from the method citation and bibliography.
   `Mizrahi2021b` explicitly states that it proposes ETAS calibration methods
   based on expectation maximization and is the appropriate source.

## Entries Verified as Real and Substantially Correct

7. **`Helmstetter2002`**:
   [10.1029/2001JB001580](https://doi.org/10.1029/2001JB001580). Correct
   authors, title, journal, volume, issue, year, and article number 2237. Its
   abstract explicitly defines the branching parameter and subcritical and
   supercritical regimes, including explosive growth in the latter. Strong
   match to the cited claim.

8. **`Jordan2011`**:
   [10.4401/ag-5350](https://doi.org/10.4401/ag-5350). Correct report,
   authorship, journal, volume, issue, and year. Strong source for definitions,
   state of knowledge, and guidelines for operational earthquake forecasting.
   It supports OEF generally, not details of this ETAS implementation.

9. **`Mizrahi2021b`**:
   [10.1029/2021JB022379](https://doi.org/10.1029/2021JB022379). Correct. Its
   abstract explicitly describes two ETAS calibration methods based on EM and
   studies temporal completeness and detection incompleteness. Strong match for
   both the inversion-method and incompleteness claims.

10. **`Ogata1988`**:
    [10.1080/01621459.1988.10478560](https://doi.org/10.1080/01621459.1988.10478560).
    Correct. Foundational source for temporal point-process earthquake models
    and residual analysis. Appropriate, although `Ogata1998` is the more direct
    source for the space--time formulation.

11. **`Ogata1998`**:
    [10.1023/A:1003403601725](https://doi.org/10.1023/A:1003403601725).
    Correct authors, title, journal, volume, issue, pages, year, and DOI. Strong
    match for the space--time ETAS description.

12. **`Rhoades2011`**:
    [10.2478/s11600-011-0013-5](https://doi.org/10.2478/s11600-011-0013-5).
    Correct. Appropriate for efficient earthquake-forecast tests. It should not
    be treated as documentation of the exact pyCSEP 0.8 implementation;
    `Savran2022` serves that role.

13. **`Seif2017`**:
    [10.1002/2016JB012809](https://doi.org/10.1002/2016JB012809). Correct. Its
    abstract explicitly addresses cutoff magnitude, finite catalog length,
    missing aftershocks, branching-ratio bias, and underestimated parameter
    errors. Strong match to the cited truncation and missing-data claims.

## Overall Assessment

The original file was not reliable enough for submission. Three of 13 entries
contained DOIs for unrelated papers, one long author list was largely
fabricated, and one real paper was cited for a method it does not describe.
After the corrections above, all remaining entries resolve to the intended
papers and their current uses are defensible at the level checked here.

This audit verifies bibliographic identity and high-level claim relevance. It
does not replace page-level checking of every technical statement against full
texts, nor does it establish that the bibliography is sufficiently broad for
BSSA peer review.

## Remaining Citation and Style Gaps

1. The GeoNet catalog extraction, magnitude-field semantics, and identification
   of the 4 March 2021 East Cape event are described from the archived data and
   code but lack a formal GeoNet data-service or event-product citation.

2. The statements about changing New Zealand network coverage and magnitude
   practice since 1960 are presented as plausible mechanisms. They need a
   New Zealand catalog-history or network-history reference before they can be
   stated as documented facts.

3. The empirical spatial-background sampler and approximate temporal sampler
   are implementation details established from source-code inspection, not
   from the cited papers. The software revision and archived source therefore
   remain essential evidence.

4. The bibliography is now internally consistent, but 11 references are sparse
   for a BSSA research paper covering ETAS estimation, parameter uncertainty,
   magnitude truncation, New Zealand catalog construction, and CSEP testing.
   Additional literature should be selected for scientific need, not added
   merely to increase the count.

5. The manuscript currently uses `\bibliographystyle{apalike}`. That style does
   not print DOI fields and is not a final BSSA production style. The entries
   should be rendered with the current SSA/BSSA template or style file at
   submission preparation. The bibliographic data themselves should remain
   valid under that change.
