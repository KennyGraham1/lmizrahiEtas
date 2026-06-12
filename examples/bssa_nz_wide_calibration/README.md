# BSSA Manuscript Package

This directory contains a reproducible manuscript based on the completed
NZ-wide ETAS calibration sweep with forecast origin 1 January 2021.

Run the derived analysis from the repository root:

```bash
python3 examples/bssa_nz_wide_calibration/analyze_sweep.py
```

Compile the manuscript:

```bash
cd examples/bssa_nz_wide_calibration
pdflatex manuscript
bibtex manuscript
pdflatex manuscript
pdflatex manuscript
```

The author names, affiliations, acknowledgments, repository URL, archival DOI,
and contribution statements remain explicit placeholders because they cannot
be inferred from the run artifacts.
