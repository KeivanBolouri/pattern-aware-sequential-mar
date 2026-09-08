# Efficient estimation and the cost of complete-case coarsening under monotone sequential MAR

This repository contains the manuscript, source code, synthetic simulation
outputs, result summaries, and figures for a two-stage sequential
missing-at-random extension of causal ATE estimation with missing confounders.

The public GitHub repository is:

https://github.com/KeivanBolouri/pattern-aware-sequential-mar

See docs/GITHUB_SETUP.md for clone and update steps.

## Scientific status

This repository is the reproducibility package for the accompanying manuscript.

The generic sequential-MAR augmentation idea is established in the monotone
missing-data literature. The contribution studied here is narrower:

1. specialization to the full-data efficient score for a point-exposure ATE;
2. a formal observed-data tangent-space derivation;
3. an exact variance penalty for collapsing the intermediate pattern; and
4. separation of efficiency loss under common validity from identification
   failure when complete-case MAR is false.

## Main verified result

In the CCMAR-compatible oracle simulation with 2,000 replicates and
2,500 observations per replicate:

- coarsened/sequential variance ratio: 1.046661;
- variance reduction from retaining the L1-only pattern: 4.4581%;
- paired-bootstrap Monte Carlo 95% interval: 2.7091% to 6.2953%; and
- independently evaluated theoretical variance reduction: 4.2251%.

The earlier 22.6% prototype value is superseded and should not be reported.

When second-stage observation depended on observed L1, complete-case MAR was
invalid. In that design, the coarsened estimator had bias -0.03222 and 46.9%
coverage; the sequential estimator had bias 0.00044 and 95.45% coverage. This
is an identification comparison, not an efficiency comparison.

## Repository contents

    .
    |-- README.md
    |-- CITATION.cff
    |-- Makefile
    |-- paper.tex
    |-- appendix_crossfit.tex
    |-- requirements.txt
    |-- requirements-lock.txt
    |-- code/
    |   |-- validate_sequential_mar.py
    |   |-- summarize_verified_simulation.py
    |   |-- verify_variance_identity.py
    |   |-- oracle_sensitivity_simulation.py
    |   |-- crossfit_sequential_ate.py
    |   |-- sequential_mar_extension.R
    |   |-- make_manuscript_figures.py
    |   |-- check_repository.py
    |   +-- independent_check/
    |-- results/
    |   |-- main_verified/
    |   |-- oracle_sensitivity/
    |   +-- crossfit/
    |-- figures/
    |-- docs/
    +-- output/pdf/

The replicate-level CSV files are synthetic Monte Carlo outputs. No patient,
clinical, or other confidential data are included.

## Quick start

Create a Python environment and install the dependencies:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

For the exact package versions used in the verification environment, use:

    python -m pip install -r requirements-lock.txt

On Windows PowerShell, activate the environment with:

    .venv\Scripts\Activate.ps1

Check the repository:

    python code/check_repository.py
    python -m py_compile code/*.py

Or, where GNU Make is available:

    make check

## Reproduce the primary analysis

The following commands write new results to scratch_results and do not
overwrite the verified files.

    python code/validate_sequential_mar.py \
      --n 2500 \
      --reps 2000 \
      --seed 20260831 \
      --outdir scratch_results/main

    python code/summarize_verified_simulation.py \
      --replicates scratch_results/main/simulation_replicates.csv \
      --truth scratch_results/main/truth.csv \
      --outdir scratch_results/main \
      --bootstrap-reps 10000 \
      --seed 20260904

    python code/verify_variance_identity.py \
      --validator-dir code \
      --draws 1500000 \
      --sample-size 2500 \
      --seed 86420 \
      --out scratch_results/main/theoretical_variance_check.csv

The stored authoritative outputs are in results/main_verified.

## Reproduce the sensitivity study

    python code/oracle_sensitivity_simulation.py \
      --validator-dir code \
      --n 2500 \
      --reps 5000 \
      --seed 20260904 \
      --outdir scratch_results/oracle_sensitivity

## Run the preliminary cross-fitted implementation

    python code/crossfit_sequential_ate.py \
      --n 4000 \
      --reps 100 \
      --folds 2 \
      --seed 20260904 \
      --outdir scratch_results/crossfit

This estimated-nuisance experiment is preliminary. It has only 100 replicates
per scenario and is not the headline evidence.

## Rebuild figures and manuscript

Recreate both manuscript figures from the archived summaries:

    python code/make_manuscript_figures.py

This writes:

    figures/estimator_comparison.png
    figures/sensitivity_summary.pdf
    figures/sensitivity_summary.png

Compile the manuscript:

    latexmk -xelatex -interaction=nonstopmode -halt-on-error paper.tex

The compiled review PDF is included at:

    output/pdf/pattern_aware_sequential_MAR_paper.pdf

## Documentation

- docs/METHOD_SUMMARY.md: sequential-MAR notation, efficient influence function, and the exact variance identity.
- docs/DATA_DICTIONARY.md: row counts, column definitions, and the difference between replicate-level data and summaries.
- docs/VERIFICATION.md: completed reproducibility audit and headline numbers.
- docs/ENVIRONMENT.md: Python, scientific-library, and TeX versions used in the audit.
- docs/GITHUB_SETUP.md: GitHub location and clone commands.

## Data and results

See docs/DATA_DICTIONARY.md for row counts, column definitions, and the
difference between replicate-level data and summary results. See
docs/VERIFICATION.md for the completed reproducibility audit.

## Relationship to the original CCMAR repository

The original implementation associated with Levis, Mukherjee, Wang, and
Haneuse is available at:

https://github.com/alexlevis/flex-ate-confounders-MAR

That external repository is prior work and is not copied here. The files in
this repository implement and assess the separate sequential-pattern
extension.

## Before public release

- Obtain independent review of the efficient-influence-function and remainder
  proofs.
- Expand the estimated-nuisance simulation to at least 1,000 to 2,000
  replicates per design point.
- Add model-misspecification, learner, and near-positivity experiments.
- Preferably add a real-data or high-fidelity plasmode application.
- Complete correspondence, funding, and conflict-of-interest fields.
- Choose an explicit software and data license. No license is included in this
  package, so public reuse is not automatically granted.

## Citation

Citation metadata are provided in CITATION.cff. Until peer review is complete,
cite this as an unpublished manuscript or software archive, not as a journal
article.
