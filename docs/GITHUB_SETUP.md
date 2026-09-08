# GitHub repository

## Status

A public GitHub repository holds this project for journal access.

| Item | Value |
| --- | --- |
| Owner | KeivanBolouri |
| Repository name | pattern-aware-sequential-mar |
| Visibility | Public |
| Default branch | main |
| Web URL | https://github.com/KeivanBolouri/pattern-aware-sequential-mar |
| Clone URL | https://github.com/KeivanBolouri/pattern-aware-sequential-mar.git |
| Short description | Efficient causal ATE estimation with monotone partially observed confounders under sequential MAR. |

This repository is public so that journal editors and reviewers can open the reproducibility package without a GitHub login.

## What is on GitHub

The GitHub repository root is this folder (`pattern-aware-sequential-mar`). `README.md` is at the repository root. GitHub was not asked to add a second README, gitignore, or license.

Included:

- manuscript source (`paper.tex`, `appendix_crossfit.tex`);
- Python analysis code, an independent-check implementation, and an R mirror;
- verified synthetic Monte Carlo results;
- manuscript figures in `figures/`;
- documentation in `docs/`;
- citation metadata (`CITATION.cff`).

Not included:

- local virtual environments (`.venv/`);
- scratch reproduction output (`scratch_results/`);
- a software license (none has been chosen yet);
- patient, clinical, or other confidential data (none are present).

## Clone

    git clone https://github.com/KeivanBolouri/pattern-aware-sequential-mar.git
    cd pattern-aware-sequential-mar

Then follow the Quick start section in `README.md`.

## Suggested GitHub topics

Add these on the repository About page if they are not already set:

    causal-inference
    missing-data
    semiparametric-statistics
    influence-functions
    average-treatment-effect
    reproducible-research

## Update after local changes

From this folder:

    git add .
    git commit -m "Describe the change"
    git push origin main

## Recreate the repository from scratch

Use this only if the GitHub copy does not exist yet. Do not initialize the GitHub side with a README, gitignore, or license.

    git init
    git add .
    git commit -m "Add manuscript and verified reproducibility materials"
    git branch -M main
    gh repo create pattern-aware-sequential-mar --public --source=. --remote=origin --push

## Before making it public

1. Confirm the displayed author name and affiliation.
2. Select a license only after deciding how others may reuse the code and data.
3. Do not add another person as a coauthor without that person's explicit agreement.
4. Keep the wording "unpublished manuscript" until the article has actually completed peer review.
5. If a DOI is later created through Zenodo, add it to `CITATION.cff` and the README.
6. Complete the checklist in the README section "Before public release".

## Access

Visibility is Public. Anyone with the URL can view and clone the repository.
