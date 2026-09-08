# Verification record

## What was independently rerun

- Primary oracle simulation: `n=2500`, `reps=2000`, seed `20260831`.
- Monte Carlo summarization and 10,000-resample paired bootstrap.
- Numerical efficiency-bound identity with 1.5 million integration draws.
- Sensitivity and cross-fitted summaries, recomputed from replicate-level CSVs.
- Cross-fitted implementation smoke test.
- Python syntax compilation for every included `.py` file.

The regenerated primary truth, replicate, summary, pattern-proportion,
bootstrap-summary, and variance-identity files matched the archived outputs
byte for byte. Summary files for the sensitivity and estimated-nuisance studies
matched numerically.

## Authoritative headline result

Under the common-validity design, the empirical coarsened/sequential variance
ratio was 1.046661, corresponding to a 4.4581% reduction for the sequential
estimator. The 95% paired-bootstrap Monte Carlo interval was 2.7091% to 6.2953%.
The independently evaluated theoretical reduction was 4.2251%.

The earlier 22.6% prototype result is superseded and must not be reported with
this manuscript.

## Known limitations

- The main evidence uses oracle nuisance functions and synthetic data.
- The estimated-nuisance study has only 100 replicates per scenario.
- The R mirror was not rerun in the audit environment; Python is authoritative.
- No real-data or plasmode application is included.
- Independent semiparametric review is still required before submission.
