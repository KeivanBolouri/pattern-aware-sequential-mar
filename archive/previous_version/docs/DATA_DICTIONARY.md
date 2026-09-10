# Data dictionary

All CSV files contain synthetic Monte Carlo results. No individual-level
clinical or confidential data are present. Subject-level samples are generated
inside each replication and are not retained.

## Primary oracle experiment

Directory: results/main_verified

### simulation_replicates.csv

One row per Monte Carlo replication and response scenario. There are 4,000
rows: 2 scenarios multiplied by 2,000 replications.

| Column | Meaning |
| --- | --- |
| scenario | CCMAR-compatible or Sequential-MAR response design |
| est_Full data | Oracle full-data ATE estimate |
| est_Coarsened CCMAR | Estimate after reducing response to complete/incomplete |
| est_Sequential patterns | Pattern-aware sequential estimate |
| se_* | Influence-function standard error for the corresponding estimator |
| replicate | Replication number within scenario |

### truth.csv

Numerically integrated counterfactual means and their difference. The target
ATE is 0.2558486202827889.

### pattern_proportions.csv

Mean proportions in the complete, L1-only, and neither-observed patterns.

### simulation_summary.csv and simulation_summary_with_mcse.csv

Bias, empirical standard deviation, RMSE, estimated standard error, coverage,
and Monte Carlo standard errors by scenario and estimator.

### efficiency_comparison_with_ci.csv

Paired empirical variance ratios and 10,000-resample bootstrap Monte Carlo
intervals. The Sequential-MAR row is explicitly labeled as not being an
efficiency comparison because the coarsened estimator is invalid there.

### theoretical_variance_check.csv

Numerical evaluation of the exact efficiency-bound identity with 1.5 million
integration draws.

## Oracle sensitivity experiment

Directory: results/oracle_sensitivity

### oracle_sensitivity_replicates.csv

One row per replication and design setting. There are 30,000 rows: 6 settings
multiplied by 5,000 replications.

| Column group | Meaning |
| --- | --- |
| scenario, comparison_type | Design identity and whether it evaluates efficiency or identification |
| intercept, gamma | Stage-two response-logit parameters |
| complete_proportion, l1_only_proportion, neither_proportion | Realized pattern proportions |
| est_* | Estimates from the three methods |
| se_* | Corresponding influence-function standard errors |

### oracle_sensitivity_design.csv

The six response designs and their calibrated parameters.

### oracle_sensitivity_summary.csv

Monte Carlo performance summaries and valid efficiency ratios for the three
CCMAR settings.

## Estimated-nuisance implementation check

Directory: results/crossfit

### crossfit_replicates.csv

One row per replication and scenario. There are 200 rows: 2 scenarios
multiplied by 100 replications.

The estimate_* and se_* columns contain cross-fitted estimates and standard
errors. The min_pi* columns record the smallest out-of-fold fitted observation
probabilities and support positivity diagnostics.

### crossfit_summary.csv

Bias, empirical standard deviation, RMSE, root mean estimated variance, and
coverage. This study is preliminary and should not be used as confirmatory
evidence.
