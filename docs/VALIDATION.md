# Validation and limits

The corrected numerical package passes the enclosed implementation and result checks. This validates the specified calculations and resolves the identified code defects; it is not a guarantee of journal acceptance or of performance outside the studied settings.

## Completed numerical checks

- Five regression checks pass: observability with actual missing values; response-oracle training weights and absence of response fits; direct all-oracle equation evaluation; centered variance reduction; and the bounded design's target and probability range.
- Current-source reruns of the first common-validity replicate in both full-data designs match all archived estimate, SE, interval, and coverage fields exactly.
- All 18 estimated-nuisance/diagnostic configurations, two primary oracle scenarios, and six sensitivity configurations are present.
- All 179,500 estimate rows from 48,500 simulated datasets pass checks for replicate counts, unique indices, finite values, standard errors, interval endpoints, coverage indicators, and JSON summary agreement.
- The sensitivity response-share calibration matches its population target within 1e-12.
- The variance-gap identity is checked at two numerical resolutions. Its final residual is 5.83e-16; doubling resolution changes the reported variances by less than 1e-8.
- A sensitivity replicate file found incomplete during the audit was regenerated from the same seed. Every summary field and bootstrap endpoint matches its original completed summary exactly. The regenerated full file passes the same row-level checks.

Details are in `results/verification.json`, `results/implementation_checks.txt`, `results/selected_replicate_replay.json`, and `results/quadrature.json`.

## Results that matter for interpretation

The deterministic efficiency reduction under common validity is **4.15%**. The regenerated oracle experiment estimates **5.83%**, with a paired-bootstrap Monte Carlo 95% interval of **4.03% to 7.57%**. These are different quantities: a numerical population calculation and a finite Monte Carlo estimate.

The new bounded-propensity design uses estimated nuisances and a true ATE of 0.15:

| Sample size | Missingness scenario | Sequential bias | Nominal 95% coverage |
|---|---|---:|---:|
| 2,500 | Common validity | -0.00001 | 96.0% |
| 2,500 | Sequential MAR only | 0.00000 | 95.4% |
| 5,000 | Common validity | -0.00027 | 96.4% |
| 5,000 | Sequential MAR only | -0.00003 | 95.6% |

Each row uses 1,000 replicates. Coverage is close to nominal, with mild overcoverage in some configurations; Monte Carlo uncertainty must be considered. These results do not establish the nuisance convergence rates required by the asymptotic theorem.

The Gaussian stress design remains difficult: sequential coverage is 83.8% to 89.6% over the estimated-nuisance configurations. It lacks uniform treatment positivity, and the code corrections do not repair the normal intervals. This limitation is stated in the abstract, simulation section, discussion, and Appendix D. No unsupported bootstrap remedy, fixed-truncation sensitivity result, or general finite-sample efficiency advantage is claimed.

## Manuscript checks

The 24-page PDF compiles with no LaTeX errors, warnings, or overfull/underfull box warnings. All pages were inspected visually, and both GitHub URLs return HTTP 200. The seven tables and two figures are present.



