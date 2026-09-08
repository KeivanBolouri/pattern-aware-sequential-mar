# Independent reproduction check

A second, self-contained implementation of the manuscript's simulation study,
written from the equations printed in Section 6.1 alone — no code from the
author's archive was consulted. Its purpose is external verification, and, for
`crossfit.py`, to supply the cross-fitted study reported in Appendix D.

    pip install numpy scipy pandas matplotlib scikit-learn

| File | What it does |
|---|---|
| `dgp.py` | The full-data law, the numerical refactorisation giving `e(V)` and `mu_a(V)`, the oracle projections `Q0`, `Q1`, the true ATE, and both response mechanisms. All nuisances come from deterministic quadrature (Gauss–Legendre in Y, Gauss–Hermite in L2) rather than Monte Carlo. |
| `oracle_sim.py` | The oracle-nuisance simulation: full-data, coarsened and pattern-aware estimators, plus the paired bootstrap for the variance reduction. |
| `sensitivity_check.py` | Solves for the stage-two intercepts matching the reported complete-case shares, evaluates the exact variance identity by quadrature (Table 3), and reruns the identification sensitivity design (Table 4). |
| `crossfit.py` | The cross-fitted estimated-nuisance study of Appendix D, including the `--oracle-causal` / `--oracle-response` flags used for the nuisance-layer decomposition. |
| `make_figures.py` | Regenerates both manuscript figures. |
| `gen_appendix.py` | Builds `appendix_crossfit.tex` from the stored JSON results. |

## Reproducing

    python oracle_sim.py --reps 2000                       # Table 2 analogue
    python sensitivity_check.py                            # Tables 3 and 4
    python crossfit.py --n 2500 --reps 1000 --tag n2500     # Appendix D
    python crossfit.py --n 2500 --reps 1000 --oracle-causal   --tag oc2500
    python crossfit.py --n 2500 --reps 1000 --oracle-response --tag or2500
    python gen_appendix.py

## Findings that differ from the archived run

- True ATE: 0.2558505161 here (three independent quadrature routes, converged to
  ten digits) against 0.2558486 in the manuscript.
- Var(D_cc) = 0.79524, Var(D_seq) = 0.76224, reduction 4.15% here, against
  0.77466 / 0.74193 / 4.23% obtained by Monte Carlo integration in the archive.
  The ratio agrees to within 0.1%.
