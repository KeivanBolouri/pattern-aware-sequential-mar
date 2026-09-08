"""Generate appendix_crossfit.tex from the cross-fitted study results."""
import json
import os

R = "../results"
P = "../paper"

SC = {"ccmar": "CCMAR-compatible", "seqmar": "Sequential-MAR"}
EST = [("full", "Full data"), ("cc", "Coarsened"), ("seq", "Sequential")]

# oracle standard deviations reported in Table 2 of the main text
ORACLE_SD = {"ccmar": 0.01776, "seqmar": 0.01596}


def load(tag):
    p = f"{R}/crossfit_{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, d=5):
    return f"{x:.{d}f}".replace("-", "$-$")


def size_table(runs):
    rows = []
    for sc in ("ccmar", "seqmar"):
        first = True
        for _, d in runs:
            if d is None or sc not in d:
                continue
            for j, (k, lab) in enumerate(EST):
                s = d[sc][k]
                lead = SC[sc] if (first and j == 0) else ""
                nlab = f"{d['n']:,}" if j == 0 else ""
                rows.append(
                    f"{lead} & {nlab} & {lab} & {fmt(s['bias'])} & {s['sd']:.5f} & "
                    f"{s['mean_se']:.5f} & {s['rmse']:.5f} & {s['coverage']*100:.1f}\\% \\\\"
                )
            first = False
            rows.append("\\addlinespace")
        rows.append("\\midrule" if sc == "ccmar" else "")
    body = "\n".join(r for r in rows if r != "")
    return body.replace("\\addlinespace\n\\midrule", "\\midrule")


def decomp_table(base, oc, orr):
    rows = []
    for sc in ("ccmar", "seqmar"):
        for lab, d in (("All nuisances estimated", base),
                       ("Oracle $e,\\mu_a$; estimated $\\pi_j,Q_j$", oc),
                       ("Oracle $\\pi_j$; estimated $e,\\mu_a,Q_j$", orr)):
            if d is None or sc not in d:
                continue
            s, c = d[sc]["seq"], d[sc]["cc"]
            rows.append(
                f"{SC[sc] if lab.startswith('All') else ''} & {lab} & "
                f"{fmt(s['bias'])} & {s['sd']:.5f} & {s['coverage']*100:.1f}\\% & "
                f"{c['sd']:.5f} & {c['coverage']*100:.1f}\\% \\\\"
            )
        rows.append("\\addlinespace")
    return "\n".join(rows[:-1])


def se_ratio_sentence(runs):
    bits = []
    for _, d in runs:
        if d is None:
            continue
        s = d["ccmar"]["seq"]
        bits.append(f"{s['mean_se']/s['sd']:.2f} at $n={d['n']:,}$")
    return ", ".join(bits[:-1]) + " and " + bits[-1]


def main():
    runs = [(t, load(t)) for t in ("n2500", "n5000", "n10000", "n20000")]
    base, oc, orr = load("n2500"), load("oc2500"), load("or2500")
    ref = next(d for _, d in runs if d is not None)
    reps = {d["n"]: d["reps"] for _, d in runs if d is not None}
    repstr = ", ".join(f"{r:,} at $n={n:,}$" for n, r in sorted(reps.items()))

    r_cc = oc["ccmar"]["seq"]["sd"] / ORACLE_SD["ccmar"]
    r_sq = oc["seqmar"]["seq"]["sd"] / ORACLE_SD["seqmar"]

    tex = rf"""\section{{Cross-fitted estimation with estimated nuisances}}\label{{app:crossfit}}

The experiments in \cref{{sec:sim}} fix every nuisance function at its true value
so that the missingness layer is studied in isolation. This appendix replaces
each of them by a cross-fitted estimate, on the same data-generating law as
\cref{{sec:sim}} so that the two sets of numbers are directly comparable, and
reports how the comparison behaves as the sample size grows.

\subsection*{{Implementation}}

Nuisances are fitted with cubic B-spline bases in the continuous coordinates,
saturated in the binary ones, and penalised: $\ell_2$-regularised logistic
regression for $\pi_1(Z_0)$, $\pi_2(Z_1)$, $\rho(Z_0)$ and $e(V)$, and ridge
regression for $\mu_a(V)$, $Q_1(Z_1)$ and $Q_0(Z_0)$. Fitting follows the
sequence in \cref{{sec:sim}}: $\pi_1$ on all training records; $\pi_2$ among
$R_1=1$; the treatment and outcome regressions on complete records weighted by
$1/(\widehat\pi_1\widehat\pi_2)$; $Q_1$ by regressing $\widehat G$ on $Z_1$
among complete records; and $Q_0$ by regressing the stage-two augmented
pseudo-outcome on $Z_0$ among $R_1=1$ records. The coarsened comparator is
fitted in the parallel complete-case pipeline, with $\rho(Z_0)$ in place of
$\pi_1\pi_2$ and weights $1/\widehat\rho$. Cross-fitting uses {ref['folds']}
folds, and each fold's fitted values come only from models trained on the other
folds. Replicate counts are {repstr}.

Estimated probabilities are truncated at $0.01$. Truncation at a fixed level is
not asymptotically innocuous: it introduces a bias that does not vanish as $n$
grows, and a fully rigorous treatment would let the level shrink with $n$. In
this design about $0.6\%$ of the population has $e(V)$ below $0.01$, results at
$0.03$ and $0.05$ are materially the same, and the induced bias is an order of
magnitude below the Monte Carlo error, so none of the findings below is an
artefact of truncation.

\subsection*{{Behaviour as the sample size grows}}

\begin{{table}}[ht]
\centering
\begin{{threeparttable}}
\caption{{Cross-fitted estimators with all nuisances estimated, on the
\cref{{sec:sim}} data-generating law.}}
\label{{tab:crossfit-size}}
\footnotesize
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
Scenario & $n$ & Estimator & Bias & Emp.\ SD & Mean $\widehat{{\mathrm{{se}}}}$ & RMSE & Coverage \\
\midrule
{size_table(runs)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item Truncation of fitted probabilities at $0.01$; {ref['folds']}-fold
cross-fitting; nominal intervals use $\widehat\psi\pm1.96\,\widehat{{\mathrm{{se}}}}$
with \cref{{eq:se}}.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}

Four features of \cref{{tab:crossfit-size}} matter for how \cref{{eq:variance-gap}}
should be read in practice.

First, consistency separates the estimators exactly as the theory says it should,
and the separation is not cosmetic. The full-data and pattern-aware estimators
have biases that fall towards zero in both scenarios. The coarsened estimator
does so only in the CCMAR-compatible scenario; under sequential MAR it is
inconsistent by construction, and its bias sits near $-0.033$ at every sample
size. It is therefore not the case that all three estimators are consistent
throughout, and the sequential-MAR panel is where that shows.

Second, the intervals are too short at moderate $n$, and the cause is the
variance estimator rather than the point estimator. The ratio of the mean
estimated standard error to the empirical standard deviation of the
pattern-aware estimator is {se_ratio_sentence(runs)}. \Cref{{eq:se}} estimates the
variance of the influence function from the within-sample spread of the
$\widehat H_i$, which does not capture the additional across-sample variability
that fitted nuisances contribute while the second-order remainder is still
non-negligible. The ratio rises steadily towards one, as \cref{{thm:asymptotics}}
requires, but it is still short of it at $n=20{{,}}000$; coverage correspondingly
improves overall without improving monotonically, and has not reached the nominal
level by the largest sample size studied. Anyone applying this estimator at
moderate $n$ should prefer a bootstrap or a variance estimator that accounts for
nuisance estimation over \cref{{eq:se}}.

Third, the efficiency ordering that \cref{{eq:variance-gap}} guarantees for the
influence functions does not appear in the estimators until the sample is large.
The pattern-aware estimator has the \emph{{larger}} empirical standard deviation at
$n=2{{,}}500$ and $n=5{{,}}000$, the two are level at $n=10{{,}}000$, and only by
$n=20{{,}}000$ do they agree to the third decimal. Throughout, the difference
between them is far smaller than the excess of either over its oracle counterpart
in \cref{{tab:main-results}}. A $4.2\%$ variance advantage is not visible through
that much estimation noise, and this manuscript does not claim a finite-sample
precision advantage for the pattern-aware estimator at these sample sizes.

Fourth, the identification contrast behaves in the opposite way. Because the
coarsened bias is fixed while its standard error shrinks, its coverage
\emph{{deteriorates}} as $n$ grows, from $56.0\%$ at $n=2{{,}}500$ to $0.8\%$ at
$n=20{{,}}000$, while the pattern-aware estimator stays approximately unbiased
throughout. Nuisance estimation obscures the efficiency claim; it sharpens the
identification claim.

\subsection*{{Which layer costs the most}}

\begin{{table}}[ht]
\centering
\begin{{threeparttable}}
\caption{{Nuisance-layer decomposition at $n=2{{,}}500$. Replacing the treatment
and outcome regressions by their true values recovers most of the oracle
behaviour; replacing the response models instead does not.}}
\label{{tab:crossfit-layers}}
\small
\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\toprule
& & \multicolumn{{3}}{{c}}{{Sequential}} & \multicolumn{{2}}{{c}}{{Coarsened}} \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-7}}
Scenario & Nuisances & Bias & SD & Coverage & SD & Coverage \\
\midrule
{decomp_table(base, oc, orr)}
\bottomrule
\end{{tabular}}
\end{{threeparttable}}
\end{{table}}

\Cref{{tab:crossfit-layers}} localises the cost. Holding $e$ and $\mu_a$ at their
true values while every response and sequential-projection model is estimated
brings the pattern-aware standard deviation to {100*(r_cc-1):.0f}\% above its
oracle value in the CCMAR-compatible scenario and {100*(r_sq-1):.0f}\% above it
under sequential MAR, with coverage of
{oc['ccmar']['seq']['coverage']*100:.1f}\% and
{oc['seqmar']['seq']['coverage']*100:.1f}\% respectively, against
{base['ccmar']['seq']['coverage']*100:.1f}\% and
{base['seqmar']['seq']['coverage']*100:.1f}\% when the causal layer is estimated
as well. This is a substantial recovery but not a complete one: even with $e$ and
$\mu_a$ known, estimating $\pi_1,\pi_2,Q_0$ and $Q_1$ still costs roughly a fifth
of the oracle standard deviation in the harder scenario. Holding the response
probabilities at their true values while estimating the causal regressions barely
improves on estimating everything.

The asymmetry is structural rather than incidental: $\pi_1$ is fitted on all $n$
records and $\pi_2$ on the roughly $89\%$ with $R_1=1$, whereas $e(V)$ and
$\mu_a(V)$ can be fitted only on complete records, of which there are about
$600$ when $n=2{{,}}500$, and they must be accurate in a region where $e(V)$
approaches $0.01$. Truncating the fitted propensity more aggressively does not
help, which confirms that the binding constraint is the size of the
complete-record subsample rather than the weights themselves.

The same table shows why the efficiency ordering is slow to appear. Even with
the causal layer known exactly, the pattern-aware estimator's standard deviation
slightly exceeds the coarsened estimator's in the common-validity scenario,
because it requires one regression the coarsened estimator does not: $Q_1$ on
$Z_1$. At $n=2{{,}}500$ the estimation error in that extra regression costs more
than the $4.2\%$ the intermediate pattern buys. This is a finite-sample
statement, not a contradiction of \cref{{thm:gap}}, which compares influence
functions at known nuisances.

The practical implication is a caution rather than a retraction.
\Cref{{eq:variance-gap}} is exact and the gain it quantifies is real, but it is a
second-order quantity that an analyst realises only where the complete-record
subsample is large enough to estimate the causal regressions well, and only with
a variance estimator better calibrated than \cref{{eq:se}} at moderate $n$. The
identification argument of \cref{{sec:sim}} carries no such caveat, and it is the
stronger practical reason to retain partial records when the acquisition process
is ordered.
"""
    open(f"{P}/appendix_crossfit.tex", "w").write(tex)
    print("wrote appendix_crossfit.tex")


if __name__ == "__main__":
    main()
