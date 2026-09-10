"""Generate all numerical LaTeX tables and result prose from completed JSON runs.

Missing files are fatal. No table silently omits a simulation configuration.
"""
import json
from pathlib import Path
from crossfit import ROOT

RESULTS,PAPER = ROOT/'results',ROOT/'paper'
SCENARIOS = ('ccmar','seqmar')
LABELS = {'full':'Full data','cc':'Coarsened','seq':'Sequential'}


def load(name):
    return json.loads((RESULTS/name).read_text())


def cross(design,n,sc,variant=None):
    tag = f'{design}_n{n}' if variant is None else f'{design}_{variant}{n}'
    return load(f'crossfit_{tag}_{sc}.json')


def f(v,digits=5):
    return f'${v:.{digits}f}$'


def pct(v,digits=1):
    return f'{100*v:.{digits}f}\\%'


def table(caption,label,columns,header,rows,note='',small='small'):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering',r'\begin{threeparttable}',
       r'\caption{'+caption+'}',r'\label{'+label+'}', '\\'+small,
       r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}'+columns+r'@{}}',r'\toprule',
       header+r' \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',
       r'\begin{tablenotes}[flushleft]\footnotesize',r'\item '+note,r'\end{tablenotes}',
       r'\end{threeparttable}',r'\end{table}',''])


def empirical_table(design,sizes,label,caption):
    rows=[]
    for si,sc in enumerate(SCENARIOS):
        rows.append(r'\multicolumn{7}{l}{\textit{'+('Common validity' if sc=='ccmar' else 'Sequential MAR only')+r'}} \\')
        for n in sizes:
            result = cross(design,n,sc)
            for j,k in enumerate(('full','cc','seq')):
                d=result[k]
                cells = [f'{n:,}' if j==0 else '',LABELS[k],f(d['bias']),f(d['sd']),f(d['mean_se']),f(d['rmse']),pct(d['coverage'])]
                rows.append(' & '.join(cells)+r' \\')
            rows.append(r'\addlinespace[3pt]')
    note=r'SD is the empirical standard deviation; mean SE is the average estimated standard error. Coverage uses nominal 95\% normal intervals. '
    if design=='stress':
        note+=r'Replicate counts are 1,000, 1,000, 500, and 250 at increasing sample sizes. Coverage Monte Carlo SE is at most 1.58, 1.58, 2.24, and 3.16 percentage points, respectively.'
    else:
        note+=r'Each configuration uses 1,000 replicates; the true ATE is 0.15. Coverage Monte Carlo SE is at most 1.58 percentage points (about 0.69 near 95\% coverage).'
    return table(caption,label,'rlrrrrr',r'$n$ & Estimator & Bias & SD & Mean SE & RMSE & Coverage',rows,note)


def main():
    oracle=load('oracle.json')
    q=load('quadrature.json')
    fine=q['fine']
    sens=load('sensitivity.json')
    eff=oracle['efficiency_ccmar']
    seq=oracle['seqmar']['seq']
    cc=oracle['seqmar']['cc']
    rows=[]
    for sc in SCENARIOS:
        for j,k in enumerate(('full','cc','seq')):
            d=oracle[sc][k]
            rows.append(' & '.join([('Common validity' if sc=='ccmar' else 'Sequential MAR only') if j==0 else '',LABELS[k],f(d['bias']),f(d['sd']),f(d['rmse']),pct(d['coverage'],2)])+r' \\')
        rows.append(r'\addlinespace')
    main_table=table(r'Oracle simulation in the Gaussian stress design ($n=2{,}500$, 2,000 replicates per scenario).',
        'tab:main-results','llrrrr','Scenario & Estimator & Bias & SD & RMSE & Coverage',rows,
        r'The target ATE is '+f(oracle['ccmar']['psi'],10)+r'. The coarsened column uses true $G$, $Q_0$, and $\rho$. Bias and coverage Monte Carlo standard errors are recorded in the supplement; coverage MCSE is at most 1.12 percentage points.')
    rows=[]
    for label,d in zip(('Low','Moderate','High'),sens[:3]):
        rows.append(' & '.join([label,f(d['intercept'],5),pct(d['pattern_complete'],2),pct(d['pattern_l1_only'],2),f'{d["reduction_pct"]:.2f}\\%',f'{d["boot_lo"]:.2f}--{d["boot_hi"]:.2f}'])+r' \\')
    eff_table=table('Efficiency sensitivity under common validity (5,000 replicates per row).','tab:sensitivity-efficiency','lrrrrr',
        r'Missingness & Intercept & Complete & $L_1$ only & Reduction & MC interval',rows,
        r'Reduction is $100\{1-\widehat{\Var}(\widehat\psi_{\seq})/\widehat{\Var}(\widehat\psi_{\cc})\}$, using centered empirical variances. The interval is a paired-bootstrap 95\% Monte Carlo interval from 10,000 resamples; it is expressed in percentage points.')
    rows=[]
    for d in sens[3:]:
        rows.append(' & '.join([f(d['gamma'],2),f(d['intercept'],5),f(d['cc']['bias']),pct(d['cc']['coverage'],2),f(d['seq']['bias']),pct(d['seq']['coverage'],2)])+r' \\')
    id_table=table(r'Identification sensitivity with population complete-case proportion fixed at 24\% (5,000 replicates per row).','tab:sensitivity-identification','rrrrrr',
        r'$\gamma$ & Intercept & CC bias & CC coverage & Seq. bias & Seq. coverage',rows,
        r'CC denotes the coarsened oracle procedure using $G,Q_0,\rho$; Seq. denotes sequential augmentation. Intercepts solve the stated population response-share equation by deterministic integration.')
    bounded_table=empirical_table('bounded',[2500,5000],'tab:bounded','Estimated-nuisance performance with bounded treatment probabilities.')
    bounded_cov=[cross('bounded',n,sc)['seq']['coverage'] for n in (2500,5000) for sc in SCENARIOS]
    stress_cov=[cross('stress',n,sc)['seq']['coverage'] for n in (2500,5000,10000,20000) for sc in SCENARIOS]
    # Fixed prose templates expose actual values; no favorable-result filtering.
    simulation=r'''\section{Simulation studies}\label{sec:sim}

\subsection{Gaussian stress design and response mechanisms}

The first design isolates the missingness comparison with oracle nuisance functions, then examines the effect of estimating those functions. There are no fully observed baseline covariates $W$. The full-data law is specified retrospectively:
\begin{align*}
A&\sim\mathrm{Bernoulli}(0.5),\\
Y\mid A=0&\sim\mathrm{Beta}(2,4),\qquad
Y\mid A=1\sim\mathrm{Beta}(4,2),\\
L_1\mid A,Y&\sim\mathrm{Bernoulli}\{\expit(-0.6+0.5A+0.25Y+0.1AY)\},\\
L_2\mid A,Y,L_1&\sim N(A+Y+2.5L_1Y,1.25^2).
\end{align*}
Refactorization of this law yields $e(V)$ and $\mu_a(V)$ by numerical quadrature. The ATE is $\psi=0.2558505161$ to ten decimal places; all summaries use the unrounded computed value. Stage-one response is
\[
\pi_1=\expit(2+0.1A+0.1Y).
\]
The two stage-two mechanisms are
\[
\pi_2=\begin{cases}
\expit(-1.1+0.1A+0.1Y),&\text{common validity},\\
\expit(-1.9+0.1A+0.1Y+2.2L_1),&\text{sequential MAR only}.
\end{cases}
\]
The first permits an efficiency comparison and the second illustrates identification failure after coarsening. The oracle experiment uses 2,000 samples of size 2,500 per mechanism. All three estimators are means of their influence-function pseudo-outcomes, with normal intervals based on the empirical variance within each sample. The coarsened oracle procedure uses the true causal functions and $Q_0$ but weights by $1/\rho(Z_0)$; the supplement also reports the alternative augmentation $Q_0^{\cc}$. Neither oracle version is a feasible causal estimator outside common validity.

This Gaussian design does not satisfy the uniform treatment-positivity condition in \cref{ass:causal}. The unbounded support of $L_2$ permits treatment probabilities arbitrarily close to zero or one. The weaker moment conditions $\E(1/e)=4.04$ and $\E\{1/(1-e)\}=5.65$ nevertheless give a finite full-data efficiency bound because $Y$ is bounded. Deterministic quadrature gives $\Var(G)=0.27924$ and $\E(G^4)=105.28$. These features make this a stress design for learning the causal nuisance functions; it is not an empirical verification of \cref{thm:asymptotics}.

\subsection{Oracle comparison}

'''+main_table+f'''
The centered empirical variance reduction under common validity was {eff['reduction_pct']:.2f}\\%, with paired-bootstrap Monte Carlo 95\\% interval {eff['boot_lo']:.2f}\\%--{eff['boot_hi']:.2f}\\% (10,000 resamples). This statistic uses deviations from the replicate means, rather than squared deviations from the true ATE, which would estimate a mean-squared-error reduction.

Deterministic product quadrature gives influence-function variances {fine['var_cc_common']:.5f} and {fine['var_seq_common']:.5f} for the coarsened and sequential procedures, respectively. Their difference is {fine['gap']:.5f}, corresponding to a {fine['reduction_pct']:.2f}\\% reduction. Doubling the quadrature resolutions changes each variance by less than $10^{{-8}}$, and the numerical residual in \\cref{{eq:variance-gap}} is below $10^{{-12}}$. These are converged numerical evaluations, not symbolic integrations. The {fine['reduction_pct']:.2f}\\% value lies within the simulation interval; the interval's width also shows why a finite Monte Carlo variance ratio should not be equated with the efficiency bound.

In the sequential-only scenario, the coarsened oracle bias was {f(cc['bias'])} with {pct(cc['coverage'],2)} coverage, whereas the sequential bias was {f(seq['bias'])} with {pct(seq['coverage'],2)} coverage. Quadrature gives population coarsened bias {f(fine['coarsened_bias_outside'])}. Replacing $Q_0$ by $Q_0^{{\\cc}}$ gives simulated bias {f(oracle['seqmar']['cc_ccproj']['bias'])}; the two population biases coincide, as shown above. This is an identification comparison.

'''+r'''\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{estimator_comparison.pdf}
\caption{Oracle estimates from the same replicates used in \cref{tab:main-results}. Boxes show quartiles, whiskers extend to 1.5 times the interquartile range, and points beyond the whiskers are shown. Dashed lines mark the true ATE.}
\label{fig:main-boxplot}
\end{figure}

'''+f'''The population pattern proportions obtained by quadrature are {pct(fine['complete_common'],2)} complete, {pct(1-fine['complete_common']-fine['neither'],2)} $L_1$ only, and {pct(fine['neither'],2)} neither under common validity. The corresponding sequential-only proportions are {pct(fine['complete_outside'],2)}, {pct(1-fine['complete_outside']-fine['neither'],2)}, and {pct(fine['neither'],2)}.

'''+r'''\subsection{Sensitivity to the second-stage response mechanism}

The sensitivity experiment keeps the Gaussian full-data law and uses $\pi_2=\expit(b+0.1A+0.1Y+\gamma L_1)$. Under common validity, $\gamma=0$ and the intercept $b$ is calibrated to population complete-case proportions 0.5117, 0.2400, and 0.1161. In the identification experiment, $\gamma$ is 0.75, 1.50, or 2.25 and $b$ is calibrated separately to maintain a population complete-case proportion of 0.24. Each configuration uses 5,000 samples of size 2,500. The intercepts, empirical response shares, biases, coverage, and uncertainty in the variance reduction are reported below. Calibration uses deterministic integration; reported empirical shares need not equal the population targets exactly.

'''+eff_table+id_table+r'''
\begin{figure}[htbp]
\centering
\includegraphics[width=0.98\textwidth]{sensitivity_summary.pdf}
\caption{Sensitivity summaries from the same results as \cref{tab:sensitivity-efficiency,tab:sensitivity-identification}. Left: centered empirical variance reduction and paired-bootstrap Monte Carlo intervals. Right: bias and Monte Carlo intervals computed as the estimated bias plus or minus 1.96 Monte Carlo standard errors.}
\label{fig:sensitivity}
\end{figure}

\FloatBarrier
\subsection{A design with bounded treatment probabilities}\label{sec:bounded}

To assess estimated-nuisance behavior under uniform positivity, we also specify a prospective full-data law. Independently, $L_1\sim\mathrm{Bernoulli}(0.5)$ and $L_2\sim\mathrm{Uniform}(-1,1)$. Let
\begin{align*}
A\mid L_1,L_2&\sim\mathrm{Bernoulli}\{\expit(-0.3+0.6L_1+0.7L_2)\},\\
\mu_a(L_1,L_2)&=0.25+0.15a+0.10L_1+0.05L_2,\\
Y\mid A,L_1,L_2&\sim\mathrm{Beta}\{8\mu_A,8(1-\mu_A)\}.
\end{align*}
Potential outcomes may be generated from these conditional beta laws independently of treatment given the confounders, ensuring conditional exchangeability. The ATE is exactly $0.15$, and $\expit(-1)\le e(V)\le\expit(1)$, approximately $[0.269,0.731]$. Outcomes and response probabilities are also bounded as required for the moment and positivity conditions. The two response mechanisms are the same functions of $A,Y,L_1$ as above. This design supplies an instance of the identifying assumptions and uniform positivity; it does not by itself verify the nuisance-rate conditions.

All nuisance functions are estimated with the fixed five-fold procedure in \cref{app:crossfit}. Each configuration uses 1,000 replicates. No tuning parameter is selected from the Monte Carlo performance. The full-data benchmark uses all simulated confounders; each incomplete-data estimator receives only its observed covariates.

'''+bounded_table+f'''Across the four bounded-design configurations, sequential coverage ranges from {pct(min(bounded_cov),1)} to {pct(max(bounded_cov),1)}. The bias, empirical variability, and average estimated standard errors are shown together so that coverage can be assessed against its Monte Carlo uncertainty. The common-validity and sequential-only scenarios must still be distinguished when interpreting the coarsened results. A smaller bias does not guarantee a smaller finite-sample mean squared error: in the sequential-only configuration at $n=2{{,}}500$, sequential RMSE is ${cross("bounded",2500,"seqmar")["seq"]["rmse"]:.5f}$, compared with ${cross("bounded",2500,"seqmar")["cc"]["rmse"]:.5f}$ for the biased coarsened estimator. The efficiency-bound comparison does not establish a uniform finite-sample MSE ordering.

'''+r'''\subsection{Estimated nuisances in the stress design}\label{sec:estimated-nuisance}

'''+f'''The corresponding Gaussian-design results appear in \\cref{{app:crossfit}}, including sample sizes 2,500 through 20,000 and oracle substitutions for selected nuisance layers. With all nuisances estimated, sequential coverage ranges from {pct(min(stress_cov),1)} to {pct(max(stress_cov),1)} across the eight sample-size and scenario configurations. The observed undercoverage is a limitation of this estimator and learner in that design; a reduction in bias or improvement with sample size does not establish the asymptotic rate conditions. The oracle-layer comparisons are diagnostics of the specified fitting procedure, rather than a general decomposition of all sources of finite-sample error.

\\FloatBarrier
'''
    (PAPER/'simulation.tex').write_text(simulation)
    stress_table=empirical_table('stress',[2500,5000,10000,20000],'tab:crossfit-n','Estimated-nuisance performance in the Gaussian stress design.')
    layerrows=[]
    for sc in SCENARIOS:
        for j,(variant,label) in enumerate([(None,'None'),('oc','Causal'),('or','Response'),('alloracle','All')]):
            d=cross('stress',2500,sc,variant)
            layerrows.append(' & '.join([('Common' if sc=='ccmar' else 'Sequential') if j==0 else '',label,
                f(d['seq']['bias']),f(d['seq']['sd']),pct(d['seq']['coverage']),
                f(d['cc']['bias']),f(d['cc']['sd']),pct(d['cc']['coverage'])])+r' \\')
        layerrows.append(r'\addlinespace')
    layers=table('Oracle-substitution diagnostics in the Gaussian design ($n=2{,}500$).','tab:crossfit-layers','llrrrrrr',
       r'Scenario & Known & Seq. bias & Seq. SD & Seq. cov. & CC bias & CC SD & CC cov.',layerrows,
       r'Causal: true $e,\mu_0,\mu_1$. Response: true $\pi_1,\pi_2,\rho$ throughout training and validation. All: true causal, response, and projection functions, with $Q_0^{\cc}$ for the coarsened procedure. There are 1,000 replicates per row except 500 for All. Corresponding configurations share random-number streams.','footnotesize')
    ratios=[cross('stress',n,'ccmar')['seq']['mean_se']/cross('stress',n,'ccmar')['seq']['sd'] for n in (2500,5000,10000,20000)]
    oc=cross('stress',2500,'ccmar','oc')['seq']; rr=cross('stress',2500,'ccmar','or')['seq']
    base=cross('stress',2500,'ccmar')['seq']
    appendix=r'''\section{Cross-fitted estimation with estimated nuisances}\label{app:crossfit}

\subsection*{Observed-data implementation}

Each sample is randomly divided into five folds. All basis transformations and fitted models used on a validation fold are trained on the other four folds. The outcome basis uses all training values of $Y$. The incomplete-data causal basis uses only $L_2$ values with $R_2=1$ in the training set. A separate full-data benchmark chooses its own basis using all training $L_2$ values. Missing covariates are represented by unavailable values: the implementation reads $L_1$ only when $R_1=1$ and $L_2$ only when $R_2=1$. There is no fitting, prediction, or knot selection using hidden simulated confounders in the incomplete-data procedures.

Continuous coordinates use cubic B-splines with six empirical quantile knots, including the boundary knots, and no spline bias column. Each fitted regression includes an intercept. The design for $Z_0$ contains the spline in $Y$, $A$, and their interactions. The design for $Z_1$ is saturated in $(A,L_1)$, including their products with the spline in $Y$. The causal design contains the spline in $L_2$, $L_1$, and their interactions. Logistic regressions use an $\ell_2$ penalty with inverse regularization parameter $C=1$; ridge regressions use penalty $10^{-3}$. Hyperparameters are fixed across designs, sample sizes, and replicates. Logistic fitting allows at most 2,000 iterations, and a convergence warning stops the run instead of being silently suppressed.

The response models are fitted on all training records for $\pi_1$ and $\rho$, and on $R_1=1$ records for $\pi_2$. Causal regressions are fitted on complete records with weights $1/(\widehat\pi_1\widehat\pi_2)$ for the sequential procedure and $1/\widehat\rho$ for the coarsened procedure. The sequential $Q_1$ regression uses complete records. Its $Q_0$ regression uses the augmented stage-two pseudo-outcome on training records with $R_1=1$. The coarsened projection regression uses complete records. These projection regressions use causal predictions fitted within the same outer training fold; there is no additional inner cross-fitting. The held-out fold is excluded from all these training operations.

Fitted treatment probabilities are truncated to $[0.01,0.99]$, and fitted response probabilities to $[0.01,1]$. Oracle substitutions are evaluated without truncation. In the stress design the true propensity can lie outside the fitted interval, so fixed truncation does not preserve the true propensity and cannot be assumed asymptotically harmless. In the bounded design the truncation thresholds lie outside the true propensity range. No truncation-sensitivity result is claimed without a corresponding run in the supplement.

\subsection*{Behavior across sample sizes}

'''+stress_table+f'''Under common validity, the ratio of mean estimated SE to empirical SD for the sequential estimator is {', '.join(f'{x:.2f}' for x in ratios)} at the four increasing sample sizes. These ratios assess the scale of interval miscalibration directly. The finite-sample precision comparison is given by the empirical SDs, whereas \\cref{{eq:variance-gap}} compares first-order efficiency bounds. They need not agree at these sample sizes. Outside common validity, the coarsened estimator's bias reflects its invalid identifying model as well as nuisance fitting.

\\FloatBarrier
\\subsection*{{Oracle substitutions and implementation checks}}

'''+layers+r'''
The causal-oracle option substitutes the true $e$ and both outcome regressions wherever $G$ is evaluated, including the training projections. The response-oracle option substitutes true $\pi_1,\pi_2,\rho$ in the causal training weights, the training pseudo-outcome for $Q_0$, and the final validation-fold augmentation. The all-oracle option also replaces the projection functions. True projection functions are projections of the true causal pseudo-outcome, so that option requires the causal-oracle option. Under sequential MAR alone, the coarsened all-oracle diagnostic uses $Q_0^{\cc}$, the complete-case conditional projection.

'''+f'''For the sequential procedure under common validity, the empirical SD is {f(base['sd'])} with all nuisances estimated, {f(oc['sd'])} with the causal functions known, and {f(rr['sd'])} with the response functions known. These comparisons identify sensitivity to the specified nuisance fitting choices; they do not show that every remaining discrepancy is caused by a single layer. Shared replicate seeds make the comparisons paired, and all replicate-level results are retained.

'''+r'''The supplied regression checks verify invariance to hidden covariate values, oracle training weights, and equality of the all-oracle calculation with \cref{eq:seq-eif}. They also distinguish variance from MSE. These checks target implementation correctness; an all-oracle match alone neither validates estimated nuisances nor establishes nominal coverage.

'''
    (PAPER/'appendix_d.tex').write_text(appendix.rstrip() + '\n')
    print('Wrote simulation.tex and appendix_d.tex from complete results.')


if __name__ == '__main__':
    main()
