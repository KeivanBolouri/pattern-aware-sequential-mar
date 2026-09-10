"""Five-fold sequential-MAR and coarsened estimators using observed inputs only.

Simulation-only oracle substitutions are explicit. Missing L1/L2 must be NaN;
the full-data benchmark is fitted by a separate function with a separate basis.
"""
import argparse
import csv
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer

from dgp import make_model

Z = 1.959963984540054
NAMES = ('full', 'cc', 'seq')
ROOT = Path(__file__).resolve().parents[1]


def _fit_basis(x, n_knots=6):
    if not np.isfinite(x).all():
        raise ValueError('Basis fitting received an unavailable covariate.')
    st = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False,
                           knots='quantile')
    return st.fit(np.asarray(x).reshape(-1, 1))


def X_z0(st, y, a):
    s = st.transform(y.reshape(-1, 1))
    a = a.reshape(-1, 1).astype(float)
    return np.hstack([s, a, a*s])


def X_z1(st, y, a, l1):
    s = st.transform(y.reshape(-1, 1))
    a, l = a.reshape(-1, 1).astype(float), l1.reshape(-1, 1)
    return np.hstack([s, a, l, a*l, a*s, l*s, a*l*s])


def X_v(st, l2, l1):
    s = st.transform(l2.reshape(-1, 1))
    l = l1.reshape(-1, 1)
    return np.hstack([s, l, l*s])


def fit_logit(X, y, w=None):
    # Convergence warnings are promoted to errors by the simulation runner.
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y, sample_weight=w)


def fit_ridge(X, y, w=None):
    return Ridge(alpha=1e-3).fit(X, y, sample_weight=w)


def _causal_fit(X, a, y, weights=None):
    m_e = fit_logit(X, a, weights)
    mu = {av: fit_ridge(X[a == av], y[a == av],
                        None if weights is None else weights[a == av]) for av in (0, 1)}
    return m_e, mu


def _G(fits, X, a, y, clip_e):
    e = np.clip(fits[0].predict_proba(X)[:, 1], clip_e, 1-clip_e)
    m0, m1 = fits[1][0].predict(X), fits[1][1].predict(X)
    return m1-m0 + a/e*(y-m1) - (1-a)/(1-e)*(y-m0)


def estimate_observed(a, y, l1, l2, r1, r2, fold_id, clip_e=.01,
                      clip_pi=.01, model=None, scenario='ccmar',
                      oracle_causal=False, oracle_response=False, oracle_q=False):
    """Return sequential and coarsened H values; never read hidden covariates.

    L1 is accessed only where R1=1, L2 only where R2=1. Outside common
    validity, oracle_q uses the complete-case conditional projection for the
    coarsened estimator (which remains inconsistent even with true G).
    """
    if oracle_q and not oracle_causal:
        raise ValueError('oracle_q requires oracle_causal: true Q projects true G.')
    if (oracle_causal or oracle_response or oracle_q) and model is None:
        raise ValueError('Oracle substitutions require a simulation model.')
    if np.any(r2 > r1):
        raise ValueError('The observation pattern must be monotone.')
    if not (0 < clip_e < .5 and 0 < clip_pi < 1):
        raise ValueError('Invalid probability truncation.')
    if not np.isfinite(l1[r1 == 1]).all() or not np.isfinite(l2[r2 == 1]).all():
        raise ValueError('An observed covariate is missing.')
    hseq, hcc = np.empty(len(a)), np.empty(len(a))
    for k in np.unique(fold_id):
        tr, te = fold_id != k, fold_id == k
        s1, cm = tr & (r1 == 1), tr & (r2 == 1)
        t1, tc = te & (r1 == 1), te & (r2 == 1)
        sty = _fit_basis(y[tr])
        # Knots are trained ONLY on observed complete-record L2.
        stv = _fit_basis(l2[cm]) if not oracle_causal else None
        z0 = lambda m: X_z0(sty, y[m], a[m])
        z1 = lambda m: X_z1(sty, y[m], a[m], l1[m])
        xv = lambda m: X_v(stv, l2[m], l1[m])
        if oracle_response:
            p1 = lambda m: model.pi1(a[m], y[m])
            p2 = lambda m: model.pi2(a[m], y[m], l1[m], scenario)
            rho = lambda m: model.rho(a[m], y[m], scenario)
        else:
            m1 = fit_logit(z0(tr), r1[tr])
            m2 = fit_logit(z1(s1), r2[s1])
            mr = fit_logit(z0(tr), r2[tr])
            p1 = lambda m: np.clip(m1.predict_proba(z0(m))[:, 1], clip_pi, 1.)
            p2 = lambda m: np.clip(m2.predict_proba(z1(m))[:, 1], clip_pi, 1.)
            rho = lambda m: np.clip(mr.predict_proba(z0(m))[:, 1], clip_pi, 1.)

        # The same response functions supply TRAINING and VALIDATION values.
        fits_seq = None if oracle_causal else _causal_fit(xv(cm), a[cm], y[cm], 1/(p1(cm)*p2(cm)))
        fits_cc = None if oracle_causal else _causal_fit(xv(cm), a[cm], y[cm], 1/rho(cm))
        def g(m, fits):
            return model.ref.G(a[m], y[m], l1[m], l2[m]) if oracle_causal else _G(fits, xv(m), a[m], y[m], clip_e)
        gc, gcc = g(cm, fits_seq), g(cm, fits_cc)
        if oracle_q:
            q1 = lambda m: model.proj.Q1(a[m], y[m], l1[m])
            q0 = lambda m: model.proj.Q0(a[m], y[m])
            qcc = lambda m: model.qcc(a[m], y[m], scenario)
        else:
            mq1 = fit_ridge(z1(cm), gc)
            q1 = lambda m: mq1.predict(z1(m))
            pseudo = q1(s1)
            complete_within_s1 = r2[s1] == 1
            pseudo[complete_within_s1] += (gc-q1(cm))/p2(cm)
            mq0 = fit_ridge(z0(s1), pseudo)
            mqcc = fit_ridge(z0(cm), gcc)
            q0 = lambda m: mq0.predict(z0(m))
            qcc = lambda m: mqcc.predict(z0(m))
        # Subset assignments avoid zero times NaN and enforce observability.
        hseq[te] = q0(te)
        hseq[t1] += (q1(t1)-q0(t1))/p1(t1)
        hseq[tc] += (g(tc, fits_seq)-q1(tc))/(p1(tc)*p2(tc))
        hcc[te] = qcc(te)
        hcc[tc] += (g(tc, fits_cc)-qcc(tc))/rho(tc)
    return {'seq': hseq, 'cc': hcc}


def estimate_full(a, y, l1, l2, fold_id, clip_e=.01, model=None, oracle_causal=False):
    if oracle_causal:
        return model.ref.G(a, y, l1, l2)
    out = np.empty(len(a))
    for k in np.unique(fold_id):
        tr, te = fold_id != k, fold_id == k
        st = _fit_basis(l2[tr])
        fit = _causal_fit(X_v(st, l2[tr], l1[tr]), a[tr], y[tr])
        out[te] = _G(fit, X_v(st, l2[te], l1[te]), a[te], y[te], clip_e)
    return out


def one_replicate(n, scenario, rng, folds=5, clip_e=.01, clip_pi=.01,
                  model=None, oracle_causal=False, oracle_response=False, oracle_q=False):
    model = make_model('stress') if model is None else model
    a, y, l1, l2 = model.sample(n, rng)
    r1, _, r2, _, _ = model.response(a, y, l1, scenario, rng)
    fold_id = np.empty(n, dtype=int)
    for k, chunk in enumerate(np.array_split(rng.permutation(n), folds)):
        fold_id[chunk] = k
    l1obs = np.where(r1 == 1, l1, np.nan)
    l2obs = np.where(r2 == 1, l2, np.nan)
    H = estimate_observed(a, y, l1obs, l2obs, r1, r2, fold_id, clip_e, clip_pi,
                          model, scenario, oracle_causal, oracle_response, oracle_q)
    H['full'] = estimate_full(a, y, l1, l2, fold_id, clip_e, model, oracle_causal)
    return {k: (H[k].mean(), H[k].std(ddof=1)/np.sqrt(n)) for k in NAMES}


def summarise(est, se, psi):
    reps = len(est)
    sd = est.std(ddof=1)
    coverage = np.mean(np.abs(est-psi) <= Z*se)
    return dict(bias=float(est.mean()-psi), mcse_bias=float(sd/np.sqrt(reps)),
                sd=float(sd), rmse=float(np.sqrt(np.mean((est-psi)**2))),
                coverage=float(coverage), mcse_cov=float(np.sqrt(coverage*(1-coverage)/reps)),
                mean_se=float(se.mean()), mad=float(np.median(np.abs(est-psi))),
                iqr_sd=float((np.percentile(est,75)-np.percentile(est,25))/1.349))


def run_job(config):
    """One design/scenario/configuration; atomic JSON marks a completed run."""
    config = dict(config)
    design, scenario = config['design'], config['scenario']
    model = make_model(design)
    n, reps, seed = config['n'], config['reps'], config.get('seed',20260910)
    folds = config.get('folds',5)
    if reps < 2 or not 2 <= folds <= n:
        raise ValueError('At least two replicates and 2 <= folds <= n are required.')
    psi = model.proj.true_psi()
    outdir = Path(config.get('outdir',ROOT/'results'))
    (outdir/'replicates').mkdir(parents=True,exist_ok=True)
    tag = config['tag']
    dest = outdir/'replicates'/f'crossfit_{tag}_{scenario}.csv'
    est = {k:np.empty(reps) for k in NAMES}
    se = {k:np.empty(reps) for k in NAMES}
    t0 = time.time()
    options = {k:config.get(k,False) for k in ('oracle_causal','oracle_response','oracle_q')}
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        with dest.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['replicate','n','folds','design','scenario','psi','estimator','estimate','se','ci_lo','ci_hi','covered'])
            for r in range(reps):
                # Addressable, order-independent streams; flags deliberately do not alter seed.
                ss = np.random.SeedSequence([seed, {'stress':0,'bounded':1}[design],
                                             {'ccmar':0,'seqmar':1}[scenario],n,r+1])
                ans = one_replicate(n,scenario,np.random.default_rng(ss),folds,
                                     config.get('clip_e',.01),config.get('clip_pi',.01),model,**options)
                for k in NAMES:
                    e,s = ans[k]
                    if not np.isfinite([e,s]).all():
                        raise FloatingPointError(f'Nonfinite output: {tag}/{scenario}/{r+1}/{k}')
                    est[k][r],se[k][r] = e,s
                    lo,hi = e-Z*s,e+Z*s
                    w.writerow([r+1,n,folds,design,scenario,repr(float(psi)),k,repr(float(e)),repr(float(s)),repr(float(lo)),repr(float(hi)),int(lo <= psi <= hi)])
                if (r+1)%100 == 0:
                    fh.flush()
                    print(f'{tag}/{scenario}: {r+1}/{reps} ({time.time()-t0:.0f}s)',flush=True)
    result = dict(config=config, psi=psi, seed_scheme='SeedSequence([seed, design_id, scenario_id, n, replicate_1based])',
                  warnings_policy='ConvergenceWarning is fatal; no failed replicates omitted',
                  environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                  elapsed_seconds=time.time()-t0, **{k:summarise(est[k],se[k],psi) for k in NAMES})
    target = outdir/f'crossfit_{tag}_{scenario}.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n')
    tmp.replace(target)
    print(f'COMPLETE {tag}/{scenario}: sequential coverage={result["seq"]["coverage"]:.3f}',flush=True)
    return str(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=2500)
    ap.add_argument('--reps',type=int,default=1000)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--seed',type=int,default=20260910)
    ap.add_argument('--design',choices=['stress','bounded'],default='stress')
    ap.add_argument('--scenarios',default='ccmar,seqmar')
    ap.add_argument('--clip-e',type=float,default=.01)
    ap.add_argument('--clip-pi',type=float,default=.01)
    ap.add_argument('--tag',default='main')
    ap.add_argument('--outdir',default=str(ROOT/'results'))
    for flag in ['causal','response','q']:
        ap.add_argument('--oracle-'+flag,action='store_true')
    args = vars(ap.parse_args())
    for scenario in args.pop('scenarios').split(','):
        run_job(dict(args,scenario=scenario))


if __name__ == '__main__':
    main()
