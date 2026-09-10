"""Oracle simulation, with centered empirical variances and paired MC bootstrap."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from crossfit import ROOT,Z,summarise
from dgp import make_model


def variance_reduction(est_cc,est_seq):
    return 100*(1-np.var(est_seq,ddof=1)/np.var(est_cc,ddof=1))


def paired_bootstrap_reduction(est_cc,est_seq,B=10000,seed=20261009):
    rng = np.random.default_rng(seed)
    m = len(est_cc)
    ans = np.empty(B)
    for b in range(B):
        idx = rng.integers(0,m,m)
        ans[b] = variance_reduction(est_cc[idx],est_seq[idx])
    return np.percentile(ans,[2.5,97.5])


def run(scenario,n,reps,seed,model,p2_override=None,outfile=None):
    if n < 2 or reps < 2:
        raise ValueError('n and reps must both be at least two.')
    names = ('full','cc','cc_ccproj','seq')
    est,se = {k:np.empty(reps) for k in names},{k:np.empty(reps) for k in names}
    patterns = np.zeros(3)
    psi = model.proj.true_psi()
    fh = None
    if outfile:
        Path(outfile).parent.mkdir(parents=True,exist_ok=True)
        fh = open(outfile,'w',newline='')
        w = csv.writer(fh)
        w.writerow(['replicate','n','scenario','psi','estimator','estimate','se','ci_lo','ci_hi','covered'])
    try:
        for r in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([seed,r+1]))
            a,y,l1,l2 = model.sample(n,rng)
            p1 = model.pi1(a,y)
            f2 = (lambda l:model.pi2(a,y,l,scenario)) if p2_override is None else (lambda l:p2_override(a,y,l))
            p2 = f2(l1)
            r1,c2 = rng.binomial(1,p1),rng.binomial(1,p2)
            r2 = r1*c2
            posterior = model.posterior_l1(a,y)
            p20,p21 = f2(0),f2(1)
            p2bar = (1-posterior)*p20+posterior*p21
            rho = p1*p2bar
            G = model.ref.G(a,y,l1,l2)
            Q1,Q0 = model.proj.Q1(a,y,l1),model.proj.Q0(a,y)
            Qcc = ((1-posterior)*p20*model.proj.Q1(a,y,np.zeros_like(a))
                   +posterior*p21*model.proj.Q1(a,y,np.ones_like(a)))/p2bar
            H = dict(full=G,cc=Q0+r2/rho*(G-Q0),cc_ccproj=Qcc+r2/rho*(G-Qcc),
                     seq=Q0+r1/p1*(Q1-Q0)+r2/(p1*p2)*(G-Q1))
            for k in names:
                e,s = H[k].mean(),H[k].std(ddof=1)/np.sqrt(n)
                est[k][r],se[k][r] = e,s
                if fh:
                    lo,hi = e-Z*s,e+Z*s
                    w.writerow([r+1,n,scenario,repr(float(psi)),k,repr(float(e)),repr(float(s)),repr(float(lo)),repr(float(hi)),int(lo<=psi<=hi)])
            patterns += [r2.mean(),(r1-r2).mean(),(1-r1).mean()]
    finally:
        if fh:
            fh.close()
    result = dict(n=n,reps=reps,seed=seed,scenario=scenario,psi=psi,
                  seed_scheme='SeedSequence([seed, replicate_1based])',
                  pattern_complete=patterns[0]/reps,pattern_l1_only=patterns[1]/reps,
                  pattern_neither=patterns[2]/reps,**{k:summarise(est[k],se[k],psi) for k in names})
    return result,est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=2500)
    ap.add_argument('--reps',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=20260910)
    ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--out',default=str(ROOT/'results/oracle.json'))
    args = ap.parse_args()
    model = make_model('stress')
    output = {}
    for i,sc in enumerate(('ccmar','seqmar')):
        ans,est = run(sc,args.n,args.reps,args.seed+i,model,
                       outfile=Path(args.out).parent/'replicates'/f'oracle_{sc}.csv')
        output[sc] = ans
        if sc == 'ccmar':
            ci = paired_bootstrap_reduction(est['cc'],est['seq'],args.bootstrap,args.seed+99)
            output['efficiency_ccmar'] = dict(reduction_pct=variance_reduction(est['cc'],est['seq']),
               variance_ratio=np.var(est['cc'],ddof=1)/np.var(est['seq'],ddof=1),boot_lo=ci[0],boot_hi=ci[1],bootstrap_reps=args.bootstrap)
        print(f'oracle/{sc}: complete',flush=True)
    Path(args.out).write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
