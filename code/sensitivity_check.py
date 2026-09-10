"""Sensitivity scenarios with explicitly calibrated population response shares."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit,roots_legendre
from dgp import make_model,p_l1,pi1,y_logpdf
from oracle_sim import run,variance_reduction,paired_bootstrap_reduction
from crossfit import ROOT


def population_share(intercept,gamma):
    t,w = roots_legendre(160)
    y,w = (t+1)/2,w/2
    ans = 0.
    for a in (0,1):
        p = p_l1(a,y)
        avg = (1-p)*expit(intercept+.1*a+.1*y)+p*expit(intercept+.1*a+.1*y+gamma)
        ans += .5*np.sum(w*np.exp(y_logpdf(y,a))*pi1(a,y)*avg)
    return ans


def one(config):
    tag,gamma,target,reps = config
    intercept = brentq(lambda v:population_share(v,gamma)-target,-10,10,xtol=1e-13)
    # Distinct streams for distinct settings; each cc/seq pair shares its data.
    seed = 20261910+int(tag[-1])*100+(1000 if tag.startswith('ident') else 0)
    p2 = lambda a,y,l:expit(intercept+.1*a+.1*y+gamma*l)
    ans,est = run('ccmar' if gamma==0 else 'seqmar',2500,reps,seed,make_model('stress'),p2,
                   ROOT/'results/replicates'/f'sensitivity_{tag}.csv')
    ci = paired_bootstrap_reduction(est['cc'],est['seq'],10000,seed+99)
    ans.update(tag=tag,gamma=gamma,intercept=intercept,target_complete=target,
               population_complete=population_share(intercept,gamma),
               reduction_pct=variance_reduction(est['cc'],est['seq']),boot_lo=ci[0],boot_hi=ci[1])
    (ROOT/'results'/f'sensitivity_{tag}.json').write_text(json.dumps(ans,indent=2)+'\n')
    print(f'sensitivity/{tag}: complete',flush=True)
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps',type=int,default=5000)
    ap.add_argument('--workers',type=int,default=2)
    args = ap.parse_args()
    configs = [(f'eff{i}',0.,target,args.reps) for i,target in enumerate([.5117,.24,.1161])]
    configs += [(f'ident{i}',gamma,.24,args.reps) for i,gamma in enumerate([.75,1.5,2.25])]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        ans = list(pool.map(one,configs))
    (ROOT/'results/sensitivity.json').write_text(json.dumps(ans,indent=2)+'\n')


if __name__ == '__main__':
    main()
