"""Run the pre-specified corrected simulation grid; resume only complete jobs."""
import os
# Prevent each process from starting additional numerical-library thread pools.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
from crossfit import ROOT,run_job


def configurations():
    jobs = []
    for design in ('stress','bounded'):
        sizes = [(2500,1000),(5000,1000),(10000,500),(20000,250)] if design=='stress' else [(2500,1000),(5000,1000)]
        for n,reps in sizes:
            for scenario in ('ccmar','seqmar'):
                jobs.append(dict(design=design,scenario=scenario,n=n,reps=reps,tag=f'{design}_n{n}'))
    for variant,options in [('oc',{'oracle_causal':True}),('or',{'oracle_response':True}),
                            ('alloracle',{'oracle_causal':True,'oracle_response':True,'oracle_q':True})]:
        for scenario in ('ccmar','seqmar'):
            jobs.append(dict(design='stress',scenario=scenario,n=2500,reps=500 if variant=='alloracle' else 1000,
                             tag=f'stress_{variant}2500',**options))
    for job in jobs:
        job.update(seed=20260910,folds=5,clip_e=.01,clip_pi=.01)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--pilot',type=int,default=0,help='Smoke run; separate pilot directory; never used in manuscript.')
    args = ap.parse_args()
    jobs = configurations()
    if args.pilot:
        jobs = [dict(j,reps=args.pilot,outdir=str(ROOT/'results/pilot')) for j in jobs if j['n']==2500]
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results'/'run_manifest.json').write_text(json.dumps(configurations(),indent=2)+'\n')
    if args.resume:
        jobs = [j for j in jobs if not (Path(j.get('outdir',ROOT/'results'))/f'crossfit_{j["tag"]}_{j["scenario"]}.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job,j):j for j in jobs}
        for future in as_completed(futures):
            print(f'WROTE {future.result()}',flush=True)


if __name__ == '__main__':
    main()
