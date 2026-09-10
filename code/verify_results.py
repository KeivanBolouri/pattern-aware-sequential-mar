"""Verify all manuscript summaries against every archived replicate."""
import csv
import json
from pathlib import Path
import numpy as np
from crossfit import ROOT,Z,summarise
from run_studies import configurations
from oracle_sim import variance_reduction


def read_rows(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def check_rows(rows,summary,reps,n,psi,names):
    if len(rows) != reps*len(names):
        raise AssertionError(f'Wrong row count: {len(rows)} != {reps*len(names)}')
    for k in names:
        group=[r for r in rows if r['estimator']==k]
        assert [int(r['replicate']) for r in group] == list(range(1,reps+1)), 'Missing/repeated/out-of-order replicate'
        assert all(int(r['n'])==n for r in group)
        estimates=np.array([float(r['estimate']) for r in group])
        se=np.array([float(r['se']) for r in group])
        lo=np.array([float(r['ci_lo']) for r in group])
        hi=np.array([float(r['ci_hi']) for r in group])
        target=np.array([float(r['psi']) for r in group])
        covered=np.array([int(r['covered']) for r in group])
        assert np.isfinite([estimates,se,lo,hi,target]).all()
        assert (se>0).all()
        np.testing.assert_allclose(target,psi,rtol=0,atol=1e-14)
        np.testing.assert_allclose(lo,estimates-Z*se,rtol=0,atol=1e-14)
        np.testing.assert_allclose(hi,estimates+Z*se,rtol=0,atol=1e-14)
        np.testing.assert_array_equal(covered,(lo<=psi)&(psi<=hi))
        actual=summarise(estimates,se,psi)
        for field,value in actual.items():
            np.testing.assert_allclose(value,summary[k][field],rtol=1e-12,atol=1e-14,err_msg=f'{k}/{field}')
    return len(rows)


def main():
    count,samples=0,0
    results=ROOT/'results'
    manifest=json.loads((results/'run_manifest.json').read_text())
    assert manifest == configurations(), 'Manifest differs from run grid'
    for config in manifest:
        tag,sc=config['tag'],config['scenario']
        result=json.loads((results/f'crossfit_{tag}_{sc}.json').read_text())
        assert result['config']==config, 'Wrong configuration metadata'
        rows=read_rows(results/'replicates'/f'crossfit_{tag}_{sc}.csv')
        assert all(r['design']==config['design'] and r['scenario']==sc and int(r['folds'])==config['folds'] for r in rows)
        count+=check_rows(rows,result,config['reps'],config['n'],result['psi'],('full','cc','seq'))
        samples+=config['reps']
    oracle=json.loads((results/'oracle.json').read_text())
    for sc in ('ccmar','seqmar'):
        result=oracle[sc]
        rows=read_rows(results/'replicates'/f'oracle_{sc}.csv')
        count+=check_rows(rows,result,result['reps'],result['n'],result['psi'],('full','cc','cc_ccproj','seq'))
        samples+=result['reps']
        if sc=='ccmar':
            vals={k:np.array([float(r['estimate']) for r in rows if r['estimator']==k]) for k in ('cc','seq')}
            np.testing.assert_allclose(variance_reduction(vals['cc'],vals['seq']),oracle['efficiency_ccmar']['reduction_pct'],atol=1e-12)
    sensitivity=json.loads((results/'sensitivity.json').read_text())
    assert [d['tag'] for d in sensitivity]==['eff0','eff1','eff2','ident0','ident1','ident2']
    for result in sensitivity:
        rows=read_rows(results/'replicates'/f'sensitivity_{result["tag"]}.csv')
        count+=check_rows(rows,result,result['reps'],result['n'],result['psi'],('full','cc','cc_ccproj','seq'))
        samples+=result['reps']
        vals={k:np.array([float(r['estimate']) for r in rows if r['estimator']==k]) for k in ('cc','seq')}
        np.testing.assert_allclose(variance_reduction(vals['cc'],vals['seq']),result['reduction_pct'],atol=1e-12)
        assert abs(result['population_complete']-result['target_complete'])<1e-12
    q=json.loads((results/'quadrature.json').read_text())
    assert abs(q['fine']['identity_residual'])<1e-12
    assert max(q['absolute_difference'][k] for k in ('var_full','var_cc_common','var_seq_common','gap'))<1e-8
    validation=dict(status='passed',estimated_nuisance_configurations=len(manifest),oracle_scenarios=2,
                    sensitivity_configurations=len(sensitivity),simulated_samples=samples,replicate_estimator_rows=count,
                    checks=['complete configuration grid','no missing or duplicate replicates','finite outputs',
                            'interval endpoints and coverage','all reported summary fields','centered variance reductions',
                            'population calibration','two-resolution variance identity'])
    (results/'verification.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))


if __name__ == '__main__':
    main()
