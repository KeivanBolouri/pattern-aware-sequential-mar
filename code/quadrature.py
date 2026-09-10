"""Deterministic checks for the stress design and its variance-gap identity."""
import json
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre,roots_hermitenorm
from dgp import Refactorization,Projections,p_l1,pi1,pi2,l2_mean,SIGMA,y_logpdf


def evaluate(n_y=160,n_gh=80,ref_nodes=160,ref_grid=5201,proj_grid=1601):
    ref = Refactorization(n_y_nodes=ref_nodes,n_l2_grid=ref_grid)
    proj = Projections(ref,n_gh=n_gh,n_y_grid=proj_grid)
    ty,wy = roots_legendre(n_y)
    y,wy = (ty+1)/2,wy/2
    th,wh = roots_hermitenorm(n_gh)
    wh = wh/np.sqrt(2*np.pi)
    psi = proj.true_psi()
    result = dict(psi=psi,var_full=0.,var_seq_common=0.,var_cc_common=0.,gap=0.,
        var_seq_outside=0.,coarsened_bias_outside=0.,complete_common=0.,complete_outside=0.,
        neither=0.,inverse_e=0.,inverse_one_minus_e=0.,G_fourth_moment=0.)
    for a in (0,1):
        aa = np.full_like(y,a)
        q0 = proj.Q0(aa,y)
        for l1 in (0,1):
            p = p_l1(a,y)
            mass = .5*wy*np.exp(y_logpdf(y,a))*(p if l1 else 1-p)
            ll = np.full_like(y,l1)
            q1 = proj.Q1(aa,y,ll)
            l2 = l2_mean(a,y,l1)[:,None]+SIGMA*th
            g = ref.G(np.full(l2.size,a),np.repeat(y,n_gh),np.full(l2.size,l1),l2.ravel()).reshape(l2.shape)
            e = ref.e(np.full(l2.size,l1),l2.ravel()).reshape(l2.shape)
            p1 = pi1(a,y)
            p2c,p2s = pi2(a,y,l1,'ccmar'),pi2(a,y,l1,'seqmar')
            avgp2 = (1-p)*pi2(a,y,0,'seqmar')+p*pi2(a,y,1,'seqmar')
            residual2 = ((g-q1[:,None])**2)@wh
            result['var_full'] += mass@(((g-psi)**2)@wh)
            result['var_seq_common'] += mass@((q0-psi)**2+(q1-q0)**2/p1+residual2/(p1*p2c))
            result['var_cc_common'] += mass@((q0-psi)**2+(((g-q0[:,None])**2)@wh)/(p1*p2c))
            result['gap'] += mass@((1-p2c)/(p1*p2c)*(q1-q0)**2)
            result['var_seq_outside'] += mass@((q0-psi)**2+(q1-q0)**2/p1+residual2/(p1*p2s))
            result['coarsened_bias_outside'] += mass@(p2s/avgp2*(q1-q0))
            result['complete_common'] += mass@(p1*p2c)
            result['complete_outside'] += mass@(p1*p2s)
            result['neither'] += mass@(1-p1)
            result['inverse_e'] += mass@((1/e)@wh)
            result['inverse_one_minus_e'] += mass@((1/(1-e))@wh)
            result['G_fourth_moment'] += mass@((g**4)@wh)
    result['reduction_pct'] = 100*result['gap']/result['var_cc_common']
    result['identity_residual'] = result['var_cc_common']-result['var_seq_common']-result['gap']
    result['settings'] = dict(n_y=n_y,n_gh=n_gh,ref_nodes=ref_nodes,ref_grid=ref_grid,proj_grid=proj_grid)
    return result


def main():
    base = evaluate()
    fine = evaluate(320,160,320,10401,3201)
    delta = {k:abs(fine[k]-base[k]) for k in base if k != 'settings'}
    out = Path(__file__).resolve().parents[1]/'results/quadrature.json'
    out.write_text(json.dumps(dict(base=base,fine=fine,absolute_difference=delta),indent=2)+'\n')
    print(json.dumps(dict(fine=fine,absolute_difference=delta),indent=2))


if __name__ == '__main__':
    main()
