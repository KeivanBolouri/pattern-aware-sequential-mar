"""Regression checks for observability, oracle substitutions, and variance units."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import unittest
from unittest.mock import patch
import numpy as np
import crossfit
from dgp import make_model
from oracle_sim import variance_reduction,paired_bootstrap_reduction


class EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_model('stress')
        rng = np.random.default_rng(101)
        cls.a,cls.y,cls.l1,cls.l2 = cls.model.sample(1600,rng)
        cls.r1,_,cls.r2,_,_ = cls.model.response(cls.a,cls.y,cls.l1,'seqmar',rng)
        cls.fold = rng.permutation(1600)%5

    def evaluate(self,l1,l2,**kwargs):
        return crossfit.estimate_observed(self.a,self.y,l1,l2,self.r1,self.r2,self.fold,
            model=self.model,scenario='seqmar',**kwargs)

    def test_unavailable_values_do_not_change_results(self):
        baseline = self.evaluate(self.l1,self.l2)
        l1 = np.where(self.r1,self.l1,np.nan)
        l2 = np.where(self.r2,self.l2,np.nan)
        masked = self.evaluate(l1,l2)
        perturbed = self.evaluate(np.where(self.r1,self.l1,777),np.where(self.r2,self.l2,1e8))
        for k in baseline:
            np.testing.assert_array_equal(baseline[k],masked[k])
            np.testing.assert_array_equal(baseline[k],perturbed[k])

    def test_oracle_response_controls_training_weights_and_no_response_fit(self):
        captured = []
        original = crossfit._causal_fit
        def record(X,a,y,weights=None):
            captured.append(weights.copy())
            return original(X,a,y,weights)
        with patch.object(crossfit,'_causal_fit',side_effect=record):
            self.evaluate(self.l1,self.l2,oracle_response=True)
        for j,k in enumerate(np.unique(self.fold)):
            cm = (self.fold != k)&(self.r2 == 1)
            np.testing.assert_allclose(captured[2*j],1/(self.model.pi1(self.a[cm],self.y[cm])*
                self.model.pi2(self.a[cm],self.y[cm],self.l1[cm],'seqmar')),rtol=0,atol=0)
            np.testing.assert_allclose(captured[2*j+1],1/self.model.rho(self.a[cm],self.y[cm],'seqmar'),rtol=0,atol=0)
        # No logistic fit should occur if the causal and response functions are known.
        with patch.object(crossfit,'fit_logit',side_effect=AssertionError('unexpected logistic fit')):
            self.evaluate(self.l1,self.l2,oracle_causal=True,oracle_response=True)

    def test_all_oracle_matches_direct_equation_and_coarsened_projection(self):
        h = self.evaluate(self.l1,self.l2,oracle_causal=True,oracle_response=True,oracle_q=True)
        m,a,y,l1,l2 = self.model,self.a,self.y,self.l1,self.l2
        g,q1,q0 = m.ref.G(a,y,l1,l2),m.proj.Q1(a,y,l1),m.proj.Q0(a,y)
        qcc = m.qcc(a,y,'seqmar')
        p1,p2 = m.pi1(a,y),m.pi2(a,y,l1,'seqmar')
        np.testing.assert_allclose(h['seq'],q0+self.r1/p1*(q1-q0)+self.r2/(p1*p2)*(g-q1),atol=1e-14)
        np.testing.assert_allclose(h['cc'],qcc+self.r2/m.rho(a,y,'seqmar')*(g-qcc),atol=1e-14)

    def test_variance_reduction_is_translation_invariant(self):
        a = np.array([1.,2.,4.,7.,10.])
        b = np.array([1.,2.,2.,4.,6.])
        self.assertAlmostEqual(variance_reduction(a,b),variance_reduction(a+100,b-30))
        np.testing.assert_allclose(paired_bootstrap_reduction(a,b,500,44),
                                   paired_bootstrap_reduction(a+100,b-30,500,44),atol=1e-12)

    def test_bounded_design_and_target(self):
        model = make_model('bounded')
        l2 = np.linspace(-1,1,101)
        for l1 in (0,1):
            e = model.ref.e(l1,l2)
            self.assertTrue(np.all((e>=.2689)&(e<=.7311)))
            np.testing.assert_allclose(model.ref.mu(1,l1,l2)-model.ref.mu(0,l1,l2),.15)
        self.assertEqual(model.proj.true_psi(),.15)
        ans = crossfit.one_replicate(1600,'seqmar',np.random.default_rng(12),model=model)
        self.assertTrue(all(np.isfinite(v).all() for v in ans.values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
