"""Simulation laws and oracle nuisance functions for the revised manuscript.

The stress law is specified retrospectively:
    A ~ Bernoulli(.5), Y|A=0 ~ Beta(2,4), Y|A=1 ~ Beta(4,2),
    L1|A,Y ~ Bernoulli(expit(-.6+.5*A+.25*Y+.1*A*Y)),
    L2|A,Y,L1 ~ Normal(A+Y+2.5*L1*Y, 1.25**2).
Its causal nuisances and projections are evaluated by deterministic quadrature.

The additional bounded law is specified prospectively at the end of the file.
The simulator retains latent covariates; the incomplete-data estimator receives
only their observed versions. No real-world data are required.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import expit, roots_hermitenorm, roots_legendre

SIGMA = 1.25

# ---------------------------------------------------------------- basic pieces


def y_logpdf(y: np.ndarray, a: int) -> np.ndarray:
    """log Beta density: Beta(2,4) if a==0 else Beta(4,2).  B(2,4)=B(4,2)=1/20."""
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore"):
        if a == 0:
            return np.log(20.0) + np.log(y) + 3.0 * np.log1p(-y)
        return np.log(20.0) + 3.0 * np.log(y) + np.log1p(-y)


def p_l1(a, y):
    """P(L1=1 | A=a, Y=y)."""
    return expit(-0.6 + 0.5 * a + 0.25 * y + 0.1 * a * y)


def l2_mean(a, y, l1):
    return a + y + 2.5 * l1 * y


# ------------------------------------------------- e(V) and mu_a(V) by quadrature


class Refactorization:
    """Numerical refactorisation of the retrospective law into e(V) and mu_a(V).

    e(l1,l2)    = P(A=1 | L1=l1, L2=l2)
    mu_a(l1,l2) = E(Y | A=a, L1=l1, L2=l2)

    Each is a one-dimensional function of l2 for each l1 in {0,1}; they are
    evaluated on a dense grid by Gauss-Legendre quadrature in y and then
    interpolated with cubic splines.  All accumulation is done in log space so
    that far tails of the normal kernel do not underflow.
    """

    def __init__(self, n_y_nodes=160, l2_lo=-10.0, l2_hi=16.0, n_l2_grid=5201):
        t, w = roots_legendre(n_y_nodes)
        self.yk = 0.5 * (t + 1.0)          # nodes on (0,1)
        self.wk = 0.5 * w                  # weights on (0,1)
        self.l2_grid = np.linspace(l2_lo, l2_hi, n_l2_grid)
        self.l2_lo, self.l2_hi = l2_lo, l2_hi
        self._build()

    def _log_kernel(self, a, l1, l2):
        """log[ f_Y(y_k|a) p(l1|a,y_k) N(l2; m, sigma^2) ] on grid x nodes."""
        yk = self.yk
        m = l2_mean(a, yk, l1)                                   # (K,)
        pl1 = p_l1(a, yk)
        log_pl1 = np.log(pl1 if l1 == 1 else 1.0 - pl1)          # (K,)
        z = (l2[:, None] - m[None, :]) / SIGMA                   # (G,K)
        return (
            y_logpdf(yk, a)[None, :]
            + log_pl1[None, :]
            - 0.5 * z**2
            - np.log(SIGMA * np.sqrt(2.0 * np.pi))
        )

    def _build(self):
        g = self.l2_grid
        self.e_spl, self.mu_spl = {}, {}
        for l1 in (0, 1):
            logk = {a: self._log_kernel(a, l1, g) for a in (0, 1)}
            shift = np.maximum(logk[0].max(axis=1), logk[1].max(axis=1))
            den, num = {}, {}
            for a in (0, 1):
                k = np.exp(logk[a] - shift[:, None])
                den[a] = k @ self.wk
                num[a] = (k * self.yk[None, :]) @ self.wk
            # the common 0.5 prior on A cancels in the ratio
            e = den[1] / (den[0] + den[1])
            self.e_spl[l1] = CubicSpline(g, e, extrapolate=False)
            for a in (0, 1):
                self.mu_spl[(a, l1)] = CubicSpline(g, num[a] / den[a], extrapolate=False)

    # -- vectorised evaluation over arrays of (l1, l2) -------------------------

    def _eval(self, spl_by_l1, l1, l2):
        l2c = np.clip(l2, self.l2_lo, self.l2_hi)
        out = np.empty_like(l2c, dtype=float)
        for v in (0, 1):
            m = l1 == v
            if m.any():
                out[m] = spl_by_l1[v](l2c[m])
        return out

    def e(self, l1, l2):
        return self._eval(self.e_spl, np.asarray(l1), np.asarray(l2, dtype=float))

    def mu(self, a, l1, l2):
        return self._eval(
            {v: self.mu_spl[(a, v)] for v in (0, 1)},
            np.asarray(l1),
            np.asarray(l2, dtype=float),
        )

    def G(self, a, y, l1, l2):
        """Full-data influence function plus the ATE (causal pseudo-outcome)."""
        a = np.asarray(a, dtype=float)
        y = np.asarray(y, dtype=float)
        ev = self.e(l1, l2)
        m1 = self.mu(1, l1, l2)
        m0 = self.mu(0, l1, l2)
        return (
            m1 - m0
            + a / ev * (y - m1)
            - (1.0 - a) / (1.0 - ev) * (y - m0)
        )


# ------------------------------------------------------ Q1, Q0 and the true ATE


class Projections:
    """Oracle iterated projections Q1(Z1)=E(G|Z1) and Q0(Z0)=E(G|Z0).

    Q1 is a smooth function of y for each (a,l1); Q0 a smooth function of y for
    each a.  Both are tabulated on a dense y-grid using Gauss-Hermite quadrature
    over L2 and then splined.
    """

    def __init__(self, refac: Refactorization, n_gh=80, n_y_grid=1601):
        self.ref = refac
        t, w = roots_hermitenorm(n_gh)     # weights integrate against exp(-x^2/2)
        self.gh_nodes = t
        self.gh_w = w / np.sqrt(2.0 * np.pi)
        self.ygrid = np.linspace(1e-6, 1.0 - 1e-6, n_y_grid)
        self._build()

    def _q1_on_grid(self, a, l1):
        y = self.ygrid
        m = l2_mean(a, y, l1)
        l2 = m[:, None] + SIGMA * self.gh_nodes[None, :]          # (G,J)
        flat = l2.ravel()
        aa = np.full(flat.shape, float(a))
        yy = np.repeat(y, l2.shape[1])
        ll1 = np.full(flat.shape, l1)
        g = self.ref.G(aa, yy, ll1, flat).reshape(l2.shape)
        return g @ self.gh_w

    def _build(self):
        self.q1_spl, self.q0_spl = {}, {}
        for a in (0, 1):
            q1 = {l1: self._q1_on_grid(a, l1) for l1 in (0, 1)}
            for l1 in (0, 1):
                self.q1_spl[(a, l1)] = CubicSpline(self.ygrid, q1[l1], extrapolate=True)
            p1 = p_l1(a, self.ygrid)
            self.q0_spl[a] = CubicSpline(
                self.ygrid, (1.0 - p1) * q1[0] + p1 * q1[1], extrapolate=True
            )

    def Q1(self, a, y, l1):
        a, y, l1 = np.asarray(a), np.asarray(y, dtype=float), np.asarray(l1)
        out = np.empty_like(y)
        for av in (0, 1):
            for lv in (0, 1):
                m = (a == av) & (l1 == lv)
                if m.any():
                    out[m] = self.q1_spl[(av, lv)](y[m])
        return out

    def Q0(self, a, y):
        a, y = np.asarray(a), np.asarray(y, dtype=float)
        out = np.empty_like(y)
        for av in (0, 1):
            m = a == av
            if m.any():
                out[m] = self.q0_spl[av](y[m])
        return out

    def true_psi(self, n_nodes=400):
        """psi = E[G] = E[Q0(A,Y)], a one-dimensional integral for each a."""
        t, w = roots_legendre(n_nodes)
        y = 0.5 * (t + 1.0)
        wy = 0.5 * w
        tot = 0.0
        for a in (0, 1):
            fy = np.exp(y_logpdf(y, a))
            tot += 0.5 * np.sum(wy * fy * self.q0_spl[a](y))
        return tot


# -------------------------------------------------------- response mechanisms


def pi1(a, y):
    return expit(2.0 + 0.1 * a + 0.1 * y)


def pi2(a, y, l1, scenario):
    """Second-stage response probability P(C2=1 | R1=1, Z1)."""
    if scenario == "ccmar":
        return expit(-1.1 + 0.1 * a + 0.1 * y) * np.ones_like(np.asarray(y, dtype=float))
    if scenario == "seqmar":
        return expit(-1.9 + 0.1 * a + 0.1 * y + 2.2 * l1)
    raise ValueError(scenario)


def rho_z0(a, y, proj: Projections, scenario):
    """P(S=1 | Z0) = pi1 * E[pi2(Z1) | Z0] -- what a complete-case analyst can model."""
    p1 = p_l1(a, y)
    p2_0 = pi2(a, y, 0, scenario)
    p2_1 = pi2(a, y, 1, scenario)
    return pi1(a, y) * ((1.0 - p1) * p2_0 + p1 * p2_1)


def q0_complete_case(a, y, proj: Projections, scenario):
    """E(G | Z0, S=1) = E[pi2 Q1 | Z0] / E[pi2 | Z0] -- the projection a
    complete-case analyst actually estimates from observed data."""
    p1 = p_l1(a, y)
    p2_0, p2_1 = pi2(a, y, 0, scenario), pi2(a, y, 1, scenario)
    q1_0 = proj.Q1(a, y, np.zeros_like(y, dtype=int))
    q1_1 = proj.Q1(a, y, np.ones_like(y, dtype=int))
    num = (1.0 - p1) * p2_0 * q1_0 + p1 * p2_1 * q1_1
    den = (1.0 - p1) * p2_0 + p1 * p2_1
    return num / den


# ------------------------------------------------------------------- sampling


def draw_full_data(n, rng):
    a = rng.binomial(1, 0.5, n)
    y = np.where(a == 1, rng.beta(4.0, 2.0, n), rng.beta(2.0, 4.0, n))
    l1 = rng.binomial(1, p_l1(a, y))
    l2 = rng.normal(l2_mean(a, y, l1), SIGMA)
    return a, y, l1, l2


def draw_response(a, y, l1, scenario, rng):
    p1 = pi1(a, y)
    r1 = rng.binomial(1, p1)
    p2 = pi2(a, y, l1, scenario)
    c2 = rng.binomial(1, p2)
    r2 = r1 * c2
    return r1, c2, r2, p1, p2


class BoundedRefactorization:
    """Prospective beta-outcome law with uniform treatment positivity."""
    @staticmethod
    def e(l1, l2):
        return expit(-.3 + .6*np.asarray(l1) + .7*np.asarray(l2))

    @staticmethod
    def mu(a, l1, l2):
        return .25 + .15*a + .10*np.asarray(l1) + .05*np.asarray(l2)

    G = Refactorization.G


class BoundedProjections(Projections):
    """Bayes projections by one-dimensional quadrature in bounded L2."""
    def __init__(self, refac, n_nodes=120, n_y_grid=2401):
        from scipy.special import betaln
        self.ref = refac
        self.ygrid = np.linspace(1e-6,1-1e-6,n_y_grid)
        l2, w = roots_legendre(n_nodes)
        self.q1_spl, self.q0_spl, self.pl1_spl = {}, {}, {}
        for a in (0,1):
            mass, numer = {}, {}
            for l1 in (0,1):
                e = refac.e(l1,l2)
                mu = refac.mu(a,l1,l2)
                alpha, beta = 8*mu, 8*(1-mu)
                y = self.ygrid[:,None]
                logf = (alpha-1)*np.log(y)+(beta-1)*np.log1p(-y)-betaln(alpha,beta)
                kernel = np.exp(logf)*(e if a == 1 else 1-e)*w/2
                gg = refac.G(a,y,l1,l2)
                mass[l1] = kernel.sum(axis=1)
                numer[l1] = (kernel*gg).sum(axis=1)
                self.q1_spl[(a,l1)] = CubicSpline(self.ygrid,numer[l1]/mass[l1])
            self.q0_spl[a] = CubicSpline(self.ygrid,(numer[0]+numer[1])/(mass[0]+mass[1]))
            self.pl1_spl[a] = CubicSpline(self.ygrid,mass[1]/(mass[0]+mass[1]))

    def true_psi(self, n_nodes=400):
        # mu1-mu0 is identically .15, so no numerical integration is needed.
        return .15

    def p_l1(self,a,y):
        a,y = np.asarray(a),np.asarray(y,dtype=float)
        out = np.empty_like(y)
        for av in (0,1):
            m = a == av
            out[m] = self.pl1_spl[av](y[m])
        return out


class SimulationModel:
    def __init__(self, design):
        self.design = design
        if design == 'stress':
            self.ref = Refactorization()
            self.proj = Projections(self.ref)
        elif design == 'bounded':
            self.ref = BoundedRefactorization()
            self.proj = BoundedProjections(self.ref)
        else:
            raise ValueError(design)

    def sample(self,n,rng):
        if self.design == 'stress':
            return draw_full_data(n,rng)
        l1 = rng.binomial(1,.5,n)
        l2 = rng.uniform(-1,1,n)
        a = rng.binomial(1,self.ref.e(l1,l2))
        mu = self.ref.mu(a,l1,l2)
        y = rng.beta(8*mu,8*(1-mu))
        return a,y,l1,l2

    pi1 = staticmethod(pi1)
    pi2 = staticmethod(pi2)
    response = staticmethod(draw_response)

    def posterior_l1(self,a,y):
        return p_l1(a,y) if self.design == 'stress' else self.proj.p_l1(a,y)

    def rho(self,a,y,scenario):
        p = self.posterior_l1(a,y)
        return pi1(a,y)*((1-p)*pi2(a,y,0,scenario)+p*pi2(a,y,1,scenario))

    def qcc(self,a,y,scenario):
        p = self.posterior_l1(a,y)
        p20,p21 = pi2(a,y,0,scenario),pi2(a,y,1,scenario)
        q10 = self.proj.Q1(a,y,np.zeros_like(a))
        q11 = self.proj.Q1(a,y,np.ones_like(a))
        return ((1-p)*p20*q10+p*p21*q11)/((1-p)*p20+p*p21)


def make_model(design='stress'):
    return SimulationModel(design)
