"""External check on Tables 3 and 4 (sensitivity analyses).

The manuscript reports the resulting pattern proportions but not the stage-two
intercepts, so intercepts are solved for here to match the reported complete-case
shares.  The efficiency column is then computed from the exact variance identity
by quadrature rather than by simulation.
"""

import json

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, roots_hermitenorm, roots_legendre

from dgp import (Projections, Refactorization, SIGMA, draw_full_data, l2_mean,
                 p_l1, pi1, y_logpdf)

Z = 1.959963984540054

ref = Refactorization()
proj = Projections(ref)
PSI = proj.true_psi()

t, w = roots_legendre(400)
YK, WK = 0.5 * (t + 1), 0.5 * w
th, wh = roots_hermitenorm(240)
GH_T, GH_W = th, wh / np.sqrt(2 * np.pi)


def complete_share(c, gamma):
    """E[pi1 * E(pi2|Z0)] with pi2 = expit(c + .1A + .1Y + gamma*L1)."""
    tot = 0.0
    for a in (0, 1):
        fy = np.exp(y_logpdf(YK, a))
        p1 = p_l1(a, YK)
        p2 = (1 - p1) * expit(c + .1 * a + .1 * YK) + p1 * expit(c + .1 * a + .1 * YK + gamma)
        tot += 0.5 * np.sum(WK * fy * pi1(a, YK) * p2)
    return tot


def l1_only_share(c, gamma):
    tot = 0.0
    for a in (0, 1):
        fy = np.exp(y_logpdf(YK, a))
        p1 = p_l1(a, YK)
        p2 = (1 - p1) * expit(c + .1 * a + .1 * YK) + p1 * expit(c + .1 * a + .1 * YK + gamma)
        tot += 0.5 * np.sum(WK * fy * pi1(a, YK) * (1 - p2))
    return tot


def solve_intercept(target, gamma=0.0):
    return brentq(lambda c: complete_share(c, gamma) - target, -12.0, 12.0, xtol=1e-12)


def exact_reduction(c):
    """Theoretical % variance reduction under Assumption 3 (gamma = 0)."""
    A = B = C = gap = 0.0
    for a in (0, 1):
        fy = np.exp(y_logpdf(YK, a))
        p1 = p_l1(a, YK)
        q0 = proj.Q0(np.full(YK.shape, a), YK)
        A += 0.5 * np.sum(WK * fy * (q0 - PSI) ** 2)
        pv1 = pi1(a, YK)
        pv2 = expit(c + .1 * a + .1 * YK)
        for l1 in (0, 1):
            pl = p1 if l1 == 1 else 1 - p1
            q1 = proj.Q1(np.full(YK.shape, a), YK, np.full(YK.shape, l1))
            d2 = (q1 - q0) ** 2
            B += 0.5 * np.sum(WK * fy * pl * d2 / pv1)
            gap += 0.5 * np.sum(WK * fy * pl * (1 - pv2) / (pv1 * pv2) * d2)
            m = l2_mean(a, YK, l1)
            L2 = m[:, None] + SIGMA * GH_T[None, :]
            g = ref.G(np.full(L2.shape, float(a)).ravel(),
                      np.repeat(YK, L2.shape[1]),
                      np.full(L2.shape, l1).ravel(), L2.ravel()).reshape(L2.shape)
            C += 0.5 * np.sum(WK * fy * pl * (((g - q1[:, None]) ** 2) @ GH_W) / (pv1 * pv2))
    vseq, vcc = A + B + C, A + B + gap + C
    return 100 * (1 - vseq / vcc), vseq, vcc


def identification_run(c, gamma, n=2500, reps=5000, seed=7):
    rng = np.random.default_rng(seed)
    ecc, eseq = np.empty(reps), np.empty(reps)
    scc, sseq = np.empty(reps), np.empty(reps)
    for r in range(reps):
        a, y, l1, l2 = draw_full_data(n, rng)
        p1v = pi1(a, y)
        p2v = expit(c + .1 * a + .1 * y + gamma * l1)
        r1 = rng.binomial(1, p1v)
        c2 = rng.binomial(1, p2v)
        r2 = r1 * c2
        G = ref.G(a, y, l1, l2)
        Q1 = proj.Q1(a, y, l1)
        Q0 = proj.Q0(a, y)
        pl = p_l1(a, y)
        rho = p1v * ((1 - pl) * expit(c + .1 * a + .1 * y)
                     + pl * expit(c + .1 * a + .1 * y + gamma))
        Hcc = Q0 + r2 / rho * (G - Q0)
        Hsq = Q0 + r1 / p1v * (Q1 - Q0) + r2 / (p1v * p2v) * (G - Q1)
        ecc[r], eseq[r] = Hcc.mean(), Hsq.mean()
        scc[r] = Hcc.std(ddof=1) / np.sqrt(n)
        sseq[r] = Hsq.std(ddof=1) / np.sqrt(n)
    return {"coarsened_bias": ecc.mean() - PSI,
            "coarsened_coverage": float(np.mean(np.abs(ecc - PSI) <= Z * scc)),
            "sequential_bias": eseq.mean() - PSI,
            "sequential_coverage": float(np.mean(np.abs(eseq - PSI) <= Z * sseq))}


if __name__ == "__main__":
    out = {"psi": PSI, "efficiency": [], "identification": []}
    print("--- Table 3 check: efficiency under common validity (gamma = 0) ---")
    for label, target in [("Low", 0.5117), ("Moderate", 0.2400), ("High", 0.1161)]:
        c = solve_intercept(target)
        red, vseq, vcc = exact_reduction(c)
        row = {"label": label, "intercept": c, "complete": complete_share(c, 0.0),
               "l1_only": l1_only_share(c, 0.0), "exact_reduction_pct": red,
               "var_seq": vseq, "var_cc": vcc}
        out["efficiency"].append(row)
        print(f"{label:9s} c={c:+.4f}  complete={row['complete']*100:5.2f}%  "
              f"L1-only={row['l1_only']*100:5.2f}%  exact reduction={red:.2f}%")

    print("\n--- Table 4 check: identification as gamma grows, complete held at 24% ---")
    for gamma in (0.75, 1.50, 2.25):
        c = solve_intercept(0.24, gamma)
        r = identification_run(c, gamma, reps=5000, seed=int(1000 * gamma))
        r.update({"gamma": gamma, "intercept": c, "complete": complete_share(c, gamma)})
        out["identification"].append(r)
        print(f"gamma={gamma:4.2f} c={c:+.4f} complete={r['complete']*100:5.2f}%  "
              f"cc bias={r['coarsened_bias']:+.5f} cc cov={r['coarsened_coverage']*100:5.2f}%  "
              f"seq bias={r['sequential_bias']:+.5f} seq cov={r['sequential_coverage']*100:5.2f}%")

    with open("../results/sensitivity_independent.json", "w") as f:
        json.dump(out, f, indent=2)
