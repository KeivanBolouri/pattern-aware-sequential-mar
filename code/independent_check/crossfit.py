"""Large cross-fitted, estimated-nuisance study on the manuscript's own DGP.

This replaces the 100-replicate Appendix C check, which used a *different*
data-generating process from the main text.  Running the estimated-nuisance
study on the Section 6.1 law makes the appendix directly comparable with
Table 2, and is large enough for coverage statements.

Nuisances are fitted with cubic B-spline bases saturated in the binary
variables and penalised (ridge / L2 logistic).  Everything is cross-fitted with
K folds: a fold's fitted values come only from models trained on the other
folds.
"""

import argparse
import json
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer

from dgp import (Projections, Refactorization, draw_full_data, draw_response)

warnings.filterwarnings("ignore")
Z = 1.959963984540054


# ------------------------------------------------------------ design matrices


def spline_basis(x, knots):
    st = SplineTransformer(n_knots=knots, degree=3, include_bias=False)
    return st, st


def _fit_basis(x, n_knots):
    st = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False,
                           knots="quantile")
    st.fit(x.reshape(-1, 1))
    return st


def X_z0(st_y, y, a):
    """Saturated in A, spline in Y."""
    s = st_y.transform(y.reshape(-1, 1))
    a = a.reshape(-1, 1).astype(float)
    return np.hstack([s, a, a * s])


def X_z1(st_y, y, a, l1):
    """Saturated in (A, L1), spline in Y."""
    s = st_y.transform(y.reshape(-1, 1))
    a = a.reshape(-1, 1).astype(float)
    l = l1.reshape(-1, 1).astype(float)
    return np.hstack([s, a, l, a * l, a * s, l * s, a * l * s])


def X_v(st_l2, l2, l1):
    """Saturated in L1, spline in L2."""
    s = st_l2.transform(l2.reshape(-1, 1))
    l = l1.reshape(-1, 1).astype(float)
    return np.hstack([s, l, l * s])


# ------------------------------------------------------------------- learners


def fit_logit(X, yv, w=None, C=1.0):
    m = LogisticRegression(C=C, max_iter=2000)
    m.fit(X, yv, sample_weight=w)
    return m


def fit_ridge(X, yv, w=None, alpha=1e-3):
    m = Ridge(alpha=alpha)
    m.fit(X, yv, sample_weight=w)
    return m


def one_replicate(n, scenario, rng, folds, clip_e, clip_pi,
                  ref=None, proj=None, oracle_causal=False, oracle_response=False):
    a, y, l1, l2 = draw_full_data(n, rng)
    r1, c2, r2, _, _ = draw_response(a, y, l1, scenario, rng)

    idx = rng.permutation(n)
    fold_id = np.empty(n, dtype=int)
    for k, chunk in enumerate(np.array_split(idx, folds)):
        fold_id[chunk] = k

    H_seq = np.empty(n)
    H_cc = np.empty(n)
    H_full = np.empty(n)

    for k in range(folds):
        tr, te = fold_id != k, fold_id == k

        st_y = _fit_basis(y[tr], 6)
        st_l2 = _fit_basis(l2[tr], 6)

        # ---- stage-one and stage-two response models -----------------------
        m_pi1 = fit_logit(X_z0(st_y, y[tr], a[tr]), r1[tr])
        pi1_tr = np.clip(m_pi1.predict_proba(X_z0(st_y, y[tr], a[tr]))[:, 1], clip_pi, 1)
        pi1_te = np.clip(m_pi1.predict_proba(X_z0(st_y, y[te], a[te]))[:, 1], clip_pi, 1)

        s1 = tr & (r1 == 1)
        m_pi2 = fit_logit(X_z1(st_y, y[s1], a[s1], l1[s1]), c2[s1])
        pi2_tr = np.clip(m_pi2.predict_proba(X_z1(st_y, y[tr], a[tr], l1[tr]))[:, 1], clip_pi, 1)
        pi2_te = np.clip(m_pi2.predict_proba(X_z1(st_y, y[te], a[te], l1[te]))[:, 1], clip_pi, 1)

        # complete-case response model for the coarsened analyst
        m_rho = fit_logit(X_z0(st_y, y[tr], a[tr]), r2[tr])
        rho_te = np.clip(m_rho.predict_proba(X_z0(st_y, y[te], a[te]))[:, 1], clip_pi, 1)
        rho_tr = np.clip(m_rho.predict_proba(X_z0(st_y, y[tr], a[tr]))[:, 1], clip_pi, 1)

        # ---- causal regressions on complete records, response-reweighted ----
        cm = tr & (r2 == 1)
        w_seq = 1.0 / (pi1_tr[cm[tr]] * pi2_tr[cm[tr]]) if False else None
        # recompute weights aligned to the complete-record subset
        pi1_c = np.clip(m_pi1.predict_proba(X_z0(st_y, y[cm], a[cm]))[:, 1], clip_pi, 1)
        pi2_c = np.clip(m_pi2.predict_proba(X_z1(st_y, y[cm], a[cm], l1[cm]))[:, 1], clip_pi, 1)
        rho_c = np.clip(m_rho.predict_proba(X_z0(st_y, y[cm], a[cm]))[:, 1], clip_pi, 1)
        w_seq = 1.0 / (pi1_c * pi2_c)
        w_cc = 1.0 / rho_c

        def causal_fit(weights):
            m_e = fit_logit(X_v(st_l2, l2[cm], l1[cm]), a[cm], w=weights)
            fits = {}
            for av in (0, 1):
                sel = a[cm] == av
                Xs = X_v(st_l2, l2[cm][sel], l1[cm][sel])
                fits[av] = fit_ridge(Xs, y[cm][sel], w=weights[sel])
            return m_e, fits

        def G_of(m_e, fits, l1v, l2v, av, yv):
            if oracle_causal:
                return ref.G(av, yv, l1v, l2v)
            Xv = X_v(st_l2, l2v, l1v)
            e = np.clip(m_e.predict_proba(Xv)[:, 1], clip_e, 1 - clip_e)
            m1 = fits[1].predict(Xv)
            m0 = fits[0].predict(Xv)
            return (m1 - m0 + av / e * (yv - m1) - (1 - av) / (1 - e) * (yv - m0))

        # ================= sequential (pattern-aware) pipeline ===============
        m_e, fits = causal_fit(w_seq)
        G_c = G_of(m_e, fits, l1[cm], l2[cm], a[cm], y[cm])          # complete training
        m_q1 = fit_ridge(X_z1(st_y, y[cm], a[cm], l1[cm]), G_c)

        s1tr = tr & (r1 == 1)
        q1_s1 = m_q1.predict(X_z1(st_y, y[s1tr], a[s1tr], l1[s1tr]))
        pi2_s1 = np.clip(m_pi2.predict_proba(X_z1(st_y, y[s1tr], a[s1tr], l1[s1tr]))[:, 1],
                         clip_pi, 1)
        G_s1 = np.zeros(s1tr.sum())
        cs1 = r2[s1tr] == 1
        G_s1[cs1] = G_of(m_e, fits, l1[s1tr][cs1], l2[s1tr][cs1], a[s1tr][cs1], y[s1tr][cs1])
        pseudo = q1_s1 + (r2[s1tr] / pi2_s1) * (G_s1 - q1_s1)
        m_q0 = fit_ridge(X_z0(st_y, y[s1tr], a[s1tr]), pseudo)

        q1_te = m_q1.predict(X_z1(st_y, y[te], a[te], l1[te]))
        q0_te = m_q0.predict(X_z0(st_y, y[te], a[te]))
        G_te = np.zeros(te.sum())
        ct = r2[te] == 1
        G_te[ct] = G_of(m_e, fits, l1[te][ct], l2[te][ct], a[te][ct], y[te][ct])

        if oracle_response:
            from dgp import pi1 as _pi1, pi2 as _pi2, rho_z0 as _rho
            pi1_te = _pi1(a[te], y[te])
            pi2_te = _pi2(a[te], y[te], l1[te], scenario)
            rho_te = _rho(a[te], y[te], proj, scenario)

        H_seq[te] = (q0_te
                     + r1[te] / pi1_te * (q1_te - q0_te)
                     + r2[te] / (pi1_te * pi2_te) * (G_te - q1_te))

        # ================= coarsened complete-case pipeline ==================
        m_e2, fits2 = causal_fit(w_cc)
        G_c2 = G_of(m_e2, fits2, l1[cm], l2[cm], a[cm], y[cm])
        m_q0cc = fit_ridge(X_z0(st_y, y[cm], a[cm]), G_c2)
        q0cc_te = m_q0cc.predict(X_z0(st_y, y[te], a[te]))
        G_te2 = np.zeros(te.sum())
        G_te2[ct] = G_of(m_e2, fits2, l1[te][ct], l2[te][ct], a[te][ct], y[te][ct])
        H_cc[te] = q0cc_te + r2[te] / rho_te * (G_te2 - q0cc_te)

        # ================= full-data benchmark ==============================
        m_ef = fit_logit(X_v(st_l2, l2[tr], l1[tr]), a[tr])
        fitsf = {}
        for av in (0, 1):
            sel = a[tr] == av
            fitsf[av] = fit_ridge(X_v(st_l2, l2[tr][sel], l1[tr][sel]), y[tr][sel])
        H_full[te] = G_of(m_ef, fitsf, l1[te], l2[te], a[te], y[te])

    out = {}
    for name, H in (("full", H_full), ("cc", H_cc), ("seq", H_seq)):
        out[name] = (H.mean(), H.std(ddof=1) / np.sqrt(n))
    return out


def summarise(est, se, psi, reps):
    b = est.mean() - psi
    sd = est.std(ddof=1)
    return {"bias": b, "mcse_bias": sd / np.sqrt(reps), "sd": sd,
            "mad": float(np.median(np.abs(est - psi))),
            "iqr_sd": float((np.percentile(est, 75) - np.percentile(est, 25)) / 1.349),
            "rmse": float(np.sqrt(np.mean((est - psi) ** 2))),
            "coverage": float(np.mean(np.abs(est - psi) <= Z * se)),
            "mcse_cov": float(np.sqrt(np.mean(np.abs(est - psi) <= Z * se)
                                      * (1 - np.mean(np.abs(est - psi) <= Z * se)) / reps)),
            "mean_se": float(se.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--clip-e", type=float, default=0.01)
    ap.add_argument("--clip-pi", type=float, default=0.01)
    ap.add_argument("--scenarios", default="ccmar,seqmar")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--oracle-causal", action="store_true")
    ap.add_argument("--oracle-response", action="store_true")
    args = ap.parse_args()

    ref = Refactorization()
    proj = Projections(ref)
    psi = proj.true_psi()

    res = {"n": args.n, "reps": args.reps, "folds": args.folds, "psi": psi,
           "clip_e": args.clip_e, "clip_pi": args.clip_pi}
    t0 = time.time()
    for si, sc in enumerate(args.scenarios.split(",")):
        rng = np.random.default_rng(args.seed + 1000 * si)
        est = {k: np.empty(args.reps) for k in ("full", "cc", "seq")}
        se = {k: np.empty(args.reps) for k in ("full", "cc", "seq")}
        for r in range(args.reps):
            o = one_replicate(args.n, sc, rng, args.folds, args.clip_e, args.clip_pi,
                              ref, proj, args.oracle_causal, args.oracle_response)
            for k in est:
                est[k][r], se[k][r] = o[k]
            if (r + 1) % 50 == 0:
                print(f"  {sc} {r+1}/{args.reps}  {time.time()-t0:.0f}s", flush=True)
        res[sc] = {k: summarise(est[k], se[k], psi, args.reps) for k in est}
        res[sc]["_raw"] = {k: est[k].tolist() for k in est}

    with open(f"../results/crossfit_{args.tag}.json", "w") as f:
        json.dump(res, f, indent=2)
    for sc in args.scenarios.split(","):
        print(f"\n=== {sc} (n={args.n}, {args.reps} reps, clip_e={args.clip_e}) ===")
        for k in ("full", "cc", "seq"):
            d = res[sc][k]
            print(f"  {k:5s} bias {d['bias']:+.5f} ({d['mcse_bias']:.5f})  "
                  f"sd {d['sd']:.5f}  rmse {d['rmse']:.5f}  "
                  f"cov {d['coverage']*100:.1f}%  meanSE {d['mean_se']:.5f}  "
                  f"iqrSD {d['iqr_sd']:.5f}")
    print("elapsed %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
