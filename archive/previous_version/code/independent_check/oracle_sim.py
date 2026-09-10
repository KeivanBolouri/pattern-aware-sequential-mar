"""Independent oracle-nuisance simulation (external check on Section 6.2)."""

import argparse
import json
import time

import numpy as np

from dgp import (Projections, Refactorization, draw_full_data, draw_response,
                 pi1, pi2, q0_complete_case, rho_z0)

Z = 1.959963984540054


def run(scenario, n, reps, seed, ref, proj, psi):
    rng = np.random.default_rng(seed)
    names = ["full", "cc", "cc_ccproj", "seq"]
    est = {k: np.empty(reps) for k in names}
    se = {k: np.empty(reps) for k in names}
    pat = np.zeros(3)

    for r in range(reps):
        a, y, l1, l2 = draw_full_data(n, rng)
        r1, c2, r2, p1, p2 = draw_response(a, y, l1, scenario, rng)

        G = ref.G(a, y, l1, l2)
        Q1 = proj.Q1(a, y, l1)
        Q0 = proj.Q0(a, y)
        rho = rho_z0(a, y, proj, scenario)
        Q0cc = q0_complete_case(a, y, proj, scenario)

        H = {}
        H["full"] = G
        H["cc"] = Q0 + r2 / rho * (G - Q0)
        H["cc_ccproj"] = Q0cc + r2 / rho * (G - Q0cc)
        H["seq"] = Q0 + r1 / p1 * (Q1 - Q0) + r2 / (p1 * p2) * (G - Q1)

        for k in names:
            est[k][r] = H[k].mean()
            se[k][r] = H[k].std(ddof=1) / np.sqrt(n)

        pat += np.array([r2.mean(), (r1 * (1 - c2)).mean(), (1 - r1).mean()])

    out = {"scenario": scenario, "n": n, "reps": reps, "seed": seed,
           "psi": psi,
           "pattern_complete": pat[0] / reps,
           "pattern_l1_only": pat[1] / reps,
           "pattern_neither": pat[2] / reps}
    for k in names:
        b = est[k].mean() - psi
        sd = est[k].std(ddof=1)
        rmse = np.sqrt(np.mean((est[k] - psi) ** 2))
        cov = np.mean(np.abs(est[k] - psi) <= Z * se[k])
        out[k] = {"bias": b, "sd": sd, "rmse": rmse, "coverage": cov,
                  "mcse_bias": sd / np.sqrt(reps),
                  "mcse_cov": np.sqrt(cov * (1 - cov) / reps),
                  "mean_se": se[k].mean()}
    out["_est"] = {k: est[k] for k in names}
    return out


def paired_bootstrap_reduction(est_cc, est_seq, psi, B, seed):
    """Bootstrap the % variance reduction over the same Monte Carlo replicates."""
    rng = np.random.default_rng(seed)
    m = len(est_cc)
    dcc, dsq = est_cc - psi, est_seq - psi
    red = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, m, m)
        vcc = np.mean(dcc[idx] ** 2)
        vsq = np.mean(dsq[idx] ** 2)
        red[b] = 100.0 * (1.0 - vsq / vcc)
    return np.percentile(red, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="../results/oracle_independent.json")
    args = ap.parse_args()

    t0 = time.time()
    ref = Refactorization()
    proj = Projections(ref)
    psi = proj.true_psi()

    res = {}
    for i, sc in enumerate(["ccmar", "seqmar"]):
        res[sc] = run(sc, args.n, args.reps, args.seed + i, ref, proj, psi)
        print(f"[{time.time()-t0:6.1f}s] {sc} done")

    # efficiency comparison in the common-validity scenario
    a = res["ccmar"]
    vcc = np.mean((a["_est"]["cc"] - psi) ** 2)
    vsq = np.mean((a["_est"]["seq"] - psi) ** 2)
    red = 100.0 * (1.0 - vsq / vcc)
    ci = paired_bootstrap_reduction(a["_est"]["cc"], a["_est"]["seq"], psi,
                                    args.bootstrap, args.seed + 99)
    eff = {"variance_ratio": vcc / vsq, "reduction_pct": red,
           "boot_lo": ci[0], "boot_hi": ci[1]}

    clean = {sc: {k: v for k, v in d.items() if k != "_est"} for sc, d in res.items()}
    clean["efficiency_ccmar"] = eff
    clean["psi"] = psi
    with open(args.out, "w") as f:
        json.dump(clean, f, indent=2)

    print(json.dumps(clean, indent=2))
    print("elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
