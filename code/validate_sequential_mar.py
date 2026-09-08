#!/usr/bin/env python3
"""Numerical validation for the q=2 sequential-MAR extension.

This mirrors the data-generating law in alexlevis/flex-ate-confounders-MAR.
It uses oracle nuisance functions so the experiment isolates the missingness
layer: coarsening (R1,R2) to S=R1*R2 versus retaining the R1=1,R2=0 pattern.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import gaussian_kde, norm


SIGMA_L2 = 1.25


def l1_prob(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return expit(-0.6 + 0.5 * a + 0.25 * y + 0.1 * a * y)


def draw_full_data(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    a = rng.binomial(1, 0.5, n)
    y = rng.beta(np.where(a == 0, 2.0, 4.0), np.where(a == 0, 4.0, 2.0))
    p_l1 = l1_prob(a, y)
    l1 = rng.binomial(1, p_l1)
    mean_l2 = a + y + 2.5 * l1 * y
    l2 = rng.normal(mean_l2, SIGMA_L2)
    return {"A": a, "Y": y, "L1": l1, "L2": l2, "p_L1": p_l1}


class OracleCausalScore:
    """Grid-based exact nuisance functions for the repository DGP."""

    def __init__(self, n_y: int = 801, n_l2: int = 3001) -> None:
        self.y_grid = np.linspace(1e-6, 1 - 1e-6, n_y)
        self.l2_grid = np.linspace(-7.0, 14.0, n_l2)
        self.m = {}
        self.f_l = {}

        y = self.y_grid[None, :]
        l2 = self.l2_grid[:, None]
        for a in (0, 1):
            f_y = beta_dist.pdf(self.y_grid, 2 if a == 0 else 4, 4 if a == 0 else 2)[None, :]
            p = l1_prob(np.full_like(self.y_grid, a), self.y_grid)[None, :]
            for l1 in (0, 1):
                f_l1 = p if l1 == 1 else 1 - p
                mu_l2 = a + y + 2.5 * l1 * y
                joint = f_y * f_l1 * norm.pdf(l2, loc=mu_l2, scale=SIGMA_L2)
                den = np.trapezoid(joint, self.y_grid, axis=1)
                num = np.trapezoid(joint * y, self.y_grid, axis=1)
                self.f_l[(a, l1)] = den
                self.m[(a, l1)] = num / np.maximum(den, 1e-300)

        self.e = {}
        self.f_marginal = {}
        for l1 in (0, 1):
            f0 = self.f_l[(0, l1)]
            f1 = self.f_l[(1, l1)]
            self.e[l1] = f1 / np.maximum(f0 + f1, 1e-300)
            self.f_marginal[l1] = 0.5 * (f0 + f1)

        psi = {}
        for a in (0, 1):
            psi[a] = sum(
                np.trapezoid(self.m[(a, l1)] * self.f_marginal[l1], self.l2_grid)
                for l1 in (0, 1)
            )
        self.psi0 = float(psi[0])
        self.psi1 = float(psi[1])
        self.ate = self.psi1 - self.psi0

        nodes, weights = hermgauss(20)
        self.gh_nodes = nodes
        self.gh_weights = weights / np.sqrt(np.pi)

        # The conditional expectations needed by the monotone-MAR EIF depend
        # only on (A,Y,L1). Precompute them once and interpolate in every
        # simulation replicate.
        self.cond_y_grid = np.linspace(1e-6, 1 - 1e-6, 1001)
        self.m2_grid = {}
        self.m1_grid = {}
        for a_level in (0, 1):
            aa_grid = np.full(len(self.cond_y_grid), a_level, dtype=int)
            for l1_level in (0, 1):
                center = aa_grid + self.cond_y_grid + 2.5 * l1_level * self.cond_y_grid
                l2_nodes = center[:, None] + SIGMA_L2 * np.sqrt(2.0) * self.gh_nodes[None, :]
                aa = np.repeat(aa_grid, len(self.gh_nodes))
                yy = np.repeat(self.cond_y_grid, len(self.gh_nodes))
                ll1 = np.full(aa.shape, l1_level, dtype=int)
                gg = self.score(aa, yy, ll1, l2_nodes.ravel()).reshape(len(aa_grid), -1)
                self.m2_grid[(a_level, l1_level)] = gg @ self.gh_weights
            p = l1_prob(aa_grid, self.cond_y_grid)
            self.m1_grid[a_level] = (
                (1 - p) * self.m2_grid[(a_level, 0)] + p * self.m2_grid[(a_level, 1)]
            )

    def _interp(self, values: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.l2_grid, values, left=values[0], right=values[-1])

    def score(self, a: np.ndarray, y: np.ndarray, l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
        out = np.empty_like(y, dtype=float)
        for level in (0, 1):
            keep = l1 == level
            if not np.any(keep):
                continue
            m0 = self._interp(self.m[(0, level)], l2[keep])
            m1 = self._interp(self.m[(1, level)], l2[keep])
            e1 = self._interp(self.e[level], l2[keep])
            e1 = np.clip(e1, 1e-6, 1 - 1e-6)
            d1 = m1 + (a[keep] == 1) * (y[keep] - m1) / e1
            d0 = m0 + (a[keep] == 0) * (y[keep] - m0) / (1 - e1)
            out[keep] = d1 - d0
        return out

    def conditional_scores(self, a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return m1(A,Y), m2(A,Y,L1=0), and m2(A,Y,L1=1)."""
        m1 = np.empty_like(y, dtype=float)
        m2_l0 = np.empty_like(y, dtype=float)
        m2_l1 = np.empty_like(y, dtype=float)
        for a_level in (0, 1):
            keep = a == a_level
            m1[keep] = np.interp(y[keep], self.cond_y_grid, self.m1_grid[a_level])
            m2_l0[keep] = np.interp(y[keep], self.cond_y_grid, self.m2_grid[(a_level, 0)])
            m2_l1[keep] = np.interp(y[keep], self.cond_y_grid, self.m2_grid[(a_level, 1)])
        return m1, m2_l0, m2_l1


def missingness_probabilities(
    scenario: str, a: np.ndarray, y: np.ndarray, l1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # R1 is commonly observed while R2 is often missing.  This creates a
    # substantial L1-only pattern, which is exactly the information discarded
    # by complete-versus-incomplete coarsening.
    pi1 = expit(2.00 + 0.10 * a + 0.10 * y)
    if scenario == "CCMAR-compatible":
        pi2 = expit(-1.10 + 0.10 * a + 0.10 * y)
        pi_s_marginal = pi1 * pi2
    elif scenario == "Sequential-MAR":
        pi2 = expit(-1.90 + 0.10 * a + 0.10 * y + 2.20 * l1)
        p = l1_prob(a, y)
        pi2_l0 = expit(-1.90 + 0.10 * a + 0.10 * y)
        pi2_l1 = expit(-1.90 + 0.10 * a + 0.10 * y + 2.20)
        pi_s_marginal = pi1 * ((1 - p) * pi2_l0 + p * pi2_l1)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return pi1, pi2, pi_s_marginal


def run_one(
    rng: np.random.Generator, oracle: OracleCausalScore, n: int, scenario: str
) -> tuple[dict[str, float], dict[str, float]]:
    d = draw_full_data(rng, n)
    a, y, l1, l2 = d["A"], d["Y"], d["L1"], d["L2"]
    g = oracle.score(a, y, l1, l2)
    m1, m2_l0, m2_l1 = oracle.conditional_scores(a, y)
    m2 = np.where(l1 == 1, m2_l1, m2_l0)

    pi1, pi2, pi_s = missingness_probabilities(scenario, a, y, l1)
    r1 = rng.binomial(1, pi1)
    r2 = r1 * rng.binomial(1, pi2)
    s = r1 * r2

    h_full = g
    h_cc = m1 + s / pi_s * (g - m1)
    h_seq = m1 + r1 / pi1 * (m2 - m1) + s / (pi1 * pi2) * (g - m2)

    estimates = {
        "Full data": float(np.mean(h_full)),
        "Coarsened CCMAR": float(np.mean(h_cc)),
        "Sequential patterns": float(np.mean(h_seq)),
    }
    ses = {
        "Full data": float(np.std(h_full, ddof=1) / np.sqrt(n)),
        "Coarsened CCMAR": float(np.std(h_cc, ddof=1) / np.sqrt(n)),
        "Sequential patterns": float(np.std(h_seq, ddof=1) / np.sqrt(n)),
    }
    patterns = {
        "R1=1,R2=1": float(np.mean((r1 == 1) & (r2 == 1))),
        "R1=1,R2=0": float(np.mean((r1 == 1) & (r2 == 0))),
        "R1=0,R2=0": float(np.mean(r1 == 0)),
    }
    row = {"scenario": scenario, **{f"est_{k}": v for k, v in estimates.items()}, **{f"se_{k}": v for k, v in ses.items()}}
    return row, patterns


def summarize(replicates: pd.DataFrame, truth: float) -> pd.DataFrame:
    rows = []
    for scenario in replicates["scenario"].unique():
        ss = replicates.loc[replicates["scenario"] == scenario]
        for method in ("Full data", "Coarsened CCMAR", "Sequential patterns"):
            est = ss[f"est_{method}"].to_numpy()
            se = ss[f"se_{method}"].to_numpy()
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "mean_estimate": np.mean(est),
                    "bias": np.mean(est) - truth,
                    "empirical_sd": np.std(est, ddof=1),
                    "rmse": np.sqrt(np.mean((est - truth) ** 2)),
                    "mean_se": np.mean(se),
                    "coverage_95": np.mean((est - 1.96 * se <= truth) & (truth <= est + 1.96 * se)),
                }
            )
    out = pd.DataFrame(rows)
    for scenario in out["scenario"].unique():
        cc_sd = out.loc[(out.scenario == scenario) & (out.method == "Coarsened CCMAR"), "empirical_sd"].iloc[0]
        mask = (out.scenario == scenario) & (out.method == "Sequential patterns")
        out.loc[mask, "variance_ratio_ccmar_over_seq"] = (cc_sd / out.loc[mask, "empirical_sd"]) ** 2
    return out


def figure_mu(rng: np.random.Generator, outdir: Path, n: int = 2500) -> None:
    d = draw_full_data(rng, n)
    grid = np.linspace(0.001, 0.999, 250)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = {0: "#F8766D", 1: "#00A7B5"}
    for a in (0, 1):
        values = d["Y"][d["A"] == a]
        kde = gaussian_kde(values, bw_method=0.19)
        ax.plot(grid, kde(grid), color=colors[a], lw=1.8, label=f"Estimated, A={a}")
        true = beta_dist.pdf(grid, 2 if a == 0 else 4, 4 if a == 0 else 2)
        ax.plot(grid, true, color=colors[a], lw=1.5, ls="--", label=f"True, A={a}")
    ax.set(title="Conditional density p(Y | A) is shared by both methods", xlabel="Y", ylabel="Density")
    ax.text(0.5, -0.22, "R1 and R2 are introduced after A and Y; therefore every record contributes to this nuisance estimate.",
            ha="center", transform=ax.transAxes, fontsize=9)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "figure1_mu_shared.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_comparison(replicates: pd.DataFrame, truth: float, outdir: Path) -> None:
    methods = ["Full data", "Coarsened CCMAR", "Sequential patterns"]
    colors = ["#8A8A8A", "#E76F51", "#2A9D8F"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, scenario in zip(axes, ["CCMAR-compatible", "Sequential-MAR"]):
        ss = replicates.loc[replicates.scenario == scenario]
        arrays = [ss[f"est_{m}"].to_numpy() for m in methods]
        bp = ax.boxplot(arrays, tick_labels=methods, widths=0.58, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.axhline(truth, color="black", ls="--", lw=1.4, label=f"Truth = {truth:.3f}")
        ax.set_title(scenario)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Estimated ATE")
    fig.suptitle("Keeping the L1-only pattern improves the missing-confounder estimator", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "estimator_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    oracle = OracleCausalScore()
    rng = np.random.default_rng(args.seed)
    rows = []
    pattern_rows = []
    for scenario in ("CCMAR-compatible", "Sequential-MAR"):
        for rep in range(1, args.reps + 1):
            row, patterns = run_one(rng, oracle, args.n, scenario)
            row["replicate"] = rep
            rows.append(row)
            pattern_rows.append({"scenario": scenario, "replicate": rep, **patterns})
    replicates = pd.DataFrame(rows)
    patterns = pd.DataFrame(pattern_rows)
    summary = summarize(replicates, oracle.ate)

    replicates.to_csv(args.outdir / "simulation_replicates.csv", index=False)
    summary.to_csv(args.outdir / "simulation_summary.csv", index=False)
    patterns.groupby("scenario", as_index=False).mean(numeric_only=True).drop(columns="replicate").to_csv(
        args.outdir / "pattern_proportions.csv", index=False
    )
    pd.DataFrame([{"psi0": oracle.psi0, "psi1": oracle.psi1, "ATE": oracle.ate}]).to_csv(
        args.outdir / "truth.csv", index=False
    )
    figure_mu(np.random.default_rng(args.seed + 1), args.outdir, args.n)
    figure_comparison(replicates, oracle.ate, args.outdir)

    print(f"Truth: psi0={oracle.psi0:.6f}, psi1={oracle.psi1:.6f}, ATE={oracle.ate:.6f}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
