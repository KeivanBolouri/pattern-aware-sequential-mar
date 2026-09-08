"""Regenerate the two manuscript figures from the independent replication."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgp import Projections, Refactorization
from oracle_sim import run

OUT = "../paper/figures"
GREY, SALMON, TEAL = "#8d8d8d", "#e8836b", "#2f9c95"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.7,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": "#333333", "ytick.color": "#333333",
})


def boxplot_figure(ref, proj, psi, reps=2000, n=2500, seed=20260908):
    res = {sc: run(sc, n, reps, seed + i, ref, proj, psi)
           for i, sc in enumerate(["ccmar", "seqmar"])}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    titles = {"ccmar": "CCMAR-compatible", "seqmar": "Sequential-MAR"}
    labels = ["Full data", "Coarsened CCMAR", "Sequential patterns"]
    keys = ["full", "cc", "seq"]
    for ax, sc in zip(axes, ["ccmar", "seqmar"]):
        data = [res[sc]["_est"][k] for k in keys]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color="#c8801f", linewidth=1.2),
                        whiskerprops=dict(color="#555555", linewidth=0.8),
                        capprops=dict(color="#555555", linewidth=0.8))
        for patch, c in zip(bp["boxes"], [GREY, SALMON, TEAL]):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor("#555555")
            patch.set_linewidth(0.8)
        ax.axhline(psi, ls="--", color="#222222", linewidth=1.0)
        ax.set_title(titles[sc], fontsize=8.5)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7.5)
    axes[0].set_ylabel("Estimated ATE")
    axes[1].annotate(f"Truth = {psi:.3f}", xy=(0.63, 0.94), xycoords="axes fraction",
                     fontsize=7.5)
    fig.suptitle("Keeping the $L_1$-only pattern improves the missing-confounder estimator",
                 fontsize=8.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/estimator_comparison.png", dpi=300)
    plt.close(fig)
    return res


def sensitivity_figure():
    d = json.load(open("../results/sensitivity_independent.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    eff = d["efficiency"]
    x = [100 * r["complete"] for r in eff]
    yv = [r["exact_reduction_pct"] for r in eff]
    axes[0].plot(x, yv, "o-", color=TEAL, linewidth=1.4, markersize=5)
    for r, xi, yi in zip(eff, x, yv):
        axes[0].annotate(f"{100*r['l1_only']:.1f}% $L_1$-only", (xi, yi),
                         textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=6.8)
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Complete records (%)")
    axes[0].set_ylabel("Sequential variance reduction (%)")
    axes[0].set_title("A. Common-validity efficiency", fontsize=8.5, loc="left")

    idn = d["identification"]
    g = [r["gamma"] for r in idn]
    axes[1].plot(g, [r["coarsened_bias"] for r in idn], "o-", color=SALMON,
                 linewidth=1.4, markersize=5, label="Coarsened CCMAR")
    axes[1].plot(g, [r["sequential_bias"] for r in idn], "o-", color=TEAL,
                 linewidth=1.4, markersize=5, label="Sequential patterns")
    axes[1].axhline(0, color="#999999", linewidth=0.7)
    axes[1].set_xlabel(r"$L_1$ coefficient $\gamma$ in stage-two response logit")
    axes[1].set_ylabel("Bias")
    axes[1].set_title("B. Departure from CCMAR", fontsize=8.5, loc="left")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/sensitivity_summary.pdf")
    fig.savefig(f"{OUT}/sensitivity_summary.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ref = Refactorization()
    proj = Projections(ref)
    psi = proj.true_psi()
    boxplot_figure(ref, proj, psi)
    sensitivity_figure()
    print("figures written to", OUT)
