#!/usr/bin/env python3
"""Create manuscript figures from archived simulation summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "oracle_sensitivity"
MAIN = ROOT / "results" / "main_verified"
OUT = ROOT / "figures"


def write_estimator_comparison() -> None:
    replicates = pd.read_csv(MAIN / "simulation_replicates.csv")
    truth = float(pd.read_csv(MAIN / "truth.csv").iloc[0]["ATE"])
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
    fig.savefig(OUT / "estimator_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_estimator_comparison()
    dat = pd.read_csv(SOURCE / "oracle_sensitivity_summary.csv")

    eff = dat[(dat["comparison_type"] == "efficiency") &
              (dat["method"] == "Sequential patterns")].copy()
    eff = eff.sort_values("complete_proportion", ascending=False)

    ident = dat[dat["comparison_type"] == "identification"].copy()
    ident["gamma"] = ident["scenario"].str.extract(r"gamma-([0-9.]+)").astype(float)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 180,
    })
    navy = "#17365D"
    teal = "#1B7F79"
    coral = "#D96B43"

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65))

    axes[0].plot(
        100 * eff["complete_proportion"],
        100 * eff["sequential_variance_reduction"],
        marker="o", lw=2.2, color=teal,
    )
    for _, row in eff.iterrows():
        axes[0].annotate(
            f"{100 * row['l1_only_proportion']:.1f}% L1-only",
            (100 * row["complete_proportion"],
             100 * row["sequential_variance_reduction"]),
            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8,
        )
    axes[0].set_xlabel("Complete records (%)")
    axes[0].set_ylabel("Sequential variance reduction (%)")
    axes[0].set_title("A. Common-validity efficiency")
    axes[0].invert_xaxis()

    for method, color, label in [
        ("Coarsened CCMAR", coral, "Coarsened CCMAR"),
        ("Sequential patterns", teal, "Sequential patterns"),
    ]:
        part = ident[ident["method"] == method].sort_values("gamma")
        axes[1].plot(part["gamma"], part["bias"], marker="o", lw=2.2,
                     color=color, label=label)
    axes[1].axhline(0, color=navy, ls="--", lw=1)
    axes[1].set_xlabel(r"$L_1$ coefficient $\gamma$ in stage-two response logit")
    axes[1].set_ylabel("Bias")
    axes[1].set_title("B. Departure from CCMAR")
    axes[1].legend(frameon=False, loc="lower left")

    for ax in axes:
        ax.grid(axis="y", color="#D9DEE5", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.6)
    fig.savefig(OUT / "sensitivity_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "sensitivity_summary.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
