#!/usr/bin/env python3
"""Add Monte Carlo uncertainty and a paired bootstrap to verified simulations.

This script does not rerun the data-generating process.  It reads the replicate
file produced by ``validate_sequential_mar.py`` and writes two audit-friendly
tables:

* simulation_summary_with_mcse.csv
* efficiency_comparison_with_ci.csv

The bootstrap resamples paired replicates, preserving the within-replicate
comparison between the coarsened and sequential estimators.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("Full data", "Coarsened CCMAR", "Sequential patterns")


def summarize(replicates: pd.DataFrame, truth: float) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for scenario, data in replicates.groupby("scenario", sort=False):
        n_reps = len(data)
        for method in METHODS:
            estimates = data[f"est_{method}"].to_numpy()
            standard_errors = data[f"se_{method}"].to_numpy()
            covered = (
                (estimates - 1.96 * standard_errors <= truth)
                & (truth <= estimates + 1.96 * standard_errors)
            )
            empirical_sd = estimates.std(ddof=1)
            coverage = covered.mean()
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "replicates": n_reps,
                    "mean_estimate": estimates.mean(),
                    "bias": estimates.mean() - truth,
                    "mcse_bias": empirical_sd / np.sqrt(n_reps),
                    "empirical_sd": empirical_sd,
                    "mcse_empirical_sd": empirical_sd / np.sqrt(2 * (n_reps - 1)),
                    "rmse": np.sqrt(np.mean((estimates - truth) ** 2)),
                    "mean_standard_error": standard_errors.mean(),
                    "root_mean_estimated_variance": np.sqrt(np.mean(standard_errors**2)),
                    "coverage_95": coverage,
                    "mcse_coverage_95": np.sqrt(coverage * (1 - coverage) / n_reps),
                }
            )
    return pd.DataFrame(rows)


def paired_efficiency_intervals(
    replicates: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int]] = []
    for scenario, data in replicates.groupby("scenario", sort=False):
        coarsened = data["est_Coarsened CCMAR"].to_numpy()
        sequential = data["est_Sequential patterns"].to_numpy()
        n_reps = len(data)
        observed_ratio = np.var(coarsened, ddof=1) / np.var(sequential, ddof=1)

        ratios = np.empty(bootstrap_reps)
        # Work in chunks to avoid allocating one very large index matrix.
        chunk_size = min(1000, bootstrap_reps)
        cursor = 0
        while cursor < bootstrap_reps:
            take = min(chunk_size, bootstrap_reps - cursor)
            indices = rng.integers(0, n_reps, size=(take, n_reps))
            ratios[cursor : cursor + take] = (
                np.var(coarsened[indices], axis=1, ddof=1)
                / np.var(sequential[indices], axis=1, ddof=1)
            )
            cursor += take

        reductions = 1 - 1 / ratios
        rows.append(
            {
                "scenario": scenario,
                "replicates": n_reps,
                "bootstrap_replicates": bootstrap_reps,
                "variance_ratio_coarsened_over_sequential": observed_ratio,
                "variance_ratio_ci_lower": np.quantile(ratios, 0.025),
                "variance_ratio_ci_upper": np.quantile(ratios, 0.975),
                "sequential_variance_reduction": 1 - 1 / observed_ratio,
                "variance_reduction_ci_lower": np.quantile(reductions, 0.025),
                "variance_reduction_ci_upper": np.quantile(reductions, 0.975),
                "interpretation": (
                    "efficiency comparison: both estimators valid"
                    if scenario == "CCMAR-compatible"
                    else "not an efficiency comparison: coarsened CCMAR is invalid"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    replicates = pd.read_csv(args.replicates)
    truth = float(pd.read_csv(args.truth)["ATE"].iloc[0])

    summary = summarize(replicates, truth)
    efficiency = paired_efficiency_intervals(
        replicates, bootstrap_reps=args.bootstrap_reps, seed=args.seed
    )
    summary.to_csv(args.outdir / "simulation_summary_with_mcse.csv", index=False)
    efficiency.to_csv(args.outdir / "efficiency_comparison_with_ci.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(efficiency.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
