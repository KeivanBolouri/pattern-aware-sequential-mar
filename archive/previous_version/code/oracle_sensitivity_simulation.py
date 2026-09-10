#!/usr/bin/env python3
"""Oracle sensitivity study for missingness severity and CCMAR departures.

The full-data law and oracle causal score are imported from the independently
reproduced validator.  The study keeps the full-data law fixed while varying:

* the second-stage observation rate when CCMAR is valid; and
* the strength of dependence of second-stage observation on observed L1.

For the departure scenarios the intercept is calibrated so the marginal
second-stage observation rate stays approximately equal to the moderate CCMAR
scenario.  This separates assumption failure from a simple change in the
amount of missingness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit


METHODS = ("Full data", "Coarsened CCMAR", "Sequential patterns")


def calibrate_intercept(
    gamma: float,
    target: float,
    a: np.ndarray,
    y: np.ndarray,
    l1_probability: np.ndarray,
) -> float:
    def mean_probability(intercept: float) -> float:
        p0 = expit(intercept + 0.10 * a + 0.10 * y)
        p1 = expit(intercept + 0.10 * a + 0.10 * y + gamma)
        return float(np.mean((1 - l1_probability) * p0 + l1_probability * p1))

    return float(brentq(lambda value: mean_probability(value) - target, -12, 8))


def summarize(replicates: pd.DataFrame, truth: float) -> pd.DataFrame:
    rows = []
    for scenario, data in replicates.groupby("scenario", sort=False):
        for method in METHODS:
            estimates = data[f"est_{method}"].to_numpy()
            standard_errors = data[f"se_{method}"].to_numpy()
            n_reps = len(data)
            sd = estimates.std(ddof=1)
            coverage = np.mean(
                (estimates - 1.96 * standard_errors <= truth)
                & (truth <= estimates + 1.96 * standard_errors)
            )
            rows.append(
                {
                    "scenario": scenario,
                    "comparison_type": data["comparison_type"].iloc[0],
                    "method": method,
                    "replicates": n_reps,
                    "mean_estimate": estimates.mean(),
                    "bias": estimates.mean() - truth,
                    "mcse_bias": sd / np.sqrt(n_reps),
                    "empirical_sd": sd,
                    "mcse_empirical_sd": sd / np.sqrt(2 * (n_reps - 1)),
                    "rmse": np.sqrt(np.mean((estimates - truth) ** 2)),
                    "coverage_95": coverage,
                    "mcse_coverage_95": np.sqrt(coverage * (1 - coverage) / n_reps),
                    "complete_proportion": data["complete_proportion"].mean(),
                    "l1_only_proportion": data["l1_only_proportion"].mean(),
                    "neither_proportion": data["neither_proportion"].mean(),
                }
            )

    output = pd.DataFrame(rows)
    output["variance_ratio_coarsened_over_sequential"] = np.nan
    output["sequential_variance_reduction"] = np.nan
    for scenario, data in output.groupby("scenario", sort=False):
        if data["comparison_type"].iloc[0] != "efficiency":
            continue
        coarsened_sd = data.loc[data.method == "Coarsened CCMAR", "empirical_sd"].iloc[0]
        sequential_sd = data.loc[data.method == "Sequential patterns", "empirical_sd"].iloc[0]
        ratio = (coarsened_sd / sequential_sd) ** 2
        mask = (output.scenario == scenario) & (output.method == "Sequential patterns")
        output.loc[mask, "variance_ratio_coarsened_over_sequential"] = ratio
        output.loc[mask, "sequential_variance_reduction"] = 1 - 1 / ratio
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.validator_dir.resolve()))
    from validate_sequential_mar import OracleCausalScore, draw_full_data, l1_prob  # noqa: PLC0415

    args.outdir.mkdir(parents=True, exist_ok=True)
    oracle = OracleCausalScore()
    rng = np.random.default_rng(args.seed)

    calibration = draw_full_data(np.random.default_rng(args.seed + 1), 600_000)
    cal_a, cal_y = calibration["A"], calibration["Y"]
    cal_l1_probability = l1_prob(cal_a, cal_y)
    target_second_stage_rate = float(np.mean(expit(-1.10 + 0.10 * cal_a + 0.10 * cal_y)))

    scenarios: list[dict[str, float | str]] = [
        {
            "name": "CCMAR-low-L2-missingness",
            "comparison_type": "efficiency",
            "intercept": 0.20,
            "gamma": 0.0,
        },
        {
            "name": "CCMAR-moderate-L2-missingness",
            "comparison_type": "efficiency",
            "intercept": -1.10,
            "gamma": 0.0,
        },
        {
            "name": "CCMAR-high-L2-missingness",
            "comparison_type": "efficiency",
            "intercept": -2.00,
            "gamma": 0.0,
        },
    ]
    for gamma in (0.75, 1.50, 2.25):
        scenarios.append(
            {
                "name": f"Sequential-departure-gamma-{gamma:.2f}",
                "comparison_type": "identification",
                "intercept": calibrate_intercept(
                    gamma, target_second_stage_rate, cal_a, cal_y, cal_l1_probability
                ),
                "gamma": gamma,
            }
        )

    rows = []
    for scenario in scenarios:
        name = str(scenario["name"])
        intercept = float(scenario["intercept"])
        gamma = float(scenario["gamma"])
        for replicate in range(1, args.reps + 1):
            data = draw_full_data(rng, args.n)
            a, y, l1, l2 = (data[key] for key in ("A", "Y", "L1", "L2"))
            g = oracle.score(a, y, l1, l2)
            q0, q1_l0, q1_l1 = oracle.conditional_scores(a, y)
            q1 = np.where(l1 == 1, q1_l1, q1_l0)

            pi1 = expit(2.00 + 0.10 * a + 0.10 * y)
            pi2 = expit(intercept + 0.10 * a + 0.10 * y + gamma * l1)
            l1_probability = l1_prob(a, y)
            pi2_l0 = expit(intercept + 0.10 * a + 0.10 * y)
            pi2_l1 = expit(intercept + 0.10 * a + 0.10 * y + gamma)
            pi_s = pi1 * ((1 - l1_probability) * pi2_l0 + l1_probability * pi2_l1)

            r1 = rng.binomial(1, pi1)
            c2 = rng.binomial(1, pi2)
            r2 = r1 * c2

            h_full = g
            h_coarsened = q0 + r2 / pi_s * (g - q0)
            h_sequential = (
                q0
                + r1 / pi1 * (q1 - q0)
                + r2 / (pi1 * pi2) * (g - q1)
            )

            row: dict[str, float | str | int] = {
                "scenario": name,
                "comparison_type": str(scenario["comparison_type"]),
                "intercept": intercept,
                "gamma": gamma,
                "replicate": replicate,
                "complete_proportion": np.mean(r2 == 1),
                "l1_only_proportion": np.mean((r1 == 1) & (r2 == 0)),
                "neither_proportion": np.mean(r1 == 0),
            }
            for method, values in (
                ("Full data", h_full),
                ("Coarsened CCMAR", h_coarsened),
                ("Sequential patterns", h_sequential),
            ):
                row[f"est_{method}"] = float(np.mean(values))
                row[f"se_{method}"] = float(np.std(values, ddof=1) / np.sqrt(args.n))
            rows.append(row)
        print(f"{name}: completed {args.reps} replicates", flush=True)

    replicates = pd.DataFrame(rows)
    summary = summarize(replicates, oracle.ate)
    design = pd.DataFrame(scenarios)
    design["target_second_stage_rate"] = target_second_stage_rate

    design.to_csv(args.outdir / "oracle_sensitivity_design.csv", index=False)
    replicates.to_csv(args.outdir / "oracle_sensitivity_replicates.csv", index=False)
    summary.to_csv(args.outdir / "oracle_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
