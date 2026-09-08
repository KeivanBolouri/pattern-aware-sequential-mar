#!/usr/bin/env python3
"""Check the integrity and headline values of the archived repository."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual!r} differs from expected {expected!r}")


def main() -> None:
    required = [
        ROOT / "paper.tex",
        ROOT / "output/pdf/pattern_aware_sequential_MAR_paper.pdf",
        ROOT / "figures/estimator_comparison.png",
        ROOT / "figures/sensitivity_summary.pdf",
        ROOT / "results/main_verified/simulation_replicates.csv",
        ROOT / "results/oracle_sensitivity/oracle_sensitivity_replicates.csv",
        ROOT / "results/crossfit/crossfit_replicates.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing repository files: " + ", ".join(missing))

    main_reps = pd.read_csv(ROOT / "results/main_verified/simulation_replicates.csv")
    sensitivity_reps = pd.read_csv(
        ROOT / "results/oracle_sensitivity/oracle_sensitivity_replicates.csv"
    )
    crossfit_reps = pd.read_csv(ROOT / "results/crossfit/crossfit_replicates.csv")
    assert len(main_reps) == 4_000
    assert len(sensitivity_reps) == 30_000
    assert len(crossfit_reps) == 200

    truth = pd.read_csv(ROOT / "results/main_verified/truth.csv").iloc[0]
    close(float(truth["ATE"]), 0.2558486202827889)

    efficiency = pd.read_csv(
        ROOT / "results/main_verified/efficiency_comparison_with_ci.csv"
    )
    common = efficiency.loc[efficiency["scenario"] == "CCMAR-compatible"].iloc[0]
    close(float(common["variance_ratio_coarsened_over_sequential"]), 1.0466610790094335)
    close(float(common["sequential_variance_reduction"]), 0.0445808867313513)
    close(float(common["variance_reduction_ci_lower"]), 0.02709124279038123)
    close(float(common["variance_reduction_ci_upper"]), 0.06295262301399628)

    identity = pd.read_csv(
        ROOT / "results/main_verified/theoretical_variance_check.csv"
    ).iloc[0]
    assert int(identity["integration_draws"]) == 1_500_000
    close(float(identity["sequential_variance_reduction"]), 0.04225050805353603)

    print("Repository check passed.")
    print(f"Primary replicate rows: {len(main_reps):,}")
    print(f"Sensitivity replicate rows: {len(sensitivity_reps):,}")
    print(f"Cross-fit replicate rows: {len(crossfit_reps):,}")
    print("Verified sequential variance reduction: 4.4581%")


if __name__ == "__main__":
    main()
