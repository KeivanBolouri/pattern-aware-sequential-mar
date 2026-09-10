#!/usr/bin/env python3
"""Numerically verify the exact efficiency-gap identity.

The identity is evaluated without simulating missingness indicators.  Conditional
second moments are integrated over the known full-data law, which is much more
stable than subtracting two empirical variances containing inverse weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1_500_000)
    parser.add_argument("--sample-size", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=86420)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.validator_dir.resolve()))
    from validate_sequential_mar import (  # noqa: PLC0415
        OracleCausalScore,
        draw_full_data,
        missingness_probabilities,
    )

    rng = np.random.default_rng(args.seed)
    oracle = OracleCausalScore()
    data = draw_full_data(rng, args.draws)
    a, y, l1, l2 = (data[name] for name in ("A", "Y", "L1", "L2"))

    g = oracle.score(a, y, l1, l2)
    q0, q1_l0, q1_l1 = oracle.conditional_scores(a, y)
    q1 = np.where(l1 == 1, q1_l1, q1_l0)
    pi1, pi2, _ = missingness_probabilities("CCMAR-compatible", a, y, l1)

    u = q1 - q0
    v = g - q1
    # Orthogonal martingale decomposition of the sequential EIF variance.
    var_seq = (
        np.mean((q0 - oracle.ate) ** 2)
        + np.mean(u**2 / pi1)
        + np.mean(v**2 / (pi1 * pi2))
    )
    exact_gap = np.mean((1 - pi2) * u**2 / (pi1 * pi2))
    var_coarsened = var_seq + exact_gap

    output = pd.DataFrame(
        [
            {
                "integration_draws": args.draws,
                "target_ate": oracle.ate,
                "sequential_eif_variance": var_seq,
                "coarsened_eif_variance": var_coarsened,
                "variance_gap": exact_gap,
                "variance_ratio_coarsened_over_sequential": var_coarsened / var_seq,
                "sequential_variance_reduction": 1 - var_seq / var_coarsened,
                "asymptotic_sd_sequential": np.sqrt(var_seq / args.sample_size),
                "asymptotic_sd_coarsened": np.sqrt(var_coarsened / args.sample_size),
            }
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(output.to_string(index=False, float_format=lambda x: f"{x:.8f}"))


if __name__ == "__main__":
    main()
