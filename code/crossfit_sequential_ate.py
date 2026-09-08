#!/usr/bin/env python3
"""Cross-fitted ATE example for two monotone partially observed confounders.

The file has two purposes:

1. provide a concrete implementation of the sequential estimating equation;
2. run an estimated-nuisance simulation with a known constant ATE.

The example assumes a continuous outcome because it uses weighted least squares
for the outcome regressions.  Replace ``LinearRegression`` with a learner and
loss appropriate for the outcome in an application.  All predictions used in
the final estimating equation are generated out of fold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold


TRUE_ATE = 0.5


def clip_probability(values: np.ndarray, lower: float) -> np.ndarray:
    return np.clip(values, lower, 1 - lower)


def fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None):
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(x, y, sample_weight=weights)
    return model


def fit_hgb_classifier(x: np.ndarray, y: np.ndarray, seed: int):
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def fit_hgb_regressor(x: np.ndarray, y: np.ndarray, seed: int):
    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=seed,
    )
    model.fit(x, y)
    return model


@dataclass
class CausalModels:
    propensity: LogisticRegression
    outcome0: LinearRegression
    outcome1: LinearRegression


def fit_causal_models(
    v: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> CausalModels:
    propensity = fit_logistic(v, a, weights)
    outcome0 = LinearRegression().fit(v[a == 0], y[a == 0], sample_weight=weights[a == 0])
    outcome1 = LinearRegression().fit(v[a == 1], y[a == 1], sample_weight=weights[a == 1])
    return CausalModels(propensity, outcome0, outcome1)


def aipw_pseudo_outcome(
    models: CausalModels,
    v: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    probability_floor: float,
) -> np.ndarray:
    e = clip_probability(models.propensity.predict_proba(v)[:, 1], probability_floor)
    m0 = models.outcome0.predict(v)
    m1 = models.outcome1.predict(v)
    return m1 - m0 + a * (y - m1) / e - (1 - a) * (y - m0) / (1 - e)


def simulate_data(rng: np.random.Generator, n: int, scenario: str) -> pd.DataFrame:
    x = rng.normal(size=n)
    l1 = rng.binomial(1, expit(-0.25 + 0.55 * x))
    l2 = rng.normal(0.40 * x + 0.85 * l1, 1.0, size=n)
    treatment_probability = expit(-0.30 + 0.45 * x + 0.70 * l1 + 0.45 * l2)
    a = rng.binomial(1, treatment_probability)
    y = 0.20 + TRUE_ATE * a + 0.50 * x + 0.80 * l1 + 0.60 * l2 + rng.normal(size=n)

    pi1 = expit(1.60 + 0.20 * x + 0.15 * a - 0.10 * y)
    if scenario == "CCMAR-compatible":
        pi2 = expit(-0.70 + 0.10 * x + 0.10 * a - 0.08 * y)
    elif scenario == "Sequential-MAR":
        pi2 = expit(-1.25 + 0.10 * x + 0.10 * a - 0.08 * y + 1.65 * l1)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    r1 = rng.binomial(1, pi1)
    c2 = rng.binomial(1, pi2)
    r2 = r1 * c2
    return pd.DataFrame({"X": x, "A": a, "Y": y, "L1": l1, "L2": l2, "R1": r1, "R2": r2})


def estimate_cross_fitted(
    data: pd.DataFrame,
    folds: int = 2,
    seed: int = 20260904,
    probability_floor: float = 0.025,
) -> dict[str, float]:
    """Estimate full-data, coarsened, and sequential ATEs.

    ``L1`` and ``L2`` are read only for records where their indicators permit
    their use, except for the explicitly labeled full-data simulation benchmark.
    """

    x = data["X"].to_numpy()
    a = data["A"].to_numpy(dtype=int)
    y = data["Y"].to_numpy()
    l1 = data["L1"].to_numpy(dtype=int)
    l2 = data["L2"].to_numpy()
    r1 = data["R1"].to_numpy(dtype=int)
    r2 = data["R2"].to_numpy(dtype=int)
    s = r2
    n = len(data)

    z0 = np.column_stack([x, a, y])
    z1 = np.column_stack([x, a, y, l1])
    v = np.column_stack([x, l1, l2])

    h_seq = np.empty(n)
    h_cc = np.empty(n)
    h_full = np.empty(n)
    pi1_all = np.empty(n)
    pi2_all = np.full(n, np.nan)
    pis_all = np.empty(n)

    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_id, (train, valid) in enumerate(splitter.split(np.arange(n))):
        fold_seed = seed + 1009 * (fold_id + 1)
        train_r1 = train[r1[train] == 1]
        train_complete = train[s[train] == 1]

        if len(train_complete) < 100:
            raise RuntimeError("Too few complete records for the requested learners")

        # Sequential observation models.
        pi1_model = fit_logistic(z0[train], r1[train])
        pi2_model = fit_logistic(z1[train_r1], s[train_r1])
        pi1_train = clip_probability(pi1_model.predict_proba(z0[train])[:, 1], probability_floor)
        pi1_valid = clip_probability(pi1_model.predict_proba(z0[valid])[:, 1], probability_floor)
        pi2_train_r1 = clip_probability(
            pi2_model.predict_proba(z1[train_r1])[:, 1], probability_floor
        )
        pi2_valid_r1 = clip_probability(
            pi2_model.predict_proba(z1[valid[r1[valid] == 1]])[:, 1], probability_floor
        )
        pi1_all[valid] = pi1_valid
        pi2_all[valid[r1[valid] == 1]] = pi2_valid_r1

        # Coarsened complete-record probability.  A flexible learner avoids
        # incorrectly assuming that a product of logistic hazards is logistic.
        pis_model = fit_hgb_classifier(z0[train], s[train], fold_seed)
        pis_train = clip_probability(pis_model.predict_proba(z0[train])[:, 1], probability_floor)
        pis_valid = clip_probability(pis_model.predict_proba(z0[valid])[:, 1], probability_floor)
        pis_all[valid] = pis_valid

        train_pos = {index: position for position, index in enumerate(train)}
        train_r1_pos = {index: position for position, index in enumerate(train_r1)}
        seq_weights = np.array(
            [
                1
                / (
                    pi1_train[train_pos[index]]
                    * pi2_train_r1[train_r1_pos[index]]
                )
                for index in train_complete
            ]
        )
        cc_weights = np.array([1 / pis_train[train_pos[index]] for index in train_complete])
        # Guard only against exceptional finite-sample predictions.  The chosen
        # floor already imposes the substantive positivity truncation.
        seq_weights = np.minimum(seq_weights, 1 / probability_floor**2)
        cc_weights = np.minimum(cc_weights, 1 / probability_floor)

        seq_causal = fit_causal_models(
            v[train_complete], a[train_complete], y[train_complete], seq_weights
        )
        cc_causal = fit_causal_models(
            v[train_complete], a[train_complete], y[train_complete], cc_weights
        )
        full_causal = fit_causal_models(
            v[train], a[train], y[train], np.ones(len(train))
        )

        g_seq_train = aipw_pseudo_outcome(
            seq_causal,
            v[train_complete],
            a[train_complete],
            y[train_complete],
            probability_floor,
        )
        g_cc_train = aipw_pseudo_outcome(
            cc_causal,
            v[train_complete],
            a[train_complete],
            y[train_complete],
            probability_floor,
        )

        # Sequential regressions: Q1=E(G|Z1), followed by
        # Q0=E[Q1 + C2/pi2*(G-Q1)|Z0,R1=1].
        q1_model = fit_hgb_regressor(z1[train_complete], g_seq_train, fold_seed + 1)
        q1_train_r1 = q1_model.predict(z1[train_r1])
        stage1_target = q1_train_r1.copy()
        complete_within_r1 = s[train_r1] == 1
        complete_lookup = {index: position for position, index in enumerate(train_complete)}
        for position, index in enumerate(train_r1):
            if complete_within_r1[position]:
                stage1_target[position] += (
                    g_seq_train[complete_lookup[index]] - q1_train_r1[position]
                ) / pi2_train_r1[position]
        q0_model = fit_hgb_regressor(z0[train_r1], stage1_target, fold_seed + 2)

        # The coarsened estimator has access only to complete-record G values.
        q0_cc_model = fit_hgb_regressor(z0[train_complete], g_cc_train, fold_seed + 3)

        q0_valid = q0_model.predict(z0[valid])
        q0_cc_valid = q0_cc_model.predict(z0[valid])
        h_seq[valid] = q0_valid
        h_cc[valid] = q0_cc_valid

        valid_r1 = valid[r1[valid] == 1]
        q1_valid_r1 = q1_model.predict(z1[valid_r1])
        h_seq[valid_r1] += (q1_valid_r1 - q0_valid[r1[valid] == 1]) / pi1_valid[r1[valid] == 1]

        valid_complete = valid[s[valid] == 1]
        g_seq_valid = aipw_pseudo_outcome(
            seq_causal,
            v[valid_complete],
            a[valid_complete],
            y[valid_complete],
            probability_floor,
        )
        g_cc_valid = aipw_pseudo_outcome(
            cc_causal,
            v[valid_complete],
            a[valid_complete],
            y[valid_complete],
            probability_floor,
        )
        r1_positions_in_valid = {index: position for position, index in enumerate(valid_r1)}
        valid_positions = {index: position for position, index in enumerate(valid)}
        for position, index in enumerate(valid_complete):
            pos_valid = valid_positions[index]
            pos_r1 = r1_positions_in_valid[index]
            h_seq[index] += (
                g_seq_valid[position] - q1_valid_r1[pos_r1]
            ) / (pi1_valid[pos_valid] * pi2_valid_r1[pos_r1])
            h_cc[index] += (g_cc_valid[position] - q0_cc_valid[pos_valid]) / pis_valid[pos_valid]

        h_full[valid] = aipw_pseudo_outcome(
            full_causal, v[valid], a[valid], y[valid], probability_floor
        )

    estimates = {
        "full": float(np.mean(h_full)),
        "coarsened": float(np.mean(h_cc)),
        "sequential": float(np.mean(h_seq)),
    }
    output: dict[str, float] = {}
    for name, values in (("full", h_full), ("coarsened", h_cc), ("sequential", h_seq)):
        output[f"estimate_{name}"] = estimates[name]
        output[f"se_{name}"] = float(np.std(values, ddof=1) / np.sqrt(n))
    output.update(
        {
            "complete_proportion": float(np.mean(s == 1)),
            "l1_only_proportion": float(np.mean((r1 == 1) & (s == 0))),
            "neither_proportion": float(np.mean(r1 == 0)),
            "min_pi1_oof": float(np.min(pi1_all)),
            "min_pi2_oof": float(np.nanmin(pi2_all)),
            "min_pis_oof": float(np.min(pis_all)),
        }
    )
    return output


def summarize(replicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, data in replicates.groupby("scenario", sort=False):
        for method in ("full", "coarsened", "sequential"):
            estimates = data[f"estimate_{method}"].to_numpy()
            standard_errors = data[f"se_{method}"].to_numpy()
            coverage = np.mean(
                (estimates - 1.96 * standard_errors <= TRUE_ATE)
                & (TRUE_ATE <= estimates + 1.96 * standard_errors)
            )
            sd = estimates.std(ddof=1)
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "replicates": len(data),
                    "mean_estimate": estimates.mean(),
                    "bias": estimates.mean() - TRUE_ATE,
                    "mcse_bias": sd / np.sqrt(len(data)),
                    "empirical_sd": sd,
                    "rmse": np.sqrt(np.mean((estimates - TRUE_ATE) ** 2)),
                    "root_mean_estimated_variance": np.sqrt(np.mean(standard_errors**2)),
                    "coverage_95": coverage,
                    "mcse_coverage_95": np.sqrt(coverage * (1 - coverage) / len(data)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for scenario in ("CCMAR-compatible", "Sequential-MAR"):
        for replicate in range(1, args.reps + 1):
            data = simulate_data(rng, args.n, scenario)
            result = estimate_cross_fitted(
                data,
                folds=args.folds,
                seed=args.seed + 7919 * replicate + (0 if scenario == "CCMAR-compatible" else 1),
            )
            rows.append({"scenario": scenario, "replicate": replicate, **result})
            if replicate % 10 == 0:
                print(f"{scenario}: completed {replicate}/{args.reps}", flush=True)

    replicates = pd.DataFrame(rows)
    summary = summarize(replicates)
    replicates.to_csv(args.outdir / "crossfit_replicates.csv", index=False)
    summary.to_csv(args.outdir / "crossfit_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
