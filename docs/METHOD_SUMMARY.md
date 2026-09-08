# Method summary

## Notation

The original complete-case notation and the sequential notation are connected
by

- L_c = W: confounders observed for everyone;
- L_p = (L_1, L_2): the partially observed confounder block; and
- S = R_2 = R_1 C_2: the indicator that both partially observed confounders are
  observed.

The monotone patterns are neither observed, L_1 only, and both observed. There
is no L_2-only pattern.

Let Z_0 = (W,A,Y), Z_1 = (Z_0,L_1), and Z_2 = (Z_1,L_2). Sequential MAR assumes

    R_1 is independent of (L_1,L_2) given Z_0,

and, among R_1 = 1,

    C_2 is independent of L_2 given Z_1.

## Efficient influence function

Let G be the standard full-data augmented inverse-probability pseudo-outcome
for the ATE, and define

    Q_1 = E(G | Z_1),    Q_0 = E(G | Z_0).

The observed-data efficient influence function is

    D_seq = Q_0 - psi
            + R_1/pi_1 (Q_1 - Q_0)
            + R_2/(pi_1 pi_2) (G - Q_1).

The complete tangent-space proof is in paper.tex and the compiled manuscript.

## Exact efficiency comparison

When pi_2 depends only on Z_0, both sequential MAR and complete-case MAR are
valid. The efficiency-bound difference is

    Var(D_cc) - Var(D_seq)
      = E[(1-pi_2)/(pi_1 pi_2) (Q_1-Q_0)^2] >= 0.

The inequality is strict when second-stage missingness occurs and L_1 predicts
the full-data efficient score beyond Z_0.

If pi_2 depends on observed L_1, complete-case MAR generally fails. Comparing
the methods then concerns identification and bias, not two efficiency bounds
under the same identifying model.
