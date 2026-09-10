# q = 2 monotone sequential-MAR extension of
# https://github.com/alexlevis/flex-ate-confounders-MAR
#
# This script keeps the repository's data-generating law for (A,Y,L1,L2),
# replaces the single complete-case indicator S with sequential indicators
# R1 and R2, and compares:
#   1) coarsening to S = R1*R2, and
#   2) retaining the L1-only pattern (R1,R2) = (1,0).
#
# The causal score and conditional regressions are evaluated from the known
# simulation law. This deliberately isolates the missingness layer. The
# accompanying manuscript proves the corresponding observed-data EIF and the
# repository also includes a preliminary cross-fitted nuisance implementation.

set.seed(20260831)

sigma_L2 <- 1.25
n <- as.integer(Sys.getenv("N_SAMPLE", unset = "2500"))
B <- as.integer(Sys.getenv("N_REPS", unset = "1000"))
out_dir <- Sys.getenv("OUT_DIR", unset = ".")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

expit <- plogis

l1_prob <- function(a, y) {
  expit(-0.6 + 0.5 * a + 0.25 * y + 0.1 * a * y)
}

draw_full_data <- function(n) {
  A <- rbinom(n, 1, 0.5)
  Y <- rbeta(n,
             shape1 = ifelse(A == 0, 2, 4),
             shape2 = ifelse(A == 0, 4, 2))
  p_L1 <- l1_prob(A, Y)
  L1 <- rbinom(n, 1, p_L1)
  L2 <- rnorm(n, A + Y + 2.5 * L1 * Y, sigma_L2)
  data.frame(A = A, Y = Y, L1 = L1, L2 = L2, p_L1 = p_L1)
}

trapz_rows <- function(mat, dx) {
  rowSums((mat[, -1, drop = FALSE] +
             mat[, -ncol(mat), drop = FALSE]) * 0.5) * dx
}

# ---------------------------------------------------------------------------
# Exact full-data causal score under the repository DGP
# ---------------------------------------------------------------------------

y_grid <- seq(1e-6, 1 - 1e-6, length.out = 801)
l2_grid <- seq(-7, 14, length.out = 3001)
dy <- y_grid[2] - y_grid[1]
dl2 <- l2_grid[2] - l2_grid[1]

m_grid <- list()
fL_grid <- list()

for (a in 0:1) {
  f_y <- dbeta(y_grid,
               shape1 = ifelse(a == 0, 2, 4),
               shape2 = ifelse(a == 0, 4, 2))
  p1 <- l1_prob(a, y_grid)
  for (l1 in 0:1) {
    f_l1 <- if (l1 == 1) p1 else 1 - p1
    mean_l2 <- a + y_grid + 2.5 * l1 * y_grid
    normal_mat <- vapply(
      seq_along(y_grid),
      function(j) dnorm(l2_grid, mean_l2[j], sigma_L2),
      numeric(length(l2_grid))
    )
    joint <- sweep(normal_mat, 2, f_y * f_l1, `*`)
    den <- trapz_rows(joint, dy)
    num <- trapz_rows(sweep(joint, 2, y_grid, `*`), dy)
    key <- paste(a, l1, sep = "_")
    fL_grid[[key]] <- den
    m_grid[[key]] <- num / pmax(den, 1e-300)
  }
}

e_grid <- list()
f_marginal_grid <- list()
for (l1 in 0:1) {
  f0 <- fL_grid[[paste(0, l1, sep = "_")]]
  f1 <- fL_grid[[paste(1, l1, sep = "_")]]
  e_grid[[as.character(l1)]] <- f1 / pmax(f0 + f1, 1e-300)
  f_marginal_grid[[as.character(l1)]] <- 0.5 * (f0 + f1)
}

trapz <- function(x, y) {
  sum(diff(x) * (head(y, -1) + tail(y, -1)) * 0.5)
}

psi <- numeric(2)
for (a in 0:1) {
  psi[a + 1] <- sum(vapply(0:1, function(l1) {
    trapz(l2_grid,
          m_grid[[paste(a, l1, sep = "_")]] *
            f_marginal_grid[[as.character(l1)]])
  }, numeric(1)))
}
truth <- psi[2] - psi[1]

interp_l2 <- function(values, x) {
  approx(l2_grid, values, xout = x, rule = 2, ties = "ordered")$y
}

causal_score <- function(A, Y, L1, L2) {
  ans <- numeric(length(Y))
  for (level in 0:1) {
    keep <- L1 == level
    if (!any(keep)) next
    m0 <- interp_l2(m_grid[[paste(0, level, sep = "_")]], L2[keep])
    m1 <- interp_l2(m_grid[[paste(1, level, sep = "_")]], L2[keep])
    e1 <- interp_l2(e_grid[[as.character(level)]], L2[keep])
    e1 <- pmin(pmax(e1, 1e-6), 1 - 1e-6)
    d1 <- m1 + (A[keep] == 1) * (Y[keep] - m1) / e1
    d0 <- m0 + (A[keep] == 0) * (Y[keep] - m0) / (1 - e1)
    ans[keep] <- d1 - d0
  }
  ans
}

# ---------------------------------------------------------------------------
# Q1(A,Y,L1) = E[G | A,Y,L1] and Q0(A,Y) = E[G | A,Y]
# ---------------------------------------------------------------------------

gh_nodes <- c(
  -5.3874808900112328, -4.6036824495507442, -3.9447640401156252,
  -3.3478545673832163, -2.7888060584281305, -2.2549740020892757,
  -1.7385377121165861, -1.2340762153953231, -0.73747372854539439,
  -0.24534070830090124, 0.24534070830090124, 0.73747372854539439,
  1.2340762153953231, 1.7385377121165861, 2.2549740020892757,
  2.7888060584281305, 3.3478545673832163, 3.9447640401156252,
  4.6036824495507442, 5.3874808900112328
)
gh_weights <- c(
  1.2578006724379234e-13, 2.4820623623151755e-10,
  6.127490259982928e-08, 4.402121090230851e-06,
  0.00012882627996192928, 0.00183010313108049,
  0.013997837447101022, 0.061506372063976897,
  0.16173933398399998, 0.26079306344955488,
  0.26079306344955488, 0.16173933398399998,
  0.061506372063976897, 0.013997837447101022,
  0.00183010313108049, 0.00012882627996192928,
  4.402121090230851e-06, 6.127490259982928e-08,
  2.4820623623151755e-10, 1.2578006724379234e-13
)

cond_y_grid <- seq(1e-6, 1 - 1e-6, length.out = 1001)
Q1_grid <- list()
Q0_grid <- list()

for (a in 0:1) {
  for (l1 in 0:1) {
    center <- a + cond_y_grid + 2.5 * l1 * cond_y_grid
    l2_nodes <- outer(center, sqrt(2) * sigma_L2 * gh_nodes, `+`)
    yy <- rep(cond_y_grid, each = length(gh_nodes))
    gg <- causal_score(
      A = rep(a, length(yy)),
      Y = yy,
      L1 = rep(l1, length(yy)),
      L2 = as.vector(t(l2_nodes))
    )
    gg_mat <- matrix(gg, nrow = length(cond_y_grid), byrow = TRUE)
    Q1_grid[[paste(a, l1, sep = "_")]] <- as.vector(gg_mat %*% gh_weights)
  }
  p1 <- l1_prob(a, cond_y_grid)
  Q0_grid[[as.character(a)]] <-
    (1 - p1) * Q1_grid[[paste(a, 0, sep = "_")]] +
    p1 * Q1_grid[[paste(a, 1, sep = "_")]]
}

interp_y <- function(values, y) {
  approx(cond_y_grid, values, xout = y, rule = 2, ties = "ordered")$y
}

predict_Q <- function(A, Y, L1) {
  Q0 <- Q1_L0 <- Q1_L1 <- numeric(length(Y))
  for (a in 0:1) {
    keep <- A == a
    Q0[keep] <- interp_y(Q0_grid[[as.character(a)]], Y[keep])
    Q1_L0[keep] <- interp_y(Q1_grid[[paste(a, 0, sep = "_")]], Y[keep])
    Q1_L1[keep] <- interp_y(Q1_grid[[paste(a, 1, sep = "_")]], Y[keep])
  }
  list(Q0 = Q0, Q1 = ifelse(L1 == 1, Q1_L1, Q1_L0))
}

# ---------------------------------------------------------------------------
# Missingness mechanisms and estimators
# ---------------------------------------------------------------------------

missingness_probabilities <- function(scenario, A, Y, L1) {
  # Most records reveal L1, while many still lack L2. This makes the partial
  # pattern common enough to reveal the information lost by coarsening.
  pi1 <- expit(2.00 + 0.10 * A + 0.10 * Y)
  if (scenario == "CCMAR-compatible") {
    pi2 <- expit(-1.10 + 0.10 * A + 0.10 * Y)
    piS <- pi1 * pi2
  } else if (scenario == "Sequential-MAR") {
    pi2 <- expit(-1.90 + 0.10 * A + 0.10 * Y + 2.20 * L1)
    p1 <- l1_prob(A, Y)
    pi2_L0 <- expit(-1.90 + 0.10 * A + 0.10 * Y)
    pi2_L1 <- expit(-1.90 + 0.10 * A + 0.10 * Y + 2.20)
    piS <- pi1 * ((1 - p1) * pi2_L0 + p1 * pi2_L1)
  } else {
    stop("Unknown scenario")
  }
  list(pi1 = pi1, pi2 = pi2, piS = piS)
}

run_one <- function(scenario, n) {
  dat <- draw_full_data(n)
  G <- with(dat, causal_score(A, Y, L1, L2))
  QQ <- with(dat, predict_Q(A, Y, L1))
  pp <- with(dat, missingness_probabilities(scenario, A, Y, L1))

  R1 <- rbinom(n, 1, pp$pi1)
  R2 <- R1 * rbinom(n, 1, pp$pi2)
  S <- R1 * R2

  # Coarsened complete/incomplete estimating function.
  H_ccmar <- QQ$Q0 + S / pp$piS * (G - QQ$Q0)

  # New q=2 monotone sequential-MAR estimating function:
  # Q0 + R1/pi1 (Q1-Q0) + R1*R2/(pi1*pi2) (G-Q1).
  H_seq <- QQ$Q0 +
    R1 / pp$pi1 * (QQ$Q1 - QQ$Q0) +
    S / (pp$pi1 * pp$pi2) * (G - QQ$Q1)

  estimates <- c(
    `Full data` = mean(G),
    `Coarsened CCMAR` = mean(H_ccmar),
    `Sequential patterns` = mean(H_seq)
  )
  ses <- c(
    `Full data` = sd(G) / sqrt(n),
    `Coarsened CCMAR` = sd(H_ccmar) / sqrt(n),
    `Sequential patterns` = sd(H_seq) / sqrt(n)
  )
  patterns <- c(
    `R1=1,R2=1` = mean(R1 == 1 & R2 == 1),
    `R1=1,R2=0` = mean(R1 == 1 & R2 == 0),
    `R1=0,R2=0` = mean(R1 == 0)
  )
  list(estimates = estimates, ses = ses, patterns = patterns)
}

scenarios <- c("CCMAR-compatible", "Sequential-MAR")
replicate_rows <- vector("list", length(scenarios) * B)
pattern_rows <- vector("list", length(scenarios) * B)
row_id <- 1

for (scenario in scenarios) {
  for (b in seq_len(B)) {
    ans <- run_one(scenario, n)
    replicate_rows[[row_id]] <- data.frame(
      scenario = scenario,
      replicate = b,
      est_full = ans$estimates["Full data"],
      est_ccmar = ans$estimates["Coarsened CCMAR"],
      est_sequential = ans$estimates["Sequential patterns"],
      se_full = ans$ses["Full data"],
      se_ccmar = ans$ses["Coarsened CCMAR"],
      se_sequential = ans$ses["Sequential patterns"],
      row.names = NULL
    )
    pattern_rows[[row_id]] <- data.frame(
      scenario = scenario,
      replicate = b,
      complete = ans$patterns["R1=1,R2=1"],
      L1_only = ans$patterns["R1=1,R2=0"],
      neither = ans$patterns["R1=0,R2=0"],
      row.names = NULL
    )
    row_id <- row_id + 1
  }
}

replicates <- do.call(rbind, replicate_rows)
patterns <- do.call(rbind, pattern_rows)

method_map <- list(
  `Full data` = c(est = "est_full", se = "se_full"),
  `Coarsened CCMAR` = c(est = "est_ccmar", se = "se_ccmar"),
  `Sequential patterns` = c(est = "est_sequential", se = "se_sequential")
)

summary_rows <- list()
k <- 1
for (scenario in scenarios) {
  ss <- replicates[replicates$scenario == scenario, ]
  for (method in names(method_map)) {
    est <- ss[[method_map[[method]]["est"]]]
    se <- ss[[method_map[[method]]["se"]]]
    summary_rows[[k]] <- data.frame(
      scenario = scenario,
      method = method,
      mean_estimate = mean(est),
      bias = mean(est) - truth,
      empirical_sd = sd(est),
      rmse = sqrt(mean((est - truth)^2)),
      mean_se = mean(se),
      coverage_95 = mean(est - 1.96 * se <= truth & truth <= est + 1.96 * se)
    )
    k <- k + 1
  }
}
summary_table <- do.call(rbind, summary_rows)
summary_table$variance_ratio_ccmar_over_seq <- NA_real_
for (scenario in scenarios) {
  cc_sd <- summary_table$empirical_sd[
    summary_table$scenario == scenario & summary_table$method == "Coarsened CCMAR"
  ]
  keep <- summary_table$scenario == scenario & summary_table$method == "Sequential patterns"
  summary_table$variance_ratio_ccmar_over_seq[keep] <-
    (cc_sd / summary_table$empirical_sd[keep])^2
}

pattern_summary <- aggregate(
  cbind(complete, L1_only, neither) ~ scenario,
  data = patterns,
  FUN = mean
)

write.csv(replicates, file.path(out_dir, "simulation_replicates_R.csv"), row.names = FALSE)
write.csv(summary_table, file.path(out_dir, "simulation_summary_R.csv"), row.names = FALSE)
write.csv(pattern_summary, file.path(out_dir, "pattern_proportions_R.csv"), row.names = FALSE)
write.csv(data.frame(psi0 = psi[1], psi1 = psi[2], ATE = truth),
          file.path(out_dir, "truth_R.csv"), row.names = FALSE)

# Figure corresponding to the repository's Figure 1. The sequential extension
# does not change p(Y|A), because A and Y are fully observed for every record.
fig_dat <- draw_full_data(n)
png(file.path(out_dir, "figure1_mu_shared_R.png"), width = 1800, height = 1100, res = 220)
den0 <- density(fig_dat$Y[fig_dat$A == 0], from = 0, to = 1, bw = 0.055)
den1 <- density(fig_dat$Y[fig_dat$A == 1], from = 0, to = 1, bw = 0.055)
plot(den0, col = "#F8766D", lwd = 2, ylim = c(0, 2.7),
     main = "Conditional density p(Y | A) is shared by both methods",
     xlab = "Y", ylab = "Density")
lines(den1, col = "#00A7B5", lwd = 2)
curve(dbeta(x, 2, 4), add = TRUE, col = "#F8766D", lty = 2, lwd = 2)
curve(dbeta(x, 4, 2), add = TRUE, col = "#00A7B5", lty = 2, lwd = 2)
legend("top", ncol = 2, bty = "n",
       legend = c("Estimated, A=0", "Estimated, A=1", "True, A=0", "True, A=1"),
       col = c("#F8766D", "#00A7B5", "#F8766D", "#00A7B5"),
       lty = c(1, 1, 2, 2), lwd = 2)
dev.off()

png(file.path(out_dir, "estimator_comparison_R.png"), width = 2300, height = 1100, res = 220)
par(mfrow = c(1, 2), mar = c(7, 5, 4, 1))
for (scenario in scenarios) {
  ss <- replicates[replicates$scenario == scenario, ]
  boxplot(
    list(
      `Full data` = ss$est_full,
      `Coarsened CCMAR` = ss$est_ccmar,
      `Sequential patterns` = ss$est_sequential
    ),
    col = c("#8A8A8A", "#E76F51", "#2A9D8F"),
    las = 2, ylab = "Estimated ATE", main = scenario,
    outline = FALSE
  )
  abline(h = truth, lty = 2, lwd = 2)
}
dev.off()

cat(sprintf("Truth: psi0=%.6f, psi1=%.6f, ATE=%.6f\n", psi[1], psi[2], truth))
print(summary_table, digits = 5, row.names = FALSE)
print(pattern_summary, digits = 4, row.names = FALSE)
