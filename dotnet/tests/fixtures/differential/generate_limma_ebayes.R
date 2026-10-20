#!/usr/bin/env Rscript
#
# Generates limma_ebayes.json: the moderated t from limma's own lmFit + eBayes, called whole.
#
# This is the reference Skyline-PRISM's moderated t is held to, as proteomics-toolkit's is, for:
#
#   1. The limma-trend prior, eBayes(trend = TRUE) - the default variance prior from 26.9.0.
#      fitFDist fits a natural cubic spline of the log residual variance on each feature's mean
#      log2 expression (lmFit's Amean), with 1 + (n >= 3) + (n >= 6) + (n >= 30) spline df, and
#      re-estimates the prior df around that curve.
#   2. The cap limma's eBayes puts on the moderated t's degrees of freedom:
#
#        df.total <- pmin(df.residual + df.prior, sum(df.residual))
#
#      The toolkit used df.residual + df.prior uncapped before 26.9.0, so with an infinite prior df
#      it read the t off a normal. The cap binds whenever the prior df is infinite, and otherwise
#      only when few features are tested and the prior df is large, so every binding case here is
#      small, and the large cases are there to show it changes nothing when it does not bind.
#
# Nothing here imports Skyline-PRISM or the toolkit. Each matrix is drawn here, written into the
# fixture, and the tests hand that matrix to the code under test, so it never depends on R's random
# numbers.
#
# Why R and not inmoose: inmoose 0.9.1's eBayes crashes whenever df.prior is infinite (its
# B-statistic step applies `~` to a Python bool), which is exactly the case the cap matters most
# for, and its ns() refuses the two-column (spline df 2) basis that limma uses for 3 to 5 features.
# limma itself is the definition.
#
# The same generator, cases and seeds live in proteomics-toolkit
# (tests/fixtures/generate_limma_ebayes.R), and the two write byte-identical JSON:
# the toolkit's moderated linear model and PRISM's moderated t - every prior, design and
# missing-value rule here - are pinned to one reference, which is how the two tools are held to the
# same numbers. Change a case in both, or neither. The first four cases are also the matrices of
# PRISM's older ebayes.json.
#
# Prerequisites (one-off):
#   install.packages("BiocManager")
#   BiocManager::install("limma")
#
# Run from the repository root (on Windows, Rscript.exe is in the R installation's bin folder):
#   Rscript dotnet/tests/fixtures/differential/generate_limma_ebayes.R
#
# Generated with R 4.5.1 and limma 3.66.0. The versions that actually ran are written into the
# fixture's `versions` field, so a regeneration under different ones shows up in the diff.

suppressPackageStartupMessages(library(limma))

out_path <- file.path("dotnet", "tests", "fixtures", "differential", "limma_ebayes.json")
if (!dir.exists(dirname(out_path))) stop("run from the repository root: ", dirname(out_path), " not found")

# ------------------------------------------------------------------------------------------------
# Encoding: every float is a JSON STRING holding 17 significant digits, which round-trips a double
# exactly, and the non-finite values are spelled the way Python's float() reads them ("inf",
# "-inf", "nan"). Bare NaN and Infinity are not valid JSON, and a float that goes through a JSON
# number is at the mercy of two encoders' shortest-repr rules.
# ------------------------------------------------------------------------------------------------

num <- function(x) {
  x <- as.numeric(x)
  out <- sprintf("%.17g", x)
  out[is.na(x)] <- "nan"
  out[is.infinite(x) & x > 0] <- "inf"
  out[is.infinite(x) & x < 0] <- "-inf"
  out
}

js_str <- function(s) paste0("\"", gsub("\"", "\\\\\"", s), "\"")
js_vec <- function(x) paste0("[", paste(js_str(num(x)), collapse = ", "), "]")
js_labels <- function(x) paste0("[", paste(js_str(as.character(x)), collapse = ", "), "]")
js_mat <- function(m, indent) {
  m <- as.matrix(m)
  rows <- vapply(seq_len(nrow(m)), function(i) js_vec(m[i, ]), "")
  pad <- strrep(" ", indent + 2)
  paste0("[\n", paste0(pad, rows, collapse = ",\n"), "\n", strrep(" ", indent), "]")
}

# One JSON object from a named list whose values are already-encoded JSON text.
js_obj <- function(fields, indent) {
  pad <- strrep(" ", indent + 2)
  body <- paste0(pad, js_str(names(fields)), ": ", unlist(fields), collapse = ",\n")
  paste0("{\n", body, "\n", strrep(" ", indent), "}")
}

# ------------------------------------------------------------------------------------------------
# One case. `design` is the design matrix the toolkit builds for `analysis_type` from the sample
# labels written beside it, and `coef` is the column it tests:
#
#   unpaired      [1, group == "B"]                        coef 2, logFC = B - A
#   paired        [1, time == "Post", subject dummies]     coef 2, logFC = Post - Pre
#   linear_trend  [1, time]                                coef 2, logFC = slope per unit time
#
# `binds` says whether the df cap should bind; a case that does not do what its name says is
# refused rather than written, so the golden cannot quietly stop exercising the cap.
# `spline_df` is the number of spline columns limma's fitFDist uses for the trend (NA for the
# global prior), recorded so a reader can see which branch each case reaches.
# ------------------------------------------------------------------------------------------------

encode_case <- function(name, note, expr, analysis_type, design, labels, trend, binds,
                        legacy = NULL, indent = 4) {
  fit <- eBayes(lmFit(expr, design), trend = trend, legacy = legacy)

  df_residual <- fit$df.residual
  df_prior <- fit$df.prior[1]
  df_pooled <- sum(df_residual)
  bound <- any(df_residual + df_prior > df_pooled)
  if (binds != bound)
    stop(sprintf("%s: expected the cap to %sbind; df.residual %s + df.prior %g vs pooled %g",
                 name, if (binds) "" else "not ", paste(unique(df_residual), collapse = "/"),
                 df_prior, df_pooled))
  stopifnot(all(fit$df.total == pmin(df_residual + fit$df.prior, df_pooled)))

  n_ok <- sum(is.finite(fit$sigma) & df_residual > 0)
  spline_df <- if (trend) min(1L + (n_ok >= 3L) + (n_ok >= 6L) + (n_ok >= 30L), length(unique(fit$Amean))) else NA

  cat(sprintf("  %-28s features %4d, df.residual %s, df.prior %s, pooled %g, spline df %s\n",
              name, nrow(expr), paste(unique(df_residual), collapse = "/"), num(df_prior),
              df_pooled, if (is.na(spline_df)) "-" else spline_df))

  js_obj(c(
    list(
      name = js_str(name),
      note = js_str(note),
      analysis_type = js_str(analysis_type),
      trend = if (trend) "true" else "false",
      legacy = if (is.null(legacy)) "null" else if (legacy) "true" else "false"
    ),
    lapply(labels, js_labels),
    list(
      expr = js_mat(expr, indent + 2),
      spline_df = if (is.na(spline_df)) "null" else as.character(spline_df),
      df_residual = js_vec(df_residual),
      df_prior = js_str(num(df_prior)),
      df_pooled = js_str(num(df_pooled)),
      df_total = js_vec(fit$df.total),
      s2_prior = js_vec(rep_len(fit$s2.prior, nrow(expr))),
      s2_post = js_vec(fit$s2.post),
      amean = js_vec(fit$Amean),
      logfc = js_vec(fit$coefficients[, 2]),
      t = js_vec(fit$t[, 2]),
      p = js_vec(fit$p.value[, 2])
    )
  ), indent)
}

two_arm <- function(n_a, n_b) {
  list(design = cbind(1, c(rep(0, n_a), rep(1, n_b))),
       labels = list(group = c(rep("A", n_a), rep("B", n_b))))
}

# Every row shares one residual pattern, so every residual variance is the same: fitFDist's evar
# is <= 0 and the prior df is infinite. Rows differ in level and in effect, so t differs.
equal_variance_rows <- function(levels, effects, spread = 0.5) {
  t(mapply(function(lv, e) c(lv - spread, lv, lv + spread, lv + e - spread, lv + e, lv + e + spread),
           levels, effects))
}

# Per-feature variances with a known prior: a trend in mean intensity times a scaled inverse
# chi-square, so the true prior df is finite and the residuals scatter around the curve by more
# than sampling noise. Technical noise on the log2 scale falls with intensity towards a floor.
trend_variance <- function(level, d_true = 6) {
  (0.03 + 0.6 * exp(-(level - 12) / 3)) * d_true / rchisq(length(level), d_true)
}

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
cases <- character()

# --- The df cap under the global prior (the same matrices as Skyline-PRISM's ebayes.json) -------

a <- two_arm(3, 3)
cases <- c(cases, encode_case(
  "global_infinite_prior",
  "8 features, 3 vs 3, equal residual variances: df.prior is infinite and df.total is the pooled 8 x 4 = 32, not infinity",
  equal_variance_rows(9:16, c(0.2, -0.4, 0.9, 1.6, -1.1, 0.05, 2.3, -0.7)),
  "unpaired", a$design, a$labels, trend = FALSE, binds = TRUE))

# Five residual scales whose log-variances have a sample variance of 0.70 - just above
# trigamma(2) = 0.645, the sampling variance of a log-variance at 4 df - so fitFDist's evar is small
# and positive and df.prior is large but finite, past the pooled 5 x 4 = 20.
z <- c(-2, -1, 0, 1, 2) / sqrt(2.5)
scales <- exp(sqrt(0.70 / 4) * z)
effects <- c(0.3, -0.8, 1.4, 0.1, -1.9)
finite <- t(sapply(1:5, function(i) {
  lv <- 10 + i; s <- scales[i]; e <- effects[i]
  c(lv - 0.5 * s, lv, lv + 0.5 * s, lv + e - 0.5 * s, lv + e, lv + e + 0.5 * s)
}))
cases <- c(cases, encode_case(
  "global_large_finite_prior",
  "5 features, 3 vs 3, residual variances spread just past their sampling variance: df.prior is finite, but df.residual + df.prior exceeds the pooled 20",
  finite, "unpaired", a$design, a$labels, trend = FALSE, binds = TRUE))

set.seed(271)
n_a <- 6; n_b <- 7
noise <- 0.15 + 1.5 * ((0:39) / 39)^2
many <- 18 + noise * matrix(rnorm(40 * (n_a + n_b)), nrow = 40)
many[1:5, (n_a + 1):(n_a + n_b)] <- many[1:5, (n_a + 1):(n_a + n_b)] + 1.7
b <- two_arm(n_a, n_b)
cases <- c(cases, encode_case(
  "global_cap_does_not_bind",
  "40 features, 6 vs 7, heterogeneous noise: df.residual + df.prior is far below the pooled 440, so the cap changes nothing",
  many, "unpaired", b$design, b$labels, trend = FALSE, binds = FALSE))

# --- limma-trend ------------------------------------------------------------------------------

cases <- c(cases, encode_case(
  "trend_infinite_prior",
  "limma-trend over 12 features spanning 8..19 log2 with equal residual variances: the trend is flat, df.prior is infinite and df.total is the pooled 12 x 4 = 48",
  equal_variance_rows(8:19, c(0.4, -0.6, 1.2, 0.0, -1.5, 0.8, 2.0, -0.3, 0.6, -1.0, 1.7, 0.2)),
  "unpaired", a$design, a$labels, trend = TRUE, binds = TRUE))

# The ordinary case: hundreds of features over four decades, noise falling with intensity, so the
# trend matters, and a finite prior df, so every residual variance reaches its own t. Spline df 4.
set.seed(1018)
n_feat <- 300
level <- sort(runif(n_feat, 12, 28))
sd_f <- sqrt(trend_variance(level))
expr <- level + sd_f * matrix(rnorm(n_feat * 8), nrow = n_feat)
moved <- seq(1, n_feat, by = 15)
expr[moved, 5:8] <- expr[moved, 5:8] + 1
c4 <- two_arm(4, 4)
cases <- c(cases, encode_case(
  "trend_unpaired",
  "300 features over 12..28 log2, 4 vs 4, variance falling with intensity and scattered about the curve (true prior df 6): spline df 4, finite df.prior, cap does not bind",
  expr, "unpaired", c4$design, c4$labels, trend = TRUE, binds = FALSE))

# 6 <= n < 30 gives a 3-column spline: one interior knot, at the median Amean.
set.seed(2026)
level <- sort(runif(20, 12, 26))
expr <- level + sqrt(trend_variance(level, d_true = 3)) * matrix(rnorm(20 * 6), nrow = 20)
expr[1:4, 4:6] <- expr[1:4, 4:6] + 1.5
cases <- c(cases, encode_case(
  "trend_twenty_features",
  "20 features, 3 vs 3: spline df 3 (one interior knot at the median of Amean)",
  expr, "unpaired", a$design, a$labels, trend = TRUE, binds = FALSE))

# 3 <= n < 6 gives a 2-column spline - no interior knot, so the trend is a straight line in Amean.
# Built, not drawn: five features are too few for a random draw to land reliably on either side of
# the cap. Each row's residual variance is s^2 / 4 (three points 0.5 s apart in each arm), and the
# log-variances are a falling line in Amean plus a departure orthogonal to [1, Amean] whose mean
# square over the 5 - 2 residual df is 0.70 - just above trigamma(2) = 0.645, the sampling variance
# of a log-variance at 4 df. So fitFDist's evar is small and positive, df.prior is large but
# finite, and df.residual + df.prior passes the pooled 5 x 4 = 20.
level <- c(11, 14, 17, 20, 23)
effects <- c(0.6, -1.8, 0.2, 1.1, -0.4)
amean <- level + effects / 2
departure <- residuals(lm(c(1, -2, 0, 2, -1) ~ amean))
departure <- departure * sqrt(0.70 * 3 / sum(departure^2))
log_s2 <- -0.15 * (amean - 17) + departure
scales <- 2 * exp(log_s2 / 2)
expr <- t(sapply(1:5, function(i) {
  lv <- level[i]; s <- scales[i]; e <- effects[i]
  c(lv - 0.5 * s, lv, lv + 0.5 * s, lv + e - 0.5 * s, lv + e, lv + e + 0.5 * s)
}))
cases <- c(cases, encode_case(
  "trend_five_features",
  "5 features, 3 vs 3: spline df 2, so the trend is a straight line in Amean (no interior knot); a large df.prior and only 20 pooled df, so the cap binds",
  expr, "unpaired", a$design, a$labels, trend = TRUE, binds = TRUE))

# A paired design: [1, Post, subject dummies]. The subject block removes a 1.5 log2 spread between
# people from the residual, which is why the prior has to be fitted to THIS design's residuals.
set.seed(612)
n_feat <- 120; n_subj <- 6
level <- sort(runif(n_feat, 13, 27))
subj <- rnorm(n_subj, sd = 1.5)
sd_f <- sqrt(trend_variance(level))
expr <- cbind(
  level + outer(rep(1, n_feat), subj) + sd_f * matrix(rnorm(n_feat * n_subj), nrow = n_feat),
  level + outer(rep(1, n_feat), subj) + sd_f * matrix(rnorm(n_feat * n_subj), nrow = n_feat))
moved <- seq(1, n_feat, by = 10)
expr[moved, (n_subj + 1):(2 * n_subj)] <- expr[moved, (n_subj + 1):(2 * n_subj)] - 0.8
subject <- rep(sprintf("P%d", 1:n_subj), 2)
time <- rep(c("Pre", "Post"), each = n_subj)
paired_design <- cbind(1, as.numeric(time == "Post"),
                       sapply(sprintf("P%d", 2:n_subj), function(s) as.numeric(subject == s)))
cases <- c(cases, encode_case(
  "trend_paired",
  "120 features, 6 subjects Pre then Post, design [1, Post, subject dummies]: the subject block removes a 1.5 log2 spread; spline df 4, finite df.prior",
  expr, "paired", paired_design, list(subject = subject, time = time), trend = TRUE, binds = FALSE))

# A linear trend over an uneven schedule: [1, week], the slope tested.
set.seed(1204)
n_feat <- 80
weeks <- rep(c(0, 2, 4, 6, 12), 3)
level <- sort(runif(n_feat, 12, 26))
expr <- level + sqrt(trend_variance(level)) * matrix(rnorm(n_feat * length(weeks)), nrow = n_feat)
moved <- seq(1, n_feat, by = 8)
expr[moved, ] <- expr[moved, ] + outer(rep(1, length(moved)), -0.08 * weeks)
cases <- c(cases, encode_case(
  "trend_linear",
  "80 features at weeks 0, 2, 4, 6, 12 (three samples each), design [1, week], the slope tested: spline df 4, finite df.prior",
  expr, "linear_trend", cbind(1, weeks), list(time = weeks), trend = TRUE, binds = FALSE))

# Missing values. A feature with a NaN has fewer residual df than the rest, and limma 3.58+ then
# switches squeezeVar to fitFDistUnequalDF1, a different estimator the toolkit does not implement.
# legacy = TRUE keeps limma on fitFDist - the estimator the toolkit uses throughout, as limma did
# before 3.58 - so this case pins what the toolkit does with a NaN: Amean over the observed values,
# each feature's own df, and a pooled df that sums the reduced ones. One NaN per arm at most, so
# every feature stays estimable.
set.seed(55)
n_feat <- 150
level <- sort(runif(n_feat, 12, 27))
expr <- level + sqrt(trend_variance(level)) * matrix(rnorm(n_feat * 8), nrow = n_feat)
expr[seq(3, n_feat, by = 12), 5:8] <- expr[seq(3, n_feat, by = 12), 5:8] + 1.2
expr[cbind(seq(2, n_feat, by = 7), 1 + (seq(2, n_feat, by = 7) %% 4))] <- NA
expr[cbind(seq(5, n_feat, by = 11), 5 + (seq(5, n_feat, by = 11) %% 4))] <- NA
cases <- c(cases, encode_case(
  "trend_missing_values_legacy",
  "150 features, 4 vs 4, some with one NaN per arm (residual df 5 or 4 instead of 6), eBayes(trend = TRUE, legacy = TRUE): fitFDist's moments over unequal df, which is the toolkit's estimator; limma's default for unequal df (fitFDistUnequalDF1) is not",
  expr, "unpaired", c4$design, c4$labels, trend = TRUE, binds = FALSE, legacy = TRUE))

# --- Designs a feature's missing values make rank-deficient -------------------------------------
#
# lmFit fits each row with missing values on its observed samples, and R's lm.fit pivots out any
# design column those samples cannot estimate - the LATER of an aliased pair - setting its
# coefficient to NA. The row keeps n_observed - rank residual df, and its variance goes into the
# prior and its df into the pool whether or not the tested coefficient survived. Every case here
# is legacy = TRUE, for the reason given above: the df are unequal.

# An arm entirely missing: [1, group] on samples that are all in B makes the group column equal
# the intercept, so it is the one dropped. No contrast, but two residual df each. With equal
# variances otherwise the prior df is infinite, so the cap binds - at 8 x 4 + 8 x 2 = 48, not 32.
set.seed(4)
complete <- 10 + matrix(rnorm(8 * 6, sd = 0.5), nrow = 8)
b_only <- 10 + matrix(rnorm(8 * 6, sd = 0.5), nrow = 8)
b_only[, 1:3] <- NA
cases <- c(cases, encode_case(
  "global_missing_arm_legacy",
  "8 complete features plus 8 observed only in arm B, 3 vs 3: the B-only rows have no contrast (NA) but 2 residual df each, which go into the prior and the pool (48, not 32); df.prior is infinite, so the cap binds",
  rbind(complete, b_only), "unpaired", a$design, a$labels, trend = FALSE, binds = TRUE, legacy = TRUE))

set.seed(808)
n_feat <- 150
level <- sort(runif(n_feat, 12, 27))
expr <- level + sqrt(trend_variance(level)) * matrix(rnorm(n_feat * 8), nrow = n_feat)
expr[seq(4, n_feat, by = 9), 5:8] <- expr[seq(4, n_feat, by = 9), 5:8] + 1.2
expr[seq(6, n_feat, by = 10), 1:4] <- NA
expr[cbind(seq(2, n_feat, by = 13), 5 + (seq(2, n_feat, by = 13) %% 4))] <- NA
cases <- c(cases, encode_case(
  "trend_missing_arm_legacy",
  "150 features, 4 vs 4; every 10th has arm A entirely missing (no contrast, 3 residual df, still in the trend and the pool), every 13th one NaN in arm B",
  expr, "unpaired", c4$design, c4$labels, trend = TRUE, binds = FALSE, legacy = TRUE))

# A paired design with subjects missing. A subject with no value at either timepoint leaves its
# dummy column empty, which is dropped; when it is the reference subject the intercept absorbs,
# the other dummies sum to the intercept and the LAST dummy is dropped instead. Either way the
# Post coefficient is estimable from the remaining subjects. A subject missing one timepoint
# keeps full rank and costs one residual df.
set.seed(613)
n_feat <- 120; n_subj <- 6
level <- sort(runif(n_feat, 13, 27))
subj <- rnorm(n_subj, sd = 1.5)
sd_f <- sqrt(trend_variance(level))
expr <- cbind(
  level + outer(rep(1, n_feat), subj) + sd_f * matrix(rnorm(n_feat * n_subj), nrow = n_feat),
  level + outer(rep(1, n_feat), subj) + sd_f * matrix(rnorm(n_feat * n_subj), nrow = n_feat))
moved <- seq(1, n_feat, by = 10)
expr[moved, (n_subj + 1):(2 * n_subj)] <- expr[moved, (n_subj + 1):(2 * n_subj)] - 0.8
expr[seq(3, n_feat, by = 9), c(1, n_subj + 1)] <- NA      # the reference subject, P1, absent
expr[seq(5, n_feat, by = 11), c(4, n_subj + 4)] <- NA     # P4 absent
expr[seq(7, n_feat, by = 8), 2] <- NA                     # P2 missing its Pre value only
cases <- c(cases, encode_case(
  "trend_paired_missing_subject_legacy",
  "120 features, 6 subjects Pre then Post, [1, Post, subject dummies]: every 9th feature lacks the reference subject P1 at both timepoints (the last dummy is aliased), every 11th lacks P4 (its dummy is empty), every 8th lacks P2's Pre value only; Post stays estimable throughout",
  expr, "paired", paired_design, list(subject = subject, time = time), trend = TRUE, binds = FALSE,
  legacy = TRUE))

# A within-subject trend, [1, week, subject dummies], with whole subjects missing for some features.
set.seed(1205)
n_feat <- 90
weeks <- rep(c(0, 2, 4, 6, 12), 4)
who <- rep(sprintf("S%d", 1:4), each = 5)
level <- sort(runif(n_feat, 12, 26))
person <- matrix(rnorm(n_feat * 4, sd = 1.2), nrow = n_feat)[, rep(1:4, each = 5)]
expr <- level + person + sqrt(trend_variance(level)) * matrix(rnorm(n_feat * 20), nrow = n_feat)
moved <- seq(1, n_feat, by = 9)
expr[moved, ] <- expr[moved, ] + outer(rep(1, length(moved)), -0.06 * weeks)
expr[seq(2, n_feat, by = 10), who == "S1"] <- NA
expr[seq(4, n_feat, by = 13), who == "S3"] <- NA
trend_design <- cbind(1, weeks, sapply(sprintf("S%d", 2:4), function(s) as.numeric(who == s)))
cases <- c(cases, encode_case(
  "trend_within_subject_missing_legacy",
  "90 features, 4 subjects at weeks 0, 2, 4, 6, 12, [1, week, subject dummies]: every 10th feature lacks the reference subject S1 entirely, every 13th lacks S3; the slope stays estimable from the others",
  expr, "linear_trend", trend_design, list(time = weeks, subject = who), trend = TRUE, binds = FALSE,
  legacy = TRUE))

# --- Residual variances of zero --------------------------------------------------------------------
#
# A feature whose replicates agree exactly within each arm has a residual variance of zero, which
# has no logarithm. fitFDist neither drops it nor fails: it floors every variance at 1e-5 x the
# median (warning "Zero sample variances detected") and fits the prior with it. The posterior still
# uses the feature's own zero. (Through lmFit's QR the residual may come out at 1e-30 rather than
# exactly 0; it is floored to the same value either way.) Dropping those features from the prior fit
# - what the toolkit did before 26.9.0 - changes the prior for every feature, so these cases fail it.
zero_rows <- function(levels, effects, n_a, n_b) {
  t(mapply(function(lv, e) c(rep(lv, n_a), rep(lv + e, n_b)), levels, effects))
}

set.seed(3030)
n_feat <- 40
noise <- 0.2 + 0.8 * ((0:(n_feat - 1)) / (n_feat - 1))
expr <- 16 + noise * matrix(rnorm(n_feat * 8), nrow = n_feat)
expr[1:4, 5:8] <- expr[1:4, 5:8] + 1.5
expr <- rbind(expr, zero_rows(c(12.5, 17.25, 21), c(0.75, -1.5, 0), 4, 4))
cases <- c(cases, encode_case(
  "global_zero_variance",
  "40 features plus 3 whose replicates agree exactly within each arm (residual variance 0), 4 vs 4: fitFDist floors every variance at 1e-5 x the median and fits the prior with them, rather than dropping them",
  expr, "unpaired", c4$design, c4$labels, trend = FALSE, binds = FALSE))

set.seed(3131)
n_feat <- 80
level <- sort(runif(n_feat, 12, 27))
expr <- level + sqrt(trend_variance(level)) * matrix(rnorm(n_feat * 8), nrow = n_feat)
expr[seq(5, n_feat, by = 11), 5:8] <- expr[seq(5, n_feat, by = 11), 5:8] + 1.2
expr <- rbind(expr, zero_rows(c(13, 19.5, 25.25), c(0.5, 0, -0.25), 4, 4))
cases <- c(cases, encode_case(
  "trend_zero_variance",
  "limma-trend over 80 features plus 3 with zero residual variance at low, middle and high Amean, 4 vs 4: the floored variances enter the spline fit and the prior df",
  expr, "unpaired", c4$design, c4$labels, trend = TRUE, binds = FALSE))

versions <- js_obj(list(
  R = js_str(paste(R.version$major, R.version$minor, sep = ".")),
  limma = js_str(as.character(packageVersion("limma")))
), 2)

json <- paste0(
  "{\n",
  "  \"reference\": \"R limma::lmFit + limma::eBayes(trend = TRUE | FALSE), called whole\",\n",
  "  \"note\": ", js_str(paste(
    "Each case's design is the one the toolkit builds for its analysis_type from the sample labels",
    "beside it, and the second coefficient is tested. df_total is limma's",
    "pmin(df.residual + df.prior, sum(df.residual)) and every p here is computed on it; df_pooled is",
    "sum(df.residual), recorded so a reader can see where the cap binds. s2_prior is the prior",
    "variance per feature (constant under the global prior).")), ",\n",
  "  \"versions\": ", versions, ",\n",
  "  \"cases\": [\n    ", paste(cases, collapse = ",\n    "), "\n  ]\n",
  "}\n")

writeLines(json, out_path, sep = "", useBytes = TRUE)
cat("wrote", out_path, "\n")
