#!/usr/bin/env Rscript
#
# Generates limma_trend_intensity_n_peptides.json: the reference for the
# limma-trend-intensity-n-peptides variance prior.
#
# That prior is limma-trend with a second covariate. limma's fitFDist regresses each feature's
# bias-corrected log residual variance on a natural spline of its mean log expression (Amean);
# this prior regresses it on that spline AND a natural spline of the protein's log peptide count,
# then estimates the prior df around the combined curve by the same trigamma inversion. limma
# itself cannot be called whole for this - fitFDist, squeezeVar and eBayes all take a single
# covariate vector - so the reference is built from limma's own pieces:
#
#   - lmFit gives the coefficients, residual variances (sigma^2), residual df and Amean;
#   - fit_fdist() below is limma 3.66.0's fitFDist, transcribed line for line, with the covariate
#     design matrix replaced by cbind(ns(Amean, df, intercept = TRUE), <count block>), where the
#     count block is ns(log(n_peptides), df = 2) - one interior knot, at the median - or, when the
#     counts take only two values, log(n_peptides) itself;
#   - the posterior variance is limma's .squeezeVar, and the p-value is read on eBayes's
#     df.total = pmin(df.residual + df.prior, sum(df.residual)).
#
# Because the middle step is a transcription, every case first runs it WITHOUT the count block and
# requires it to reproduce limma's own eBayes(trend = TRUE) - prior variance, prior df, posterior
# variance and p-value - to 1e-12. A case whose transcription disagrees with limma is refused, so
# the only thing this file adds to limma is the second column block.
#
# The same generator, cases and seeds live in proteomics-toolkit
# (tests/fixtures/generate_limma_trend_intensity_n_peptides.R), and the two write byte-identical
# JSON: Skyline-PRISM's VariancePrior.LimmaTrendIntensityNPeptides and the toolkit's
# moderation="limma-trend-intensity-n-peptides" are pinned to one reference, which is how the two
# tools are held to the same numbers. Change a case in both, or neither.
#
# Nothing here imports Skyline-PRISM or the toolkit. Each matrix and its peptide counts are drawn
# here and written into the fixture, and the tests hand them to the code under test.
#
# Prerequisites (one-off):
#   install.packages("BiocManager")
#   BiocManager::install("limma")
#
# Run from the repository root (on Windows, Rscript.exe is in the R installation's bin folder):
#   Rscript dotnet/tests/fixtures/differential/generate_limma_trend_intensity_n_peptides.R
#
# Generated with R 4.5.1 and limma 3.66.0; the versions that ran are written into the fixture.

suppressPackageStartupMessages(library(limma))

out_path <- file.path("dotnet", "tests", "fixtures", "differential", "limma_trend_intensity_n_peptides.json")
if (!dir.exists(dirname(out_path))) stop("run from the repository root: ", dirname(out_path), " not found")

# Floats are JSON strings of 17 significant digits, as in limma_ebayes.json.
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
js_ints <- function(x) paste0("[", paste(as.integer(x), collapse = ", "), "]")
js_labels <- function(x) paste0("[", paste(js_str(as.character(x)), collapse = ", "), "]")
js_mat <- function(m, indent) {
  m <- as.matrix(m)
  rows <- vapply(seq_len(nrow(m)), function(i) js_vec(m[i, ]), "")
  pad <- strrep(" ", indent + 2)
  paste0("[\n", paste0(pad, rows, collapse = ",\n"), "\n", strrep(" ", indent), "]")
}
js_obj <- function(fields, indent) {
  pad <- strrep(" ", indent + 2)
  body <- paste0(pad, js_str(names(fields)), ": ", unlist(fields), collapse = ",\n")
  paste0("{\n", body, "\n", strrep(" ", indent), "}")
}

# ------------------------------------------------------------------------------------------------
# limma 3.66.0's fitFDist, for complete data (every variance finite and every df positive, which
# the cases guarantee), with `count_block` as an optional second block of design columns. With
# count_block = NULL it is fitFDist(x, df1, covariate = amean) exactly.
# ------------------------------------------------------------------------------------------------

fit_fdist <- function(x, df1, amean, count_block = NULL) {
  nok <- length(x)
  stopifnot(all(is.finite(x)), all(x > -1e-15), all(is.finite(df1)), all(df1 > 1e-15))
  splinedf <- 1L + (nok >= 3L) + (nok >= 6L) + (nok >= 30L)
  splinedf <- min(splinedf, length(unique(amean)))
  stopifnot(splinedf >= 2L)
  x <- pmax(x, 0)
  m <- median(x)
  stopifnot(m > 0, all(x > 0))
  x <- pmax(x, 1e-05 * m)
  z <- log(x)
  e <- z + log(df1 / 2) - digamma(df1 / 2)  # limma's logmdigamma(df1/2)
  design <- splines::ns(amean, df = splinedf, intercept = TRUE)
  if (!is.null(count_block)) design <- cbind(design, count_block)
  fit <- lm.fit(design, e)
  stopifnot(fit$rank == ncol(design))
  emean <- fit$fitted.values
  evar <- mean(fit$effects[-(1:fit$rank)]^2)
  evar <- evar - mean(trigamma(df1 / 2))
  if (evar > 0) {
    df2 <- 2 * trigammaInverse(evar)
    s20 <- exp(emean - (log(df2 / 2) - digamma(df2 / 2)))
  } else {
    df2 <- Inf
    s20 <- exp(emean)
  }
  list(scale = s20, df2 = df2, splinedf = splinedf, ncol = ncol(design))
}

# limma's .squeezeVar posterior, and eBayes's moderated t on the capped df.
moderate <- function(fit, prior, coef = 2) {
  s2 <- fit$sigma^2
  df <- fit$df.residual
  s2_post <- if (is.finite(prior$df2)) (df * s2 + prior$df2 * prior$scale) / (df + prior$df2) else prior$scale
  t <- fit$coefficients[, coef] / fit$stdev.unscaled[, coef] / sqrt(s2_post)
  df_total <- pmin(df + prior$df2, sum(df))
  list(s2_post = s2_post, t = t, df_total = df_total, p = 2 * pt(-abs(t), df = df_total))
}

count_block_for <- function(n_peptides) {
  log_n <- log(n_peptides)
  if (length(unique(log_n)) >= 3L) splines::ns(log_n, df = 2) else matrix(log_n, ncol = 1)
}

encode_case <- function(name, note, expr, n_peptides, analysis_type, design, labels, binds,
                        count_cols, indent = 4) {
  stopifnot(nrow(expr) == length(n_peptides), all(n_peptides >= 1), all(n_peptides == round(n_peptides)))
  fit <- lmFit(expr, design)
  s2 <- fit$sigma^2
  df <- fit$df.residual

  # The transcription, without the count block, must be limma's own limma-trend.
  ref <- eBayes(fit, trend = TRUE)
  plain <- fit_fdist(s2, df, fit$Amean)
  plain_mod <- moderate(fit, plain)
  close <- function(a, b) isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = 1e-12))
  if (!(close(plain$scale, ref$s2.prior) && identical(is.finite(plain$df2), is.finite(ref$df.prior[1])) &&
        (!is.finite(plain$df2) || close(plain$df2, ref$df.prior[1])) &&
        close(plain_mod$s2_post, ref$s2.post) && close(plain_mod$p, ref$p.value[, 2])))
    stop(name, ": the transcribed fitFDist does not reproduce limma's eBayes(trend = TRUE)")

  block <- count_block_for(n_peptides)
  if (ncol(block) != count_cols) stop(sprintf("%s: expected %d count columns, got %d", name, count_cols, ncol(block)))
  prior <- fit_fdist(s2, df, fit$Amean, block)
  mod <- moderate(fit, prior)
  if (!is.finite(prior$df2) && !is.finite(ref$df.prior[1]))
    stop(name, ": limma-trend's df.prior is infinite too, so this case does not show the count term at work")
  bound <- any(df + prior$df2 > sum(df))
  if (binds != bound) stop(sprintf("%s: expected the df cap to %sbind (df.prior %s)", name, if (binds) "" else "not ", num(prior$df2)))

  cat(sprintf("  %-26s features %4d, df.residual %s, spline df %d + %d count, df.prior %s (limma-trend's %s)\n",
              name, nrow(expr), paste(unique(df), collapse = "/"), prior$splinedf, ncol(block),
              num(prior$df2), num(ref$df.prior[1])))

  js_obj(c(
    list(name = js_str(name), note = js_str(note), analysis_type = js_str(analysis_type)),
    lapply(labels, js_labels),
    list(
      expr = js_mat(expr, indent + 2),
      n_peptides = js_ints(n_peptides),
      spline_df = as.character(prior$splinedf),
      count_cols = as.character(ncol(block)),
      df_residual = js_vec(df),
      df_prior = js_str(num(prior$df2)),
      df_prior_limma_trend = js_str(num(ref$df.prior[1])),
      df_pooled = js_str(num(sum(df))),
      df_total = js_vec(mod$df_total),
      s2_prior = js_vec(prior$scale),
      s2_post = js_vec(mod$s2_post),
      amean = js_vec(fit$Amean),
      logfc = js_vec(fit$coefficients[, 2]),
      t = js_vec(mod$t),
      p = js_vec(mod$p)
    )
  ), indent)
}

two_arm <- function(n_a, n_b) {
  list(design = cbind(1, c(rep(0, n_a), rep(1, n_b))), labels = list(group = c(rep("A", n_a), rep("B", n_b))))
}

# Peptide counts log-uniform on 1..64 (median about 8), and a protein's variance: technical noise
# that falls with intensity and is divided by the count (scaled to match at 8 peptides), plus
# biology that does not depend on the count, times a scaled inverse chi-square so the true prior
# df is finite.
peptide_counts <- function(n) pmax(1L, as.integer(round(2^runif(n, 0, 6))))
protein_variance <- function(level, n_peptides, d_true = 6) {
  ((0.03 + 0.6 * exp(-(level - 12) / 3)) * 8 / n_peptides + 0.05) * d_true / rchisq(length(level), d_true)
}

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
cases <- character()

set.seed(1019)
n_feat <- 300
level <- sort(runif(n_feat, 12, 28))
n_pep <- peptide_counts(n_feat)
expr <- level + sqrt(protein_variance(level, n_pep)) * matrix(rnorm(n_feat * 8), nrow = n_feat)
moved <- seq(1, n_feat, by = 15)
expr[moved, 5:8] <- expr[moved, 5:8] + 1
c4 <- two_arm(4, 4)
cases <- c(cases, encode_case(
  "protein_unpaired",
  "300 proteins over 12..28 log2 with 1..64 peptides, 4 vs 4, variance falling with intensity and as 1/count plus a flat part: spline df 4 plus 2 count columns, finite df.prior",
  expr, n_pep, "unpaired", c4$design, c4$labels, binds = FALSE, count_cols = 2))

set.seed(613)
n_feat <- 120; n_subj <- 6
level <- sort(runif(n_feat, 13, 27))
n_pep <- peptide_counts(n_feat)
subj <- rnorm(n_subj, sd = 1.5)
sd_f <- sqrt(protein_variance(level, n_pep))
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
  "protein_paired",
  "120 proteins, 6 subjects Pre then Post, design [1, Post, subject dummies]: spline df 4 plus 2 count columns, finite df.prior",
  expr, n_pep, "paired", paired_design, list(subject = subject, time = time), binds = FALSE, count_cols = 2))

set.seed(1205)
n_feat <- 80
weeks <- rep(c(0, 2, 4, 6, 12), 3)
level <- sort(runif(n_feat, 12, 26))
n_pep <- peptide_counts(n_feat)
expr <- level + sqrt(protein_variance(level, n_pep)) * matrix(rnorm(n_feat * length(weeks)), nrow = n_feat)
moved <- seq(1, n_feat, by = 8)
expr[moved, ] <- expr[moved, ] + outer(rep(1, length(moved)), -0.08 * weeks)
cases <- c(cases, encode_case(
  "protein_linear_trend",
  "80 proteins at weeks 0, 2, 4, 6, 12 (three samples each), design [1, week], the slope tested: spline df 4 plus 2 count columns",
  expr, n_pep, "linear_trend", cbind(1, weeks), list(time = weeks), binds = FALSE, count_cols = 2))

set.seed(2027)
level <- sort(runif(20, 12, 26))
n_pep <- peptide_counts(20)
a <- two_arm(3, 3)
expr <- level + sqrt(protein_variance(level, n_pep, d_true = 4)) * matrix(rnorm(20 * 6), nrow = 20)
expr[1:4, 4:6] <- expr[1:4, 4:6] + 1.5
cases <- c(cases, encode_case(
  "protein_twenty",
  "20 proteins, 3 vs 3: limma's spline df 3 for Amean (one interior knot) plus 2 count columns",
  expr, n_pep, "unpaired", a$design, a$labels, binds = FALSE, count_cols = 2))

# Two distinct counts: no spline can be placed in log count, so the count block is log count itself.
set.seed(77)
n_feat <- 60
level <- sort(runif(n_feat, 12, 26))
n_pep <- sample(c(1L, 6L), n_feat, replace = TRUE)
expr <- level + sqrt(protein_variance(level, n_pep)) * matrix(rnorm(n_feat * 8), nrow = n_feat)
expr[seq(2, n_feat, by = 9), 5:8] <- expr[seq(2, n_feat, by = 9), 5:8] + 1.2
cases <- c(cases, encode_case(
  "protein_two_count_values",
  "60 proteins with 1 or 6 peptides, 4 vs 4: the count block is the single column log(n_peptides)",
  expr, n_pep, "unpaired", c4$design, c4$labels, binds = FALSE, count_cols = 1))

# Built so the log residual variance is exactly linear in log count and constant in Amean: every
# row has the same residual pattern, scaled by 1/sqrt(n_peptides). The two-covariate regression
# then leaves nothing (evar <= 0), so df.prior is infinite and the cap binds at the pooled
# 12 x 4 = 48 - while limma-trend alone, which cannot see the counts, gets a finite df.prior: the
# alternating counts leave log(16)^2 / 4 = 1.92 of spread about its Amean spline, well past the
# trigamma(2) = 0.645 sampling variance.
lv <- 8:19
effects <- c(0.4, -0.6, 1.2, 0.0, -1.5, 0.8, 2.0, -0.3, 0.6, -1.0, 1.7, 0.2)
n_pep <- rep(c(1L, 16L), 6)
spread <- 0.5 / sqrt(n_pep)
expr <- t(sapply(seq_along(lv), function(i) {
  s <- spread[i]; e <- effects[i]; l <- lv[i]
  c(l - s, l, l + s, l + e - s, l + e, l + e + s)
}))
cases <- c(cases, encode_case(
  "protein_infinite_prior",
  "12 proteins with 1 or 16 peptides, 3 vs 3, residual variance exactly 0.25 / n_peptides: the count term explains it all, df.prior is infinite and df.total is the pooled 48; limma-trend's df.prior is finite",
  expr, n_pep, "unpaired", a$design, a$labels, binds = TRUE, count_cols = 1))

versions <- js_obj(list(
  R = js_str(paste(R.version$major, R.version$minor, sep = ".")),
  limma = js_str(as.character(packageVersion("limma")))
), 2)

json <- paste0(
  "{\n",
  "  \"reference\": ", js_str(paste(
    "R limma 3.66.0: lmFit for coefficients, sigma, df.residual and Amean; limma::fitFDist's algebra",
    "transcribed with the covariate design cbind(splines::ns(Amean, df, intercept = TRUE),",
    "splines::ns(log(n_peptides), df = 2) or log(n_peptides)), fitted by stats::lm.fit, df.prior by",
    "limma::trigammaInverse; posterior as limma's .squeezeVar; p on eBayes's capped df.total.",
    "Without the count block the transcription reproduces limma::eBayes(trend = TRUE) to 1e-12 on",
    "every case, which the generator checks before writing.")), ",\n",
  "  \"note\": ", js_str(paste(
    "Each case's design is the one the toolkit builds for its analysis_type from the sample labels",
    "beside it, and the second coefficient is tested. spline_df is limma's rule for the Amean spline;",
    "count_cols is 2 for a natural spline of log count with one interior knot at the median, 1 when",
    "the counts take only two values. df_prior_limma_trend is limma-trend's prior df on the same",
    "case, for comparison.")), ",\n",
  "  \"versions\": ", versions, ",\n",
  "  \"cases\": [\n    ", paste(cases, collapse = ",\n    "), "\n  ]\n",
  "}\n")

writeLines(json, out_path, sep = "", useBytes = TRUE)
cat("wrote", out_path, "\n")
