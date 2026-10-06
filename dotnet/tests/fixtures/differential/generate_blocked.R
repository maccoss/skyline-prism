#!/usr/bin/env Rscript
#
# Generates blocked.json: the golden for the subject-blocked moderated design, produced by
# limma's own duplicateCorrelation + lmFit(block=, correlation=) + eBayes.
#
# Why R and not the Python generator beside it: inmoose 0.9.1 (the limma port generate.py uses)
# has no duplicateCorrelation and no blocked lmFit, so there is no Python reference for this
# estimator. limma itself is the definition.
#
# What the blocked design is for: a contrast whose groups are constant within subject (sex, onset
# site) when subjects contribute several samples. A fixed subject effect cannot be fitted there -
# subject is nested in the group and absorbs the contrast - so the repeated samples are modelled
# as correlated instead: one intra-subject correlation shared by every feature (the trimmed mean
# of per-feature REML estimates, on the atanh scale), then generalized least squares with that
# correlation, then the usual empirical-Bayes moderation. Smyth, Michaud & Scott 2005,
# Bioinformatics 21(9):2067-2075, doi:10.1093/bioinformatics/bti270.
#
# Prerequisites (one-off; statmod is installed with limma, which imports it):
#   install.packages("BiocManager")
#   BiocManager::install("limma")
#
# Run from the repository root:
#   Rscript dotnet/tests/fixtures/differential/generate_blocked.R
#
# Windows:
#   & "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" dotnet/tests/fixtures/differential/generate_blocked.R
#
# Generated with R 4.6.1, limma 3.68.5, statmod 1.5.2 (Bioconductor 3.23). The versions that
# actually ran are written into the fixture's `versions` field, so a regeneration under different
# ones shows up in the diff.

suppressPackageStartupMessages(library(limma))

out_path <- file.path("dotnet", "tests", "fixtures", "differential", "blocked.json")
if (!dir.exists(dirname(out_path))) stop("run from the repository root: ", dirname(out_path), " not found")

# ------------------------------------------------------------------------------------------------
# Encoding - the same contract as generate.py: every float is a JSON STRING, and the non-finite
# ones are spelled the way Python spells them ("inf", "-inf", "nan"), because that is what
# DifferentialGoldenTests maps. R's "%.17g" is not Python's shortest repr, but 17 significant
# digits round-trip a double exactly, so double.Parse recovers the same 64 bits either way.
# NA (R's missing) is written as "nan": the C# side has one spelling of missing.
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
js_strvec <- function(s) paste0("[", paste(js_str(as.character(s)), collapse = ", "), "]")
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
# The reference: limma, called exactly as a user of limma would call it.
# ------------------------------------------------------------------------------------------------

reference <- function(expr, design, block) {
  dc <- duplicateCorrelation(expr, design, block = block)
  fit <- lmFit(expr, design, block = block, correlation = dc$consensus.correlation)
  eb <- eBayes(fit)
  ebt <- eBayes(fit, trend = TRUE)

  # How far limma's per-feature estimate is from the converged REML optimum. duplicateCorrelation
  # caps statmod's Fisher scoring at maxit = 20 with an absolute tol of 1e-6 on the score step, so
  # an implementation that maximizes the same REML likelihood by another route agrees with
  # `atanh_correlations` only to roughly this gap - and one that replays statmod's iterations
  # agrees far more tightly. Both are recorded so a test can say which it is holding PRISM to.
  # This replays duplicateCorrelation's per-feature loop with only the solver settings changed;
  # the bounds below are limma's (rhomin = 1/(1 - max block size) + 0.01, rhomax = 0.99).
  rho_conv <- rep(NA_real_, nrow(expr))
  for (i in seq_len(nrow(expr))) {
    y <- expr[i, ]
    o <- is.finite(y)
    A <- factor(block[o])
    if (sum(o) > ncol(design) + 2L && nlevels(A) > 1L && nlevels(A) < sum(o) - 1L) {
      s <- tryCatch(suppressWarnings(statmod::mixedModel2Fit(
        y[o], design[o, , drop = FALSE], model.matrix(~0 + A),
        only.varcomp = TRUE, tol = 1e-14, maxit = 1000)$varcomp), error = function(e) NA)
      if (!is.na(s[1])) rho_conv[i] <- s[2] / sum(s)
    }
  }
  rhomin <- 1 / (1 - max(table(block))) + 0.01
  rho_conv <- pmin(pmax(rho_conv, rhomin), 0.99)

  list(dc = dc, fit = fit, eb = eb, ebt = ebt, atanh_conv = atanh(rho_conv))
}

encode_case <- function(name, note, expr, design, block, indent = 4) {
  r <- reference(expr, design, block)
  i2 <- indent + 2
  fields <- list(
    name = js_str(name),
    note = js_str(note),
    block = js_strvec(block),
    coef_names = js_strvec(colnames(design)),
    expr = js_mat(expr, i2),
    design = js_mat(design, i2),
    atanh_correlations = js_vec(r$dc$atanh.correlations),
    atanh_correlations_converged = js_vec(r$atanh_conv),
    consensus_correlation = js_str(num(r$dc$consensus.correlation)),
    coefficients = js_mat(r$fit$coefficients, i2),
    stdev_unscaled = js_mat(r$fit$stdev.unscaled, i2),
    sigma = js_vec(r$fit$sigma),
    df_residual = js_vec(r$fit$df.residual),
    amean = js_vec(r$fit$Amean),
    global = js_obj(list(
      df_prior = js_str(num(r$eb$df.prior)),
      s2_prior = js_vec(r$eb$s2.prior),
      t = js_mat(r$eb$t, i2 + 2),
      p = js_mat(r$eb$p.value, i2 + 2)
    ), i2),
    trend = js_obj(list(
      df_prior = js_str(num(r$ebt$df.prior)),
      s2_prior = js_vec(r$ebt$s2.prior),
      t = js_mat(r$ebt$t, i2 + 2),
      p = js_mat(r$ebt$p.value, i2 + 2)
    ), i2)
  )
  cat(sprintf("  %-28s consensus rho = %.6f, df_prior = %s, NA rho = %d\n", name,
              r$dc$consensus.correlation, format(r$eb$df.prior), sum(is.na(r$dc$atanh.correlations))))
  js_obj(fields, indent)
}

# Simulates features x samples on the log2 scale: a per-feature level, a per-subject random
# intercept (sd tau), residual noise (sd sigma), and `effect` added to the samples where `shift`
# is TRUE for the first `n_de` features. Subjects are the block.
simulate <- function(n_features, block, shift, tau, sigma, effect, n_de) {
  subjects <- unique(block)
  expr <- matrix(0, n_features, length(block),
                 dimnames = list(sprintf("F%03d", seq_len(n_features)), sprintf("s%02d", seq_along(block))))
  for (f in seq_len(n_features)) {
    u <- setNames(rnorm(length(subjects), 0, tau[f]), subjects)
    expr[f, ] <- 14 + 0.08 * f + u[block] + rnorm(length(block), 0, sigma[f]) +
      ifelse(f <= n_de & shift, effect, 0)
  }
  expr
}

visits <- function(prefix, counts) rep(sprintf("%s%02d", prefix, seq_along(counts)), times = counts)

set.seed(20261006)
cases <- character(0)

# --- 1. The shape this design exists for: subject nested in a two-level group --------------------
{
  block <- c(visits("F", c(3, 4, 4, 5, 4, 3, 5)), visits("M", c(5, 4, 6, 3, 4, 5, 4, 4)))
  grp <- as.numeric(startsWith(block, "M"))
  nf <- 60
  sigma <- 0.25 + 0.5 * runif(nf)
  expr <- simulate(nf, block, grp == 1, tau = sigma, sigma = sigma, effect = 1.0, n_de = 6)
  design <- cbind(Intercept = 1, groupB = grp)
  cases <- c(cases, encode_case(
    "nested_unequal_visits",
    paste("7 subjects (28 samples) vs 8 subjects (35), 3-6 visits each, every subject in one",
          "group only; true intra-subject correlation 0.5 in every feature, a 1.0 log2 shift in",
          "the first 6. The case a fixed subject block cannot fit."),
    expr, design, block))
}

# --- 2. Singleton subjects, and a numeric covariate beside the group -----------------------------
{
  block <- c(visits("A", c(1, 3, 2, 1, 4)), visits("B", c(2, 1, 3, 1, 3)))
  grp <- as.numeric(startsWith(block, "B"))
  age <- setNames(round(runif(10, 40, 75)), unique(block))[block]
  nf <- 30
  sigma <- 0.3 + 0.4 * runif(nf)
  expr <- simulate(nf, block, grp == 1, tau = 0.8 * sigma, sigma = sigma, effect = 0.8, n_de = 4)
  expr <- expr + outer(0.01 * seq_len(nf), age - mean(age))
  design <- cbind(Intercept = 1, groupB = grp, age = age - mean(age))
  cases <- c(cases, encode_case(
    "singletons_and_covariate",
    paste("Four subjects contribute one sample (a block of size 1), and a subject-level numeric",
          "covariate, mean-centered as PRISM centers one, sits beside the group column."),
    expr, design, block))
}

# --- 3. Missing values: limma's per-feature GLS path ----------------------------------------------
{
  block <- c(visits("A", c(3, 4, 3, 5, 4)), visits("B", c(4, 3, 5, 3, 4)))
  grp <- as.numeric(startsWith(block, "B"))
  nf <- 25
  sigma <- 0.3 + 0.4 * runif(nf)
  expr <- simulate(nf, block, grp == 1, tau = sigma, sigma = sigma, effect = 0.9, n_de = 3)
  expr[cbind(sample(nf, 20, replace = TRUE), sample(ncol(expr), 20, replace = TRUE))] <- NA
  expr[5, block == "A02"] <- NA                       # one subject missing entirely from a feature
  expr[9, -c(1, 2, 20, 21)] <- NA                     # too few values for a correlation estimate
  design <- cbind(Intercept = 1, groupB = grp)
  cases <- c(cases, encode_case(
    "missing_values",
    paste("Scattered NA cells, so lmFit takes its per-feature GLS loop (the correlation matrix",
          "subset to each feature's observed samples) instead of one shared transform. F005 lacks",
          "every sample of subject A02; F009 keeps 4 values, too few for duplicateCorrelation, so its",
          "correlation is NA and excluded from the trimmed mean while its GLS fit still runs."),
    expr, design, block))
}

# --- 4. Time within subject, group between subjects, and their interaction -----------------------
{
  counts_a <- c(4, 3, 5, 4, 3, 4)
  counts_b <- c(3, 5, 4, 4, 5, 3)
  block <- c(visits("A", counts_a), visits("B", counts_b))
  grp <- as.numeric(startsWith(block, "B"))
  time <- unlist(lapply(c(counts_a, counts_b), function(k) seq(0, k - 1))) * 3   # months
  nf <- 40
  sigma <- 0.25 + 0.4 * runif(nf)
  expr <- simulate(nf, block, grp == 1, tau = 1.2 * sigma, sigma = sigma, effect = 0.6, n_de = 4)
  expr[1:5, ] <- expr[1:5, ] + outer(rep(0.05, 5), time * grp)    # slope differs by group
  design <- cbind(Intercept = 1, groupB = grp, time = time, groupB_time = grp * time)
  cases <- c(cases, encode_case(
    "time_by_group",
    paste("time + group + time:group with subject as the block - the combined within- and",
          "between-subject question. Every coefficient is pinned; the interaction (column 4) is",
          "the one a slope-difference test reads. A slope difference is planted in the first 5."),
    expr, design, block))
}

# --- 5. The correlation bounds: no subject effect, and an overwhelming one ------------------------
{
  block <- c(visits("A", c(8, 6, 7, 5)), visits("B", c(6, 8, 5, 7)))
  grp <- as.numeric(startsWith(block, "B"))
  nf <- 40
  sigma <- rep(0.4, nf)
  tau <- c(rep(0, 36), rep(6, 4))
  sigma[37:40] <- 0.02
  expr <- simulate(nf, block, grp == 1, tau = tau, sigma = sigma, effect = 0, n_de = 0)
  design <- cbind(Intercept = 1, groupB = grp)
  cases <- c(cases, encode_case(
    "correlation_bounds",
    paste("No subject effect in F001-F036, so some REML estimates fall below limma's floor",
          "1/(1 - 8) + 0.01 and are raised to it; an overwhelming one in F037-F040 (sd 6 against",
          "residual 0.02), so those exceed 0.99 and are lowered to it. Null group effect."),
    expr, design, block))
}

# --- 6. The two degenerate blocks duplicateCorrelation answers with zero --------------------------
degenerate <- character(0)
{
  set.seed(7)
  expr <- matrix(rnorm(5 * 8, 10, 0.5), 5, 8)
  design <- cbind(Intercept = 1, groupB = rep(0:1, each = 4))
  dc1 <- suppressWarnings(duplicateCorrelation(expr, design, block = sprintf("S%d", 1:8)))
  subj <- rep(sprintf("S%d", 1:4), 2)
  design2 <- cbind(design, model.matrix(~ factor(subj))[, -1])
  dc2 <- suppressWarnings(duplicateCorrelation(expr, design2, block = subj))
  for (d in list(list("all_blocks_size_one", "every sample its own subject", design, sprintf("S%d", 1:8), dc1),
                 list("block_in_design", "subject dummies already in the design (the paired design)", design2, subj, dc2))) {
    degenerate <- c(degenerate, js_obj(list(
      name = js_str(d[[1]]),
      note = js_str(paste0(d[[2]], "; limma warns and sets the correlation to 0")),
      block = js_strvec(d[[4]]),
      expr = js_mat(expr, 6),
      design = js_mat(d[[3]], 6),
      consensus_correlation = js_str(num(d[[5]]$consensus.correlation))
    ), 4))
  }
}

versions <- js_obj(list(
  R = js_str(paste(R.version$major, R.version$minor, sep = ".")),
  limma = js_str(as.character(packageVersion("limma"))),
  statmod = js_str(as.character(packageVersion("statmod")))
), 2)

json <- paste0(
  "{\n",
  "  \"reference\": ", js_str(paste(
    "limma::duplicateCorrelation(expr, design, block = block), then",
    "limma::lmFit(expr, design, block = block, correlation = consensus), then",
    "limma::eBayes(fit) and limma::eBayes(fit, trend = TRUE)")), ",\n",
  "  \"note\": ", js_str(paste(
    "`expr` is features x samples (PRISM's orientation), `design` is samples x coef, `block` names",
    "each sample's subject. Matrices of results are features x coef. `atanh_correlations` are",
    "limma's per-feature values after its bounds are applied (nan where limma estimates none);",
    "`consensus_correlation` is tanh of their 15%-trimmed mean. `atanh_correlations_converged`",
    "re-solves each feature's REML fit to tol 1e-14 / maxit 1000 instead of limma's 1e-6 / 20,",
    "so the gap between the two is how tightly an independent REML optimizer can be held to limma.",
    "`global` is eBayes with a constant prior, `trend` with limma-trend's prior on Amean.")), ",\n",
  "  \"versions\": ", versions, ",\n",
  "  \"cases\": [\n    ", paste(cases, collapse = ",\n    "), "\n  ],\n",
  "  \"degenerate\": [\n    ", paste(degenerate, collapse = ",\n    "), "\n  ]\n",
  "}\n")

# Binary, so Windows does not turn every newline into CRLF.
writeBin(charToRaw(json), out_path)
cat("wrote", out_path, "\n")
