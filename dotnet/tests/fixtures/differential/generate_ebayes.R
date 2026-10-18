#!/usr/bin/env Rscript
#
# Generates ebayes.json: the moderated t from limma's own lmFit + eBayes, called whole.
#
# What it pins that moderated_t.json does not: moderated_t.json composes limma's steps by hand from
# pinned references (lstsq, inmoose's squeezeVar, scipy's t), so it pins each step - and a hand
# composition can leave a step out. One did: limma's eBayes caps the moderated t's degrees of
# freedom at the residual df pooled over every feature,
#
#   df.total <- pmin(df.residual + df.prior, sum(df.residual))
#
# and both that composition and PRISM used df.residual + df.prior, so they agreed with each other
# and not with limma. The cap binds whenever the prior df is infinite - uncapped, the t would be read
# off a normal - and otherwise only when few features are tested and the prior df is large. Its
# effect is largest on few features, so every binding case here is small, plus one ordinary case
# where the cap must change nothing.
#
# Why R and not generate.py: inmoose 0.9.1's eBayes applies the same cap, but crashes whenever
# df.prior is infinite (its B-statistic step applies `~` to a Python bool, giving -2, and indexes a
# DataFrame with it), which is exactly the case that matters most. limma itself is the definition.
#
# Prerequisites (one-off):
#   install.packages("BiocManager")
#   BiocManager::install("limma")
#
# Run from the repository root (on Windows, Rscript.exe is in the R installation's bin folder):
#   Rscript dotnet/tests/fixtures/differential/generate_ebayes.R
#
# Generated with R 4.5.1 and limma 3.66.0. The versions that actually ran are written into the
# fixture's `versions` field, so a regeneration under different ones shows up in the diff.

suppressPackageStartupMessages(library(limma))

out_path <- file.path("dotnet", "tests", "fixtures", "differential", "ebayes.json")
if (!dir.exists(dirname(out_path))) stop("run from the repository root: ", dirname(out_path), " not found")

# ------------------------------------------------------------------------------------------------
# Encoding - the same contract as generate.py: every float is a JSON STRING, and the non-finite
# ones are spelled the way Python spells them ("inf", "-inf", "nan"), because that is what
# DifferentialGoldenTests maps. R's "%.17g" is not Python's shortest repr, but 17 significant
# digits round-trip a double exactly, so double.Parse recovers the same 64 bits either way.
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
# One case: two arms, design [1, groupB], samples ordered [A..., B...], coefficient 2 tested.
# `binds` says whether the cap should bind; a case that does not do what its name says is refused
# rather than written, so the golden cannot quietly stop exercising the cap.
# ------------------------------------------------------------------------------------------------

encode_case <- function(name, note, expr, n_a, trend, binds, indent = 4) {
  n <- ncol(expr)
  design <- cbind(1, c(rep(0, n_a), rep(1, n - n_a)))
  fit <- eBayes(lmFit(expr, design), trend = trend)

  df_residual <- fit$df.residual[1]
  df_prior <- fit$df.prior[1]
  df_pooled <- sum(fit$df.residual)
  if (binds != (df_residual + df_prior > df_pooled))
    stop(sprintf("%s: expected the cap to %sbind; df.residual %g + df.prior %g vs pooled %g",
                 name, if (binds) "" else "not ", df_residual, df_prior, df_pooled))
  stopifnot(all(fit$df.total == pmin(fit$df.residual + fit$df.prior, df_pooled)))

  cat(sprintf("  %-22s df.residual %g, df.prior %s, pooled %g, df.total %g\n",
              name, df_residual, num(df_prior), df_pooled, fit$df.total[1]))

  js_obj(list(
    name = js_str(name),
    note = js_str(note),
    expr = js_mat(expr, indent + 2),
    n_a = as.character(n_a),
    n_b = as.character(n - n_a),
    trend = if (trend) "true" else "false",
    df_residual = js_str(num(df_residual)),
    df_prior = js_str(num(df_prior)),
    df_pooled = js_str(num(df_pooled)),
    df_total = js_str(num(fit$df.total[1])),
    logfc = js_vec(fit$coefficients[, 2]),
    t = js_vec(fit$t[, 2]),
    p = js_vec(fit$p.value[, 2])
  ), indent)
}

# Every row shares one residual pattern, so every residual variance is the same: fitFDist's evar
# is <= 0 and the prior df is infinite. Rows differ in level and in effect, so t differs.
equal_variance_rows <- function(levels, effects, spread = 0.5) {
  t(mapply(function(lv, e) c(lv - spread, lv, lv + spread, lv + e - spread, lv + e, lv + e + spread),
           levels, effects))
}

cases <- character()

cases <- c(cases, encode_case(
  "infinite_prior",
  "8 features, 3 vs 3, equal residual variances: df.prior is infinite and df.total is the pooled 8 x 4 = 32, not infinity",
  equal_variance_rows(9:16, c(0.2, -0.4, 0.9, 1.6, -1.1, 0.05, 2.3, -0.7)),
  n_a = 3, trend = FALSE, binds = TRUE))

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
  "large_finite_prior",
  "5 features, 3 vs 3, residual variances spread just past their sampling variance: df.prior is finite, but df.residual + df.prior exceeds the pooled 20",
  finite, n_a = 3, trend = FALSE, binds = TRUE))

cases <- c(cases, encode_case(
  "trend_infinite_prior",
  "limma-trend over 12 features spanning 8..19 log2 with equal residual variances: the trend is flat, df.prior is infinite and df.total is the pooled 12 x 4 = 48",
  equal_variance_rows(8:19, c(0.4, -0.6, 1.2, 0.0, -1.5, 0.8, 2.0, -0.3, 0.6, -1.0, 1.7, 0.2)),
  n_a = 3, trend = TRUE, binds = TRUE))

# The ordinary case - many features, noise that differs between them, a small prior df. The cap
# must change nothing here. The generator is pinned explicitly so a regeneration draws the same
# matrix; the matrix itself is in the fixture, so the C# side never depends on it.
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(271)
n_a <- 6; n_b <- 7
noise <- 0.15 + 1.5 * ((0:39) / 39)^2
many <- 18 + noise * matrix(rnorm(40 * (n_a + n_b)), nrow = 40)
many[1:5, (n_a + 1):(n_a + n_b)] <- many[1:5, (n_a + 1):(n_a + n_b)] + 1.7
cases <- c(cases, encode_case(
  "cap_does_not_bind",
  "40 features, 6 vs 7, heterogeneous noise: df.residual + df.prior is far below the pooled 440, so the cap changes nothing",
  many, n_a = n_a, trend = FALSE, binds = FALSE))

versions <- js_obj(list(
  R = js_str(paste(R.version$major, R.version$minor, sep = ".")),
  limma = js_str(as.character(packageVersion("limma")))
), 2)

json <- paste0(
  "{\n",
  "  \"reference\": \"R limma::lmFit + limma::eBayes, called whole\",\n",
  "  \"note\": ", js_str(paste(
    "Two arms, design [1, groupB], samples ordered [A..., B...]; the group coefficient is tested.",
    "df_total is limma's pmin(df.residual + df.prior, sum(df.residual)) and every p here is",
    "computed on it; df_pooled is sum(df.residual), recorded so a reader can see where the cap binds.")), ",\n",
  "  \"versions\": ", versions, ",\n",
  "  \"cases\": [\n    ", paste(cases, collapse = ",\n    "), "\n  ]\n",
  "}\n")

writeLines(json, out_path, sep = "", useBytes = TRUE)
cat("wrote", out_path, "\n")
