#!/usr/bin/env Rscript
#
# Generates the cross-language ComBat golden fixtures: inputs, batch labels, and the output of
# the REFERENCE implementation, R's sva::ComBat. Both PRISM engines are held to these.
#
# Why this exists: PRISM's ComBat says it is "based on the original R sva package", but until
# 2026-08 nothing checked. It diverged in nine places, one of which (NaN propagation) turned a
# single missing value into an all-NaN output matrix. Python-vs-C# parity tests cannot catch that
# class of bug - they only prove the two engines agree, not that either is right.
#
# Prerequisites (one-off):
#   install.packages("BiocManager")
#   BiocManager::install("sva")
#
# Run from the repository root:
#   Rscript dotnet/tests/fixtures/sva/generate.R
#
# Windows, from the Claude Code prompt:
#   ! & "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" dotnet/tests/fixtures/sva/generate.R

suppressPackageStartupMessages(library(sva))

out_dir <- file.path("dotnet", "tests", "fixtures", "sva")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 17 significant digits round-trips a double exactly, and NA is written as NaN so the readers on
# the other side do not have to know R's spelling of missing.
write_matrix <- function(m, path) {
  con <- file(path, "w")
  on.exit(close(con))
  writeLines(paste(c("feature", colnames(m)), collapse = ","), con)
  for (i in seq_len(nrow(m))) {
    vals <- sprintf("%.17g", m[i, ])
    vals[is.na(m[i, ])] <- "NaN"
    writeLines(paste(c(rownames(m)[i], vals), collapse = ","), con)
  }
}

write_batches <- function(samples, batch, path) {
  con <- file(path, "w")
  on.exit(close(con))
  writeLines("sample,batch", con)
  writeLines(paste(samples, batch, sep = ","), con)
}

# A cohort with a real batch effect: per-batch offsets and per-batch scales on top of a
# per-feature baseline, which is what ComBat is meant to remove.
make_case <- function(n_features, sizes, missing_fraction = 0, degenerate = "none") {
  batch <- rep(paste0("batch", seq_along(sizes)), times = sizes)
  n_samples <- length(batch)
  samples <- sprintf("s%02d", seq_len(n_samples))

  m <- matrix(0, nrow = n_features, ncol = n_samples,
              dimnames = list(sprintf("F%04d", seq_len(n_features)), samples))
  offsets <- c(0, 0.8, -0.5)[seq_along(sizes)]
  scales  <- c(1, 1.6, 0.7)[seq_along(sizes)]
  for (f in seq_len(n_features)) {
    baseline <- 15 + 6 * runif(1)
    for (b in seq_along(sizes)) {
      cols <- which(batch == paste0("batch", b))
      m[f, cols] <- baseline + offsets[b] + scales[b] * rnorm(length(cols), sd = 0.4)
    }
  }

  if (degenerate == "constant_in_batch") {
    # Constant within one batch but not overall. sva drops the whole feature; PRISM keeps the
    # estimable location correction and skips only the scale. Deliberate divergence.
    cols <- which(batch == "batch2")
    m[1, cols] <- 17.25
    m[2, cols] <- 19.5
  } else if (degenerate == "single_obs") {
    # A single observation in one batch: its location is estimable, its scale is not. sva's
    # delta.hat becomes NA (var of one value) and the it.sol loop never terminates cleanly.
    cols <- which(batch == "batch1")
    m[4, cols[-1]] <- NA
  } else if (degenerate == "absent_from_batch") {
    # Never observed in one batch, so that batch's effect on it is undefined. sva cannot express
    # this: Beta.NA drops the unobserved rows, the batch's design column becomes all-zero, and
    # solve() dies with "system is exactly singular". PRISM holds the feature out instead.
    m[3, which(batch == "batch3")] <- NA
  }

  if (missing_fraction > 0) {
    # Missing at random, but never a whole feature or a whole (feature, batch) cell beyond the
    # cases constructed above - those are what the degenerate fixture is for.
    for (f in seq_len(n_features)) {
      for (b in seq_along(sizes)) {
        cols <- which(batch == paste0("batch", b))
        drop <- cols[runif(length(cols)) < missing_fraction]
        if (length(drop) < length(cols) - 1) m[f, drop] <- NA
      }
    }
  }

  list(matrix = m, batch = batch, samples = samples)
}

run_case <- function(name, case, expect_failure = FALSE) {
  cat("=== ", name, " ===\n", sep = "")
  write_matrix(case$matrix, file.path(out_dir, paste0(name, "_input.csv")))
  write_batches(case$samples, case$batch, file.path(out_dir, paste0(name, "_batches.csv")))

  corrected <- tryCatch(
    ComBat(dat = case$matrix, batch = case$batch, mod = NULL, par.prior = TRUE),
    error = function(e) {
      cat("  sva::ComBat FAILED: ", conditionMessage(e), "\n", sep = "")
      writeLines(conditionMessage(e), file.path(out_dir, paste0(name, "_sva_error.txt")))
      NULL
    }
  )

  if (is.null(corrected)) {
    if (!expect_failure)
      stop("sva::ComBat unexpectedly FAILED for case '", name, "'")
    return(invisible(NULL))
  }
  if (expect_failure)
    stop("sva::ComBat unexpectedly SUCCEEDED for case '", name,
         "' - the fixture no longer demonstrates what it claims")

  write_matrix(corrected, file.path(out_dir, paste0(name, "_sva.csv")))
  cat("  wrote ", nrow(corrected), " x ", ncol(corrected),
      "; NaN in output: ", sum(is.na(corrected)), "\n", sep = "")
}

set.seed(20260806)

# Every engine must reproduce this one exactly (to floating-point tolerance).
run_case("dense", make_case(n_features = 40, sizes = c(5, 4, 5)))

# The case the bug was about: PRISM used to turn ANY missing value into an all-NaN matrix.
run_case("sparse", make_case(n_features = 40, sizes = c(6, 5, 6), missing_fraction = 0.12))

# Where PRISM deliberately differs from sva - see PRISM-BUG-combat-nan-propagation.md rows 2-4.
# sva drops a feature that is constant within any batch; PRISM keeps its (estimable) location
# correction and skips only the scale. The golden records sva's answer so the divergence stays
# documented and deliberate rather than accidental.
run_case("constant_in_batch",
         make_case(n_features = 40, sizes = c(6, 5, 6), degenerate = "constant_in_batch"))

# The last two are cases sva cannot express at all: it errors out rather than producing numbers.
# There is no golden matrix to match, only the error - and PRISM's behaviour (hold the feature out,
# or correct its location only) is an extension of sva rather than a divergence from it.
run_case("single_obs",
         make_case(n_features = 40, sizes = c(6, 5, 6), degenerate = "single_obs"),
         expect_failure = TRUE)

run_case("absent",
         make_case(n_features = 40, sizes = c(6, 5, 6), degenerate = "absent_from_batch"),
         expect_failure = TRUE)

cat("\nDone. Wrote fixtures to ", out_dir, "\n", sep = "")
cat("sva version: ", as.character(packageVersion("sva")), "\n", sep = "")
