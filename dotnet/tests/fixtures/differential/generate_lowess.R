#!/usr/bin/env Rscript
#
# Generates lowess.json: R's own lowess() at its default interpolation distance, for the smoother
# behind the toolkit's LOWESS-based variance priors (peptide-count, intensity-trend,
# intensity-peptide-trend).
#
# Those priors fit LOWESS with span 0.5 and three robustness iterations. R's lowess() and
# statsmodels' are both Cleveland's algorithm; they differ in the default delta, the distance within
# which points are interpolated rather than fitted. R defaults to 1% of the range of x, statsmodels
# to 0, and Skyline-PRISM uses R's (VariancePriors.SmoothTrend). From 26.9.0 the toolkit passes R's
# default too (_prior_lowess), so this pins it to R's lowess() itself - on inputs where the
# interpolation actually happens, which is the normal case on real data: points crowd within 1% of
# the range of each other, and peptide counts tie.
#
# Nothing here imports Skyline-PRISM or the toolkit. Each case's x and y are written into the
# fixture; the tests hand them to the code under test, so it never depends on R's random numbers.
#
# The same generator lives in proteomics-toolkit (tests/fixtures/generate_lowess.R) and the two write
# byte-identical JSON, which pins PRISM's Lowess.Fit (VariancePriors.SmoothTrend) and the toolkit's
# _prior_lowess to the same reference. Change a case in both, or neither.
#
# Run from the repository root (on Windows, Rscript.exe is in the R installation's bin folder):
#   Rscript dotnet/tests/fixtures/differential/generate_lowess.R
#
# Generated with R 4.5.1; the version that ran is written into the fixture's `versions` field.

out_path <- file.path("dotnet", "tests", "fixtures", "differential", "lowess.json")
if (!dir.exists(dirname(out_path))) stop("run from the repository root: ", dirname(out_path), " not found")

# Every float is a JSON string of 17 significant digits, which round-trips a double exactly.
num <- function(x) {
  x <- as.numeric(x)
  out <- sprintf("%.17g", x)
  out[is.na(x)] <- "nan"
  out
}
js_str <- function(s) paste0("\"", gsub("\"", "\\\\\"", s), "\"")
js_vec <- function(x) paste0("[", paste(js_str(num(x)), collapse = ", "), "]")
js_obj <- function(fields, indent) {
  pad <- strrep(" ", indent + 2)
  body <- paste0(pad, js_str(names(fields)), ": ", unlist(fields), collapse = ",\n")
  paste0("{\n", body, "\n", strrep(" ", indent), "}")
}

encode_case <- function(name, note, x, y, indent = 4) {
  fit <- lowess(x, y, f = 0.5, iter = 3)  # delta left at its default, 0.01 * diff(range(x))
  delta <- 0.01 * diff(range(x))
  gaps <- diff(sort(x))
  cat(sprintf("  %-26s n %4d, delta %.4g, gaps within delta %d of %d\n",
              name, length(x), delta, sum(gaps <= delta), length(gaps)))
  if (!any(gaps <= delta)) stop(name, ": no two points within delta, so the case would not exercise it")
  js_obj(list(
    name = js_str(name),
    note = js_str(note),
    x = js_vec(x),
    y = js_vec(y),
    delta = js_str(num(delta)),
    fitted_x = js_vec(fit$x),
    fitted_y = js_vec(fit$y)
  ), indent)
}

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
cases <- character()

# Log variance against log mean intensity, the intensity trend's cloud: 600 points over four
# decades, so neighbors sit far closer together than 1% of the range.
set.seed(901)
x <- log(10^runif(600, 4, 8))
y <- log(0.04 + 2.5 * exp(-(x - log(1e4)) / 2)) + rnorm(600, sd = 0.7)
cases <- c(cases, encode_case(
  "crowded_intensity",
  "600 points of log variance on log mean intensity over four decades: neighbors are far closer than 1% of the range, so most points are interpolated",
  x, y))

# Log variance against log peptide count, the peptide-count prior's cloud: integer counts tie, and
# above about 25 peptides consecutive log counts are within 1% of the range.
set.seed(902)
counts <- pmax(1, round(exp(runif(400, 0, log(80)))))
y <- log(0.05 + 0.6 / counts) + rnorm(400, sd = 0.6)
cases <- c(cases, encode_case(
  "tied_peptide_counts",
  "400 proteins' log variance on log peptide count (1 to 80): tied counts, and consecutive log counts within 1% of the range above about 25",
  log(counts), y))

# A few hand-placed points with gaps on both sides of delta, and an outlier for the robustness
# iterations to downweight.
x <- c(0, 0.004, 0.009, 0.02, 0.35, 0.36, 0.5, 0.505, 0.75, 0.9, 0.995, 1)
y <- c(2.0, 2.1, 1.9, 2.2, 1.4, 1.5, 1.1, 4.0, 0.9, 0.7, 0.8, 0.6)
cases <- c(cases, encode_case(
  "gaps_either_side",
  "12 hand-placed points on [0, 1], some closer than delta (0.01) and some farther, with an outlier at x = 0.505 for the robustness weights",
  x, y))

json <- paste0(
  "{\n",
  "  \"reference\": \"R stats::lowess(x, y, f = 0.5, iter = 3), delta at its default 0.01 * diff(range(x))\",\n",
  "  \"note\": ", js_str(paste(
    "fitted_x is x sorted and fitted_y the smoothed value at each, as lowess returns them; ties keep",
    "their order. The toolkit's _prior_lowess must reproduce fitted_y.")), ",\n",
  "  \"versions\": ", js_obj(list(R = js_str(paste(R.version$major, R.version$minor, sep = "."))), 2), ",\n",
  "  \"cases\": [\n    ", paste(cases, collapse = ",\n    "), "\n  ]\n",
  "}\n")
writeLines(json, out_path, sep = "", useBytes = TRUE)
cat("wrote", out_path, "\n")
