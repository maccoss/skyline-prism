#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas==2.3.3",
#     "pyarrow==24.0.0",
# ]
# ///
"""Check PRISM's subject-blocked design against R's limma on a real PRISM output directory.

`blocked.json` pins the estimator on simulated cases. This repeats the comparison on a cohort, end
to end: the same matrix PRISM tests, handed to limma's own `duplicateCorrelation`, `lmFit(block=,
correlation=)` and `eBayes`, against the `differential.csv` PRISM wrote. It needs R with limma
(`BiocManager::install("limma")`); like the golden generators it imports nothing from PRISM.

First run PRISM with the global prior and no correction, so both sides compute the same thing:

    prism differential -d OUT --level peptide --group-by Sex -a F -b M \
        --design blocked --subject "Patient" --prior global --correction none -o prism.csv

then, from anywhere:

    uv run dotnet/tests/fixtures/differential/check_blocked_cohort.py -d OUT --level peptide \
        --group-by Sex -a F -b M --subject "Patient" --prism-csv prism.csv [--rscript PATH]

The group and subject columns must be in the output directory's `sample_metadata.csv`; a column
that only a `--clinical` CSV supplies is not read here. Samples are selected the way PRISM selects
them: in an arm, with a non-blank subject, compared trimmed. Features are those observed in every
selected sample (PRISM tests complete features only).

Reported: the two consensus correlations, and the largest disagreement in log2FC, t and p, each
measured against max(|limma value|, 1) as the golden tests measure a quantity that can sit at zero.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R_SCRIPT = r"""
suppressPackageStartupMessages(library(limma))
args <- commandArgs(trailingOnly = TRUE)
x <- as.matrix(read.csv(args[1], row.names = 1, check.names = FALSE))
m <- read.csv(args[2], colClasses = "character")
stopifnot(identical(colnames(x), m$sample))
design <- cbind(Intercept = 1, groupB = as.numeric(m$arm == "B"))
dc <- duplicateCorrelation(x, design, block = m$subject)
fit <- eBayes(lmFit(x, design, block = m$subject, correlation = dc$consensus.correlation))
cat(sprintf("%.17g\n", dc$consensus.correlation), file = args[4])
write.csv(data.frame(feature = rownames(x), logfc = sprintf("%.17g", fit$coefficients[, 2]),
  t = sprintf("%.17g", fit$t[, 2]), p = sprintf("%.17g", fit$p.value[, 2])), args[3], row.names = FALSE)
cat("limma", as.character(packageVersion("limma")), "\n")
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-d", "--output-dir", required=True, type=Path)
    ap.add_argument("--level", choices=["peptide", "protein"], default="protein")
    ap.add_argument("--group-by", required=True)
    ap.add_argument("-a", nargs="+", required=True, help="level(s) forming arm A")
    ap.add_argument("-b", nargs="+", required=True, help="level(s) forming arm B")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--prism-csv", required=True, type=Path)
    ap.add_argument("--rscript", default="Rscript")
    o = ap.parse_args()

    meta = pd.read_csv(o.output_dir / "sample_metadata.csv", dtype=str, keep_default_na=False)
    for col in (o.group_by, o.subject):
        if col not in meta.columns:
            sys.exit(f"'{col}' is not a column of sample_metadata.csv")
    arm = np.where(meta[o.group_by].isin(o.a), "A", np.where(meta[o.group_by].isin(o.b), "B", ""))
    subject = meta[o.subject].str.strip()
    keep = (arm != "") & (subject != "")
    sel = pd.DataFrame({"sample": meta["sample_id"][keep], "arm": arm[keep], "subject": subject[keep]})
    # [A..., B...] in metadata order, as PRISM orders the selected columns.
    sel = pd.concat([sel[sel.arm == "A"], sel[sel.arm == "B"]])

    matrix = pq.read_table(o.output_dir / f"corrected_{o.level}s.parquet").to_pandas()
    prism = pd.read_csv(o.prism_csv, comment="#")
    ids = set(prism["feature_id"].astype(str))
    # The id column is whichever text column holds PRISM's feature ids.
    id_col = max((c for c in matrix.columns if matrix[c].dtype == object),
                 key=lambda c: matrix[c].astype(str).isin(ids).sum())
    x = matrix.set_index(id_col)[sel["sample"].tolist()].astype(float)
    x = np.log2(x.where(x > 0))
    x = x[x.notna().all(axis=1)]

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        x.to_csv(t / "expr.csv")
        sel.to_csv(t / "meta.csv", index=False)
        (t / "check.R").write_text(R_SCRIPT, encoding="utf-8")
        run = subprocess.run([o.rscript, str(t / "check.R"), str(t / "expr.csv"), str(t / "meta.csv"),
                              str(t / "limma.csv"), str(t / "rho.txt")],
                             capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit(run.stderr)
        print(run.stdout.strip())
        limma = pd.read_csv(t / "limma.csv").set_index("feature")
        rho = float((t / "rho.txt").read_text())

    header = [ln for ln in o.prism_csv.read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    blocked = next((ln for ln in header if ln.startswith("# blocked by subject:")), "(no blocked line)")
    print(f"samples: {len(sel)} ({(sel.arm == 'A').sum()} A, {(sel.arm == 'B').sum()} B), "
          f"subjects: {sel.subject.nunique()}, features: {len(x)} (PRISM tested {len(prism)})")
    print(f"limma consensus correlation: {rho:.17g}")
    print(f"PRISM: {blocked}")

    joined = limma.join(prism.set_index("feature_id"), how="inner")
    print(f"features compared: {len(joined)}")
    for ref, got in (("logfc", "log2fc"), ("t", "statistic"), ("p", "p_value")):
        err = ((joined[got] - joined[ref]).abs() / np.maximum(joined[ref].abs(), 1.0)).max()
        print(f"  {ref:6s} max |PRISM - limma| / max(|limma|, 1) = {err:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
