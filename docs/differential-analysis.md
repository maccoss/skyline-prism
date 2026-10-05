# Differential analysis

The Skyline tool's **Differential** pane runs a limma empirical-Bayes moderated-t contrast on a
PRISM output directory: protein or peptide level, two groups from any metadata or clinical column,
optional covariate adjustment, and a choice of empirical-Bayes variance prior. The statistics live
in `SkylinePrism.Core.DifferentialAnalysis` and are reachable without the GUI.

The **default prior is the lab's own**, `VariancePriors.IntensityTrend` - the same estimator as
`proteomics-toolkit`'s `moderation="intensity_trend"`, and pinned to it by `intensity_trend.json`
(the trend's shape) and `toolkit_end_to_end.json` (the whole analysis). limma's `trend=TRUE` is still
available as **limma-trend**, and the two are not the same thing; the section below is about exactly
that.

## What each quantity is held to

Every component with a published reference implementation is pinned to it by committed goldens in
`dotnet/tests/fixtures/differential/`, regenerable with
`uv run dotnet/tests/fixtures/differential/generate.py`. See that directory's README for the
per-quantity reference table and for the tolerances, which are not uniform and are not arbitrary.

In summary: BH, Fisher exact, the polygamma functions and OLS agree with scipy/statsmodels/numpy to
around 1e-13 or better; `squeezeVar` and everything downstream of it agrees with inmoose to 1e-9,
which is the precision limma's own Newton solve for the prior degrees of freedom delivers.

## What the pane can run

Three orthogonal choices - a **design**, a **test**, and (for the moderated t alone) a **variance
prior** - which is how `proteomics-toolkit` models it too.

| Design | Test | Reference |
|---|---|---|
| unpaired | moderated t | lstsq + `squeezeVar`, composed as limma composes them |
| unpaired | Welch t / Student t | `scipy.stats.ttest_ind(equal_var=False/True)` |
| unpaired | Mann-Whitney U | `scipy.stats.mannwhitneyu(method="asymptotic")` |
| paired | moderated t | the same composition over `[1, group, subject dummies]` |
| paired | paired t | `scipy.stats.ttest_rel` |
| paired | Wilcoxon signed-rank | `scipy.stats.wilcoxon(method="asymptotic")` |
| trend, independent | moderated t | the same composition over `[1, x]` |
| trend, within subject | moderated t | the same composition over `[1, x, subject dummies]` |

Detection (the peptide on/off view) follows the design too: Fisher exact unpaired, McNemar's exact
test paired (`statsmodels.stats.contingency_tables.mcnemar(exact=True)`), and the Firth-penalized LRT
when covariates are set — the last being unpaired whatever the design, since a paired adjusted binary
outcome needs conditional logistic regression, which is not implemented.

Variance priors: **global** (Smyth 2004), **intensity trend** (the default - see below), **limma-trend**
(`trend=TRUE`), and **peptide count** (DEqMS, protein level only, pinned to the toolkit's
`_fit_count_dependent_prior`). The intensity trend and limma-trend both estimate the prior *degrees of
freedom* around their trend. The peptide-count prior keeps limma's global value, as DEqMS does, and
is used exactly as DEqMS defines it: limma with a prior scale that follows peptide count. It is not
calibrated to the residuals the way the intensity trend is (below), because that would be a different
method under DEqMS's name. The
intensity trend's shape can be fitted on the run's QC and reference replicates instead of on the
contrast groups.

Multiple-testing corrections: Benjamini-Hochberg, Benjamini-Yekutieli, Holm, Bonferroni, none - all
pinned to `statsmodels multipletests`, and all keeping PRISM's own NaN policy (a NaN passes through and
does not count toward *m*, where statsmodels returns all-NaN).

### Two things the rank tests do not do

**Neither implements an exact permutation distribution.** scipy's `method="auto"` uses one below n = 9
for Mann-Whitney and up to n = 50 for Wilcoxon; PRISM always uses the normal approximation with a tie
correction, so on a small group its p-value differs from a default-argument scipy run. The goldens pin
`method="asymptotic"` rather than encode a branch PRISM does not have.

**scipy's two rank tests disagree about the continuity correction** - `mannwhitneyu` applies one by
default, `wilcoxon` does not - and PRISM follows each one's own convention. Copying either to the other
shifts every p-value in that test by 10-20%, which is how this was found.

### A trend reports a change across a range, not a slope

The slope coefficient is in log2 **per unit of x**, so a threshold on it would mean something
different for a column measured in days than for the same column in hours. PRISM therefore reports
`slope x (x_max - x_min)` - the modeled change across the range actually measured - and the volcano's
x-axis is that quantity, labeled with the span. The raw slope is `log2fc / span`, and the span is in
`DifferentialResult.TrendRange` and in the CLI's CSV header.

This is a strict generalization, not a separate convention: code a two-level grouping as `x = 0` and
`x = 1` and the span is 1, so the reported change **is** the log2 fold change and the moderated t is
the same test. `TrendGeneralizesTheTwoArmContrastTests` pins both halves - the reduction to the
two-arm contrast, and the fact that rescaling the column (weeks to days) moves neither the reported
effect nor any p-value.

### Repeated measures need the subject block

**Trend, independent** treats every sample as its own observation. If the same subject contributes
several samples that is wrong: samples within a subject are correlated, the standard error is
understated, and the test reports more hits than the data support. Use **Trend, within subject** and
give it the subject column.

How much it matters is not a rounding difference. On the committed `trend.json` fixture - subjects
about two log2 units apart, a real slope of 0.09 per unit - the leading feature fits to
**t = 9.51, p = 1.8e-21** with the subject block and **t = 0.86, p = 0.39** without it. The naive fit
is stored in the fixture beside the correct one so the two designs cannot quietly become the same.

Like the paired design this is a **fixed** subject effect, not a random intercept. On a balanced
design the two give very similar slopes; they diverge on unbalanced designs, on missing timepoints,
and when a random *slope* per subject is wanted. A mixed model is a different estimator and is not
implemented.

A subject with fewer than two **distinct** trend values - one sample, or several all at the same
timepoint - carries no within-subject slope while still costing a parameter, so it is excluded and
counted in the status line. Covariates constant within every subject (sex, genotype, birth year) are
dropped for the same reason they are under a paired design: a within-subject slope cannot estimate a
between-subject effect.

### The intensity prior: shape from the controls, level and weight from the design

A moderated t replaces each protein's own variance estimate, which is noisy with few samples, by a
weighted average of that estimate and a prior:

```
posterior variance = (d0 * prior + residual df * residual variance) / (d0 + residual df)
```

The prior has a **scale**, the variance expected for a protein at that intensity, and a **weight**,
`d0`, which says how far to trust that scale over the protein's own estimate. The intensity prior
builds them in two steps:

1. **Shape.** A LOWESS of log(within-group variance) on log(within-group mean intensity) says how
   noise changes with intensity: low-abundance proteins are noisier. By default it is fitted on the
   run's **QC and reference injections** whenever there are two or more of a control type, because
   they trace that dependence without any biology in the way. Otherwise it is fitted on the
   contrast's two arms. **Fit prior on controls** in the pane, `--prior-from-controls` /
   `--prior-from-groups` on the command line. The controls need not be in the design: they are
   selected from the full matrix, so QC injections with no timepoint are dropped from a trend's
   design and still give its prior a shape.
2. **Level and weight.** The trend is multiplied by one factor, and `d0` is estimated at the same
   time, both from the design's own residual variances. This is limma's `fitFDist` with the trend as
   a known covariate offset: Smyth's method of moments on `residual variance / trend`. The CLI, the
   CSV header and the quant report print the factor, for example
   `prior: intensity trend from the controls, scaled x2.01 to these residuals; prior df 3.2`.

Step 2 exists because no source of groups measures the noise a design is tested against:

- **Control pools** carry instrument and processing noise only. A study residual also carries
  biology: person-to-person spread in an unpaired design, and week-to-week variation within a person
  in a paired one. Their trend sits too **low**. On the Verapamil serum cohort, the residuals ran
  about 2x above the controls' trend, even within subject.
- **Design groups under a paired or within-subject model** contain the between-subject spread that
  the model's subject block removes. Their trend sits too **high**: about 0.7x on the same cohort.
- **Small groups** bias any trend low. A LOWESS fits the mean of log variance, and the log of a
  chi-square draw averages below the log of its mean: a factor of 0.81 at six samples per group,
  0.28 at two.

**Up to dotnet-v26.27.0 there was no step 2,** and neither was there in `proteomics-toolkit` before
v26.8.0. The trend was used at its source's level, and `d0` was estimated separately from the
residuals around their own global mean - a weight measured against one level and applied to another.
The argument for the controls was that a prior fitted on design groups "describes measurement noise
plus that biology and shrinks genuine effects toward nothing". That argument had it backwards for
any comparison between people. The spread within a group is the error a between-group effect is
tested against, not the effect itself. A prior that leaves it out makes every test optimistic. When
`d0` came out large, the posterior variance was the controls' technical variance outright.

How far off it was, as the share of null p-values below 0.05 (a calibrated test gives 5%):

| Null test | Prior shape from | Before | After |
|---|---|---|---|
| **Simulated**, 2,000 features, 5 seeds | | | |
| unpaired 6 vs 6 | control pools | 40% | 4.9% |
| unpaired 6 vs 6 | design groups | 7.4% | 5.0% |
| paired, 6 subjects | control pools | 18.5% | 4.7% |
| paired, 6 subjects | design groups | 0.17% | 5.1% |
| **Verapamil serum cohort**, 3,595 proteins, 40 permutations | | | |
| unpaired 5 vs 5 (random split of week 0) | controls | 6.1% | 4.6% |
| unpaired 5 vs 5 | design groups | 4.5% | 4.5% |
| paired week 0 vs 12 (labels flipped within subject) | controls | 7.0% | 5.8% |
| paired week 0 vs 12 | design groups | 4.4% | 5.5% |
| trend over weeks 0-12 (shuffled within subject) | controls | 5.1% | 4.8% |
| trend over weeks 0-12 | design groups (one per week)\* | 4.5% | 4.8% |
| trend, random weeks on the week-0 samples (between-subject) | controls | 6.3% | 4.8% |
| trend, random weeks on the week-0 samples | design groups (one per week)\* | 6.2% | 4.8% |

\* Measured with the toolkit, which groups a trend's samples by timepoint. PRISM has no groups for
a trend design and uses the global prior there unless the run has controls.

Both paired rows stay slightly above 5% after the change (5.5-5.8%), under either source. That
residue is not explained by the prior, and it is not investigated here.

The real cohort's prior weight is small (`d0` about 2.5 against 9 to 39 residual df), which is why
its errors were modest. The simulations have a large `d0`, which is where the old estimator failed
badly. The simulations are in `proteomics-toolkit`'s `tests/test_statistical_analysis.py`
(`TestTrendCalibration`).

On the real result this was checked against, the one finding survives: CHGA's within-subject trend
over weeks 0-12 goes from p = 5.3e-7 (adjusted 0.0019) to p = 1.0e-6 (adjusted 0.0036), and it is
still the only protein below FDR 0.05.

**What changes for an existing result.** P-values move, in a direction set by the design and the
source. A control-sourced prior gets less optimistic. A paired or within-subject analysis on design
groups gets more powerful. An unpaired analysis on design groups barely moves. Two results are not
comparable unless they used the same prior source and the same version, which is why the source and
the factor are printed everywhere the prior is.

Controls are found by looking up `sample_type` against a fixed vocabulary - Skyline's **Standard**
and **Quality Control**, plus PRISM's own `reference` and `qc`. A run whose replicates are all
Unknown therefore has no control set: the pane **grays** the option rather than hiding it, and its
tooltip says to set those sample types in the Skyline document, so the analysis that would have been
available is discoverable rather than invisible. `proteomics-toolkit` can name any metadata column
here; PRISM deliberately does not, because the prior source changes every p-value and a free-text
picker invites pointing it at something that is not a set of technical replicates. See the note in
`CLAUDE.md`.

A trend design has no design groups at all, so forcing `--prior-from-groups` there falls back to the
global prior with a message. With controls present - the ordinary case - the intensity trend is
available on a trend design like any other.

### Paired is a fixed-effect subject block

`[1, group, subject dummies]`, matching `statistical_analysis.py:1321`. It is not a random intercept
and not a mixed model; those are a different estimator and are not implemented. What the block buys is
that each subject's overall level leaves the residual, so a within-subject shift is tested against
within-subject noise instead of against the spread between people.

## From the command line

The same analysis, against the same output directory, without Skyline or Windows:

```bash
prism differential -d output/ --group-by condition -a Control -b Disease
```

`prism differential` takes the whole menu above as flags - `--level`, `--design`, `--subject`,
`--trend-over`, `--restrict-to`, `--test`, `--prior`, `--prior-from-controls`, `--adjust-for`, `--correction`,
`--alpha`, `--raw-p`, `--min-log2fc` - and writes a results
CSV (`differential.csv` in the output directory unless `-o` says otherwise) whose header records the
contrast, its direction and the method that produced it. `prism differential --help` lists every
flag with its default.

Each arm takes several levels, and the arm is their **union**, so a three-level column can be
collapsed into a two-group contrast in one command:

```bash
prism differential -d output/ --group-by stage -a Control Mild -b Severe --adjust-for sex,age
```

A level named on both sides is refused rather than dropped from one, because which side it was
dropped from would change the answer and nothing in the output would record the choice.

A contrast on a clinical column - diagnosis, age, sex from a separate table - takes `--clinical`,
which joins the table exactly as the pane's Clinical CSV input does (the key column is detected by
value) before any column name is checked:

```bash
prism differential -d output/ --clinical clinical.csv --group-by Diagnosis -a Control -b AD \
    --adjust-for Age,Sex
```

A clinical CSV that matches fewer than half the samples is refused rather than silently ignored,
since the command almost certainly depends on one of its columns. `--clinical` can be given more
than once; the tables are joined in the order given.

A trend takes `--trend-over` in place of the arms, and the within-subject form also takes
`--subject`:

```bash
prism differential -d output/ --design trend --trend-over dose_mg
prism differential -d output/ --design trend-within-subject --trend-over week --subject patient_id
```

**A timepoint written as text.** Study metadata usually names a timepoint for people, not for an
axis - `V2_Week 8`, `Day 14`, `T0`. Such a column is still a trend axis: it is offered once for each
number its values hold, and each axis is named after the word just before that number. `V2_Week 8`
holds two numbers, and they are different quantities - the visit index and the elapsed weeks - so it
gives two axes, `Longitudinal Draw Description (V)` and `Longitudinal Draw Description (Week)`.
Nothing picks one for you. Naming the wrong one gives a plausible slope against the wrong quantity,
and nothing else in the output would show it. The pane shows what the chosen axis parsed on the line
below the **Trend over** picker in its settings panel (`Reading V0_Week 0 -> 0, V1_Week 2 -> 2, ...`)
and only selects one automatically when the column is already plain numbers. On the command line, a
bare column name (`--trend-over "Longitudinal Draw Description"`) is refused when the column offers
more than one reading, and the message lists them: name the one you mean. A value with no number in it (`na`, an empty cell) leaves that
sample off the axis, and the status line counts it. A hyphen is a minus sign only at the start of a value
or after a space or other punctuation (`Week -2` is -2). Straight after a letter or a digit it is a
separator: `Week-4` is 4, `Day-14` is 14, and `Week 2-4` reads 2 and 4. Read the other way, timepoints
written with hyphens would give a negative axis and flip the sign of every slope.

**Fitting on some of the samples.** A trend pools every sample that has a value on the axis. When a
cohort holds two studies on one timescale, that merges them into one slope - two arms both drawn
"at week 4" become a single point on the line. `--restrict-to COLUMN=VALUE[,VALUE...]` keeps only the
samples whose column holds one of the values. It can be given more than once, and the restrictions
narrow together (study *and* on-drug window). A column or value the run doesn't have is refused,
with the values it does have, rather than silently keeping nothing. The pane's **Restrict to** picker
does the same. Restricted samples are counted apart from samples with no value on the axis, so a
deliberate subset never reads as missing data. A report records its restriction wherever it records
the contrast - its title (`trend over ... (restricted to Study = Verapamil)`), `quant_parameters`
(`restrict_to:`) and the `# restricted to:` line in `differential.csv` - and its marker panels use
the same samples.

```bash
prism differential -d output/ --design trend-within-subject \
    --trend-over "Longitudinal Draw Description (Week)" --subject "BRI Subject ID" \
    --restrict-to "Study=Waking Beta Cells - Verapamil"
```

The span the results header reports - "log2fc is the modeled change across *x* to *y*" - comes
from the samples the fit used, never from the whole column. Otherwise an excluded arm's earlier
timepoint would describe every row by a range the model never saw.

The pane and the command resolve their arms through the same `ContrastArms` in Core, and run the
same `Differential.Run`, so a contrast set up by clicking and one typed out mean the same samples
and give the same numbers - checked on a 192-sample cohort, where the two agree bit for bit on
log2FC, p and adjusted p.

One caveat worth knowing before comparing two runs of your own: **which arm is A and which is B is
not a pure sign flip.** Swapping them reverses the sign of log2FC as you would expect, but it also
changes the order values are accumulated in, so the magnitudes move in the last digit or two
(around 1e-13 relative on that cohort). That is floating-point summation order, not a difference in
what was computed - but it means two results are only comparable digit-for-digit if the arms were
given the same way round.

## The quant report

A self-contained `quant_report.html`, written to a `quant/` folder in the output directory - the
quantification counterpart to `qc_report.html`. It runs one contrast across every view and bundles
them into one page that shares the QC report's Analysis Information header (read from the run's
`parameters.json`), so the report always names the version, date, host and inputs of the run that
produced the numbers.

Two front ends, one implementation (`QuantAnalysis` in Core): the Differential pane's **Quant
report...** button, which uses the pane's current contrast and the panels ticked in the Markers pane,
and `prism differential --report`, which uses the command's own flags:

```bash
prism differential -d output/ --group-by condition -a Control -b Disease \
    --report --markers "EV markers (core)"
```

`--markers` names panels from the same set the Markers pane offers (your saved lists and the shipped
ones), `--markers-group-by` picks their grouping column (default: `--group-by`; required under a
trend), and `--no-enrichment` skips g:Profiler on a machine with no internet access. An unknown panel
or column is refused before anything is written.

The report contains:

- the analysis parameters, as a table and as YAML (also written to `quant_parameters.yaml` /
  `.json`), recorded in `prism differential`'s own flag values - `design: paired`, `test: moderated`,
  `prior: intensity-trend`, `correction: bh`, and each arm as a list of its levels
  (`group_a: [qc, reference]`). The prior that actually ran is recorded beside the requested one as
  `prior_used`, since a requested prior can fall back;
- **the command that regenerates it** - the full `prism differential ... --report` line, every
  statistical choice spelled out, shown in the report and recorded as `command:` in the YAML. It is
  built from what the analysis actually ran: a covariate the pane still shows ticked under a test
  that cannot use one (anything but the moderated t) is dropped from the analysis, the detection test
  and the command alike, and the report says so. The test suite runs the recorded command through the
  CLI and requires every file it writes to be byte-identical to the pane's (the HTML apart from its
  timestamp) - adjusted, paired, stale-covariate and two-clinical-CSV reports included. The
  comparison runs with enrichment off, because g:Profiler answers from its current database and its
  output can change between two runs of anything. Rerun elsewhere, the command needs the clinical CSV
  at the path it names and any marker list you saved yourself on that machine (`PRISM_PROTEIN_LISTS`
  can point at a copy of your lists file). Where a value cannot be written on a command line at all -
  a level containing a comma, which `-a`/`-b` split on, or one with leading or trailing spaces - the
  report says so, and records `command_unavailable:` with the reason, in place of a command that
  would quietly select a different contrast;
- **differential abundance** - the volcano and a ranked hit table;
- **detection frequency** - the peptide on/off test, for a two-group contrast, chosen exactly as the
  Detection pane chooses it (`DetectionAnalysis` in Core serves both): the Firth-penalized GLM when
  covariates are set, McNemar's exact test over the matched subjects when the design is paired, and
  Fisher exact otherwise. The section names the test, and a paired design that had to run unpaired
  says why;
- **functional enrichment** - the g:Profiler bars, when the network is reachable and there are
  significant genes;
- **marker panels** - a heatmap and panel-score boxplot for each selected panel.

Beside the HTML it writes the result tables as CSVs (`differential.csv` - the same file, from the
same code, as `prism differential` writes - `detection.csv`,
`enrichment_terms.csv` with full gene lists, `markers_<panel>_zscores.csv`), and the **raw per-sample
abundances in LINEAR scale** (`differential_values.csv`, `markers_<panel>_values.csv`, values are
`2^log2`, matching `corrected_*.parquet`) so the export stands on its own for reanalysis. Long tables
are capped to a preview in the HTML; the full rows are in the companion CSV. A view with nothing to
show is omitted with a note rather than failing the report - no merged_data (detection), no network or
no significant genes (enrichment), no panels (markers), a trend (detection and the raw-value table) -
and each omission is printed on the console or the pane's status line with its reason.

## PRISM, `proteomics-toolkit` and limma

The lab's [`proteomics-toolkit`](https://github.com/uw-maccosslab/proteomics-toolkit) implements the
same moderated linear model in `run_moderated_linear_model`, independently of PRISM. How the three
compare depends on the prior:

| prior | agreement |
|---|---|
| global (toolkit `moderation="limma"`) | **~1e-13** on logFC, t and P.Value between PRISM, the toolkit and limma (inmoose) - three implementations of Smyth (2004) on the same numbers |
| intensity trend (both tools' **default**) | **PRISM and the toolkit are the same estimator**, pinned end to end by `toolkit_end_to_end.json` at 1e-9, and ~1e-11 on a 3,595-protein serum cohort once the two conventions below are aligned |
| intensity trend vs limma `trend=TRUE` | **median 0.9% on P.Value, at most 18%**, identical hit lists at p < 0.05 (the `moderated_t.json` trend case) |

**Two conventions separate a default toolkit run from PRISM**, and neither changes a conclusion:

- **The log pseudocount.** The toolkit's dispatcher adds the smallest value / 100 before log2,
  because a matrix it is handed may hold zeros. PRISM's matrix is log2 of a LINEAR parquet and adds
  nothing. On the serum cohort this moved low-abundance log ratios by up to 1.4e-3. Set
  `config.log_pseudocount = 0` to remove it.
- **The LOWESS interpolation distance.** PRISM's prior LOWESS interpolates between points closer
  than 1% of the x range, which a peptide-level contrast needs in order to return in under a second
  (`VariancePriors.SmoothTrend`). The toolkit fits every point. On the serum cohort this moved the
  prior by up to 1.4e-3 relative and p-values by up to 8e-4.

**The intensity trend and limma's `trend=TRUE` are still different estimators** that happen to share
a name. limma fits a natural cubic spline of `log(s^2)` on mean **log2** expression, one point per
feature, with spline df `1 + (n>=3) + (n>=6) + (n>=30)`. The intensity trend fits a LOWESS of
log(within-group variance) on log(within-group mean) on **raw** intensities, one point per (feature,
group), converted to log space by the delta method. What they now share is the second step: each
estimates the prior's level and `d0` from the residuals around its own trend. The intensity trend
gained that step in the release after dotnet-v26.27.0, and in toolkit v26.8.0. Before it, the two
disagreed by a median 11.7%
and up to 265% on the same case, and 3 of 7 hits differed. Against the **global** prior either trend
differs more (median 5-13%), which is the point of a trend prior.

So: quote which tool, which prior and which version produced a hit list. To compare across tools,
use the intensity trend in both (the same estimator) or the global prior in both.

The toolkit's own global prior differs from limma's `fitFDist` in two small ways that do not matter
on dense proteomics data and could on sparse: it drops zero and negative residual variances from
the prior fit where limma floors them at `1e-5 * median` and keeps them, and it has no
single-feature branch (limma returns `df_prior = 0` there).

## Cross-checking a PRISM result against the toolkit

```python
cfg.moderation = "intensity_trend"   # the default in both tools
cfg.log_pseudocount = 0              # PRISM adds none
# For a prior on the QC/reference injections, as PRISM's default does when the run has them:
cfg.variance_prior_group_column = "sample_type"
cfg.variance_prior_groups = ["reference", "qc"]
```

and read the PRISM side from the same matrix PRISM reported on
(`corrected_proteins.parquet` / `corrected_peptides.parquet`, which are **linear** - the toolkit's
`normalization_method = "prism"` log2s them; see the scale conventions in `CLAUDE.md`). PRISM's
trend log2fc is the change across the observed span; divide by the span to compare with the
toolkit's per-unit slope.

## Detection (peptide on/off)

The Detection view tests whether a peptide is observed at different rates between the groups. With
no covariates that is Fisher's exact test, pinned to `scipy.stats.fisher_exact`. With covariates it
is a **Firth-penalized** likelihood-ratio test - the full design against the same design with the
group column removed, referred to chi-square on 1 df.

Firth rather than an ordinary logistic regression because the interesting case is exactly the one
an unpenalized fit cannot express: a peptide detected in every sample of one group and none of the
other has no finite maximum-likelihood estimate. No standard library implements Firth, so the
golden reference maximizes the same penalized log-likelihood with a derivative-free optimizer
instead - a different algorithm reaching the same fixed point.
