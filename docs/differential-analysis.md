﻿# Differential analysis

The Skyline tool's **Differential** pane runs a limma empirical-Bayes moderated-t contrast on a
PRISM output directory: protein or peptide level, two groups from any metadata or clinical column,
optional covariate adjustment, and a choice of empirical-Bayes variance prior. The statistics live
in `SkylinePrism.Core.DifferentialAnalysis` and are reachable without the GUI.

The **default prior is chosen by level.** On the protein-level matrix, which carries each protein's
peptide count, it is **limma-trend-intensity-n-peptides**: limma's `trend=TRUE` with log peptide
count as a second covariate. On the peptide-level matrix, which has no peptide counts, it is
**limma-trend**: limma's `trend=TRUE` alone, an intensity-dependent prior. Both curves are fitted to
the contrast's own residuals. The run names the prior it used. `proteomics-toolkit` uses the same
names and chooses its default the same way, and both tools are pinned to one R limma golden. A
default run gives the same p-values in either tool (see "PRISM, `proteomics-toolkit` and limma").

Up to dotnet-v26.28.0 the default was the lab's intensity trend, `VariancePriors.IntensityTrend`,
the same estimator as `proteomics-toolkit`'s `moderation="intensity-trend"` and pinned to it by
`intensity_trend.json` and `toolkit_end_to_end.json`. dotnet-v26.29.0 replaced it with limma-trend
at both levels, and the release after that gave protein level the peptide count too. The intensity
trend is deprecated and kept to reproduce earlier results, for the reason in "Why limma-trend
replaced the intensity trend" below. It and limma-trend are not the same estimator despite the
names.

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
| blocked by subject | moderated t | R `limma::duplicateCorrelation` + `lmFit(block=, correlation=)` + `eBayes` |
| trend, independent | moderated t | the same composition over `[1, x]` |
| trend, within subject | moderated t | the same composition over `[1, x, subject dummies]` |

Detection (the peptide on/off view) follows the design too: Fisher exact unpaired, McNemar's exact
test paired (`statsmodels.stats.contingency_tables.mcnemar(exact=True)`), and the Firth-penalized LRT
when covariates are set — the last being unpaired whatever the design, since a paired adjusted binary
outcome needs conditional logistic regression, which is not implemented. Under the blocked design
detection has no counterpart: it runs Fisher (or Firth) on samples and says that its p-values do not
account for subjects sampled more than once.

Variance priors: **global** (Smyth 2004), **limma-trend** (`trend=TRUE`, the default at peptide
level), **limma-trend-intensity-n-peptides** (limma-trend plus peptide count, the default at protein
level, protein level only - see "At protein level: intensity and peptide count" below), **peptide
count** (DEqMS, protein level only, pinned to the toolkit's `_fit_count_dependent_prior`), and
**intensity trend** (deprecated - see below). The intensity trend and both limma-trend priors
estimate the prior *degrees of freedom* around their trend. The peptide-count prior keeps limma's global value, as DEqMS does, and
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

### Why limma-trend replaced the intensity trend

Up to dotnet-v26.28.0 the default was the intensity trend with its shape taken from
the QC and reference pools (next section). Its overall false-positive rate was right - about 5% of null
p-values below 0.05 - but split by intensity it was not: optimistic for bright features and
conservative for dim ones, the two cancelling in the average. The calibration work in the next section
was all measured overall, which is how it went unseen.

The cause is the curve's shape. On the log2 scale the moderated t works on, a feature's variance is
roughly (CV / ln 2)^2. The pools carry technical noise only: its absolute SD rises with intensity
(shot noise goes as about the square root of the signal), so its CV - the log-scale noise - falls,
towards a floor set by multiplicative sources such as injection and integration. Study samples add
biological variation, which is multiplicative - a roughly constant CV - and so close to flat on the
log scale. One scale factor cannot fit a falling curve to a flatter one: scaled to fit on average, the
pool curve sits too low at high intensity and too high at low. The mismatch also roughly halves the
fitted prior df, so the prior moderates less and the test loses power. limma-trend has no source to
get wrong: it fits its curve to the contrast's own residual variances, so it follows whatever mix of
technical and biological variance the design leaves.

The share of null p-values below 0.05 in the dimmest and brightest fifth of features (a calibrated
test gives 5% in both), from random-label nulls - splits, label swaps within subject, or weeks
shuffled within subject:

| Data, design | intensity trend from pools | from design groups | limma-trend |
|---|---|---|---|
| `cohort` fixture (327 peptides), unpaired 3 vs 3 | 2.4% -> 8.9% | 6.2% -> 4.7% | 5.0% -> 5.3% |
| `cohort` fixture, between-subject trend, 10 samples | 3.5% -> 8.0% | n/a | 5.2% -> 6.0% |
| Verapamil serum (3,595 proteins), paired week 0 vs 12 | 4.6% -> 7.0% | 6.1% -> 3.7% | 5.2% -> 5.6% |
| Verapamil, within-subject trend over weeks 0-12 | 5.0% -> 6.2% | n/a | 5.2% -> 5.7% |
| Verapamil, unpaired 5 vs 5 (week-0 samples) | 3.8% -> 6.9% | 5.0% -> 4.5% | 4.7% -> 5.1% |
| simulated, paired, 6 subjects | 2.1% -> 9.4% | 9.2% -> 2.7% | 5.1% -> 5.3% |

The design-group shape is no fix: in a paired design the groups carry the between-subject spread that
the subject block removes, so it tilts the other way, and a trend design has no groups at all. Where a
paired or within-subject design's residual really is technical only - repeat draws from a person that
differ only by measurement - the pool shape is right, and simulation shows it flat at 5% there; real
people vary from visit to visit, and the Verapamil rows show it.

Power, simulated at BH 5% with a two-fold effect in 10% of features: the pool-shaped prior found 52%
of true effects against limma-trend's 71% (unpaired 5 vs 5; 30% against 58% of the dim ones), and 89%
against 95% (paired, 6 subjects), with its false hits piled on bright features. limma-trend matched
the best of everything tested in every design and size (100 to 3,000 features), including a prototype
that kept the pool curve and added a fitted flat floor.

The intensity trend is still available - `--prior intensity-trend`, or **Intensity trend, toolkit
(deprecated)** in the pane - so an earlier result can be reproduced. A recorded command always names
its prior, so one recorded before the change reproduces as it ran. Every result that uses it says it
is deprecated. `proteomics-toolkit` deprecated it too, in its 26.9.0, with the same measurements.

### At protein level: intensity and peptide count

limma-trend cannot see peptide counts, and at protein level variance depends on them: a protein's
measurement noise averages down over its peptides, while its biology does not. At the same
intensity, a one-peptide protein's residual variance on the Verapamil serum cohort was 2.3-2.9 times
a four-peptide protein's, levelling off at about 0.6-0.7 times from ten peptides on, where biology
dominates. So limma-trend tilts by peptide count: optimistic for proteins with few peptides,
conservative for proteins with many. DEqMS (`peptide-count`) is the mirror image: it uses the count
and not the intensity, and tilts by intensity.

**limma-trend-intensity-n-peptides** uses both. Each protein's bias-corrected log residual variance
is regressed on limma-trend's natural spline of mean log2 expression plus a natural spline of log
peptide count. The count spline has one interior knot, at the median. It is a straight line in log
count when the counts take only two values, or when more than half the proteins share the smallest
or largest count, which puts the knot on the boundary. The prior df is then estimated around the
combined curve by limma's trigamma inversion. Everything is fitted to the contrast's own residuals,
as for limma-trend, so it needs no QC or reference samples and assumes nothing about how fast
variance falls with peptide count.

The share of null p-values below 0.05 (a calibrated test gives 5% everywhere), in the dimmest and
brightest fifth of proteins and from the fewest to the most peptides. These were measured with
`proteomics-toolkit`, whose numbers PRISM reproduces (below). The simulation is protein-level, with
peptide counts log-uniform from 1 to 64 and technical variance divided by the count. Six draws of
2,000 proteins, so a fifth's 5% has a binomial sd of 0.44 points. The serum rows are random-label
nulls on the Verapamil cohort's 3,595 protein groups (median 4 peptides; 23% have one), 40
permutations each, with peptide count in bins (1 to 10 or more) because so many proteins tie at one.

| Data, design, prior | intensity: dimmest -> brightest | peptides: fewest -> most | prior df |
|---|---|---|---|
| simulated, unpaired 5 vs 5, limma-trend | 4.8% -> 4.7% | 7.8% -> 2.6% | 6.3 |
| same, peptide-count | 6.8% -> 3.1% | 4.6% -> 4.1% | 4.7 |
| same, limma-trend-intensity-n-peptides | 4.4% -> 4.8% | 4.3% -> 4.9% | 10.0 |
| simulated, paired, 6 subjects, limma-trend | 5.7% -> 4.4% | 9.0% -> 2.3% | 5.1 |
| same, peptide-count | 8.8% -> 1.5% | 4.8% -> 4.3% | 3.5 |
| same, limma-trend-intensity-n-peptides | 5.7% -> 4.1% | 4.9% -> 5.0% | 10.3 |
| Verapamil, unpaired 5 vs 5, limma-trend | 5.0% -> 4.7% | 6.9% -> 3.0% | 3.4 |
| same, peptide-count | 5.5% -> 3.8% | 3.9% -> 3.9% | 2.6 |
| same, limma-trend-intensity-n-peptides | 4.9% -> 4.9% | 4.4% -> 4.2% | 5.2 |
| Verapamil, paired week 0 vs 12, limma-trend | 5.0% -> 4.6% | 6.8% -> 3.1% | 4.2 |
| same, peptide-count | 5.5% -> 3.0% | 3.8% -> 4.0% | 2.5 |
| same, limma-trend-intensity-n-peptides | 4.9% -> 4.6% | 4.3% -> 4.2% | 6.7 |

The serum rows' uncertainty is the spread across permutations, which share the cohort's proteins:
0.1-0.3 points per bin unpaired and up to 0.7 paired. On a second cohort (4,657 protein groups, 35
subjects sampled twice, two PRISM outputs of the same data), designs of 5 vs 5 and of 6 pairs
showed the same: limma-trend's one-peptide rate exceeded its ten-or-more rate by 4.0-5.0 points,
limma-trend-intensity-n-peptides's by -2.6 to +1.1 (standard errors 0.3-1.6). With all 35
subjects, neither tilted: with 33-34 residual df the prior barely matters.

It matters most in small designs, where the prior carries more of each protein's variance. In
spike-ins (one protein in ten shifted, Benjamini-Hochberg at 5%) it found 28.0% of two-fold
effects against limma-trend's 23.0% on the serum cohort, unpaired 5 vs 5, and 16.5% against
12.9% on the second cohort. The gain is in well-covered proteins (ten or more peptides: 55.0%
against 40.5%), and one-peptide proteins give some up (2.9% against 7.9%). Those are the proteins
where limma-trend's null rate was 6.9%, so part of what it found there was false. Expect fewer hits
among single-peptide proteins and more among well-covered ones.

**Protein level only, and enforced.** A peptide has no peptide count, so this prior - and
`peptide-count`, the other one that uses the count - refuses, with a message saying why, unless every
row is one protein with the number of peptides it was rolled up from. It stops when there are no
peptide counts (only `corrected_proteins.parquet` carries `n_peptides`), when any count is missing,
not a whole number or below 1, when every protein has the same count (what a peptide-level table
looks like), and when a feature id repeats. Named, it also refuses when too few proteins have a
usable residual variance to fit both covariates. As the default it then uses limma-trend and says
so in the run's messages; it never falls back without saying. Up to dotnet-v26.29.0 `peptide-count`
without usable counts ran the global prior instead, with a message; it now refuses, as the toolkit
does, because a substituted prior gives different p-values under the requested prior's name.

`ProteinLevelCounts` makes the checks and `EmpiricalBayes.SqueezeVarTrend` fits the prior.
`limma_trend_intensity_n_peptides.json` pins it to R limma 3.66.0, and
`LimmaTrendIntensityNPeptidesTests` holds it within 3.5-6.5% in every intensity and every
peptide-count fifth of a simulated protein-level null, where limma-trend tilts by count (7.7% of null
p-values below 0.05 in the fewest-peptide fifth, 3.0% in the most).

### The intensity prior: shape from the controls, level and weight from the design

This section describes the deprecated intensity trend - the default up to dotnet-v26.28.0.

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
tooltip says the shape then comes from the contrast groups. Up to dotnet-v26.28.0 it also suggested
setting those sample types in the Skyline document; it no longer does, because the pool shape is what
the deprecation is about. `proteomics-toolkit` can name any metadata column
here; PRISM deliberately does not, because the prior source changes every p-value and a free-text
picker invites pointing it at something that is not a set of technical replicates. See the note in
`CLAUDE.md`.

A trend design has no design groups at all, so forcing `--prior-from-groups` there falls back to the
global prior with a message. With controls present - the ordinary case - the intensity trend is
available on a trend design like any other.

### Covariate types: a column of numbers is not always a number

A covariate enters the design one of two ways: **numeric**, as one mean-centered column (a
straight-line effect of its value), or **categorical**, as one indicator per level after the first
(a separate shift per level). Up to dotnet-v26.28.0 every column whose values all parsed as numbers
was numeric. That made an integer-coded subject ID a straight-line effect of an arbitrary ID number,
and an integer-coded batch with three or more levels a linear trend across batches - both fitted
with no word to the user. On a 63-sample longitudinal cohort, "adjusting" a sex contrast for patient
IDs 1-15 this way changed the hit list from 86 peptides to 78, only 32 of them shared.

The default is now (`CovariateTyping.Infer`):

- text values: categorical;
- whole numbers that repeat across samples, where the column name contains one of the whole words
  patient, subject, participant, individual, donor, animal, mouse, id, batch, plate, site, center,
  cohort, group, cycle, set or run (split on spaces, punctuation, case changes and letter-digit
  boundaries, so `File set` and `PatientID` match and `Onset` and `Dataset` do not): categorical.
  The values have to look like labels as well: `Patient age` in decimals and `Run order` (1, 2, 3,
  ... never repeating) stay numeric. A whole-number measurement that does repeat under such a name,
  such as `Patient age` in whole years recorded once per patient, still comes out categorical, and
  its row in the pane says so and switches it;
- any other numbers: numeric (age, a clinical score, a dose, a visit number, years from diagnosis).

There is deliberately no rule on the number of distinct values. With one - at most ten whole numbers
made a column categorical - an integer age, a run order or a dose became a factor in a small cohort:
eight subjects' ages were eight levels, dropped as nested in the groups, or partly nested, leaving
the contrast estimated from the few subjects who shared an age. Whether such a column was fitted at
all then depended on how many subjects the study happened to have.

The type is shown and can be overridden: in the pane, as a Numeric | Categorical toggle on each
ticked Adjust-for column, which also shows before the run what the design will do with that column
(`Differential.CovariateNotes`, the same builder the run uses); on the command line, `--covariate-type COLUMN=numeric|categorical`. Results name each covariate with the type it was
fitted as (`adjusted for Patient (categorical)`), `quant_parameters` records `covariate_types`, and
a recorded command carries `--covariate-type` for every covariate, so a later change to this rule
cannot make it fit a different model.

A blank value, and one written exactly as `#N/A`, `N/A`, `n/a`, `NA`, `#NA`, `NULL`, `null`, `NaN` or
`Infinity`, is missing (case matters: `Na`, sodium, is a value), for text columns as well as numbers (so a numeric column whose QC pools hold `#N/A` stays numeric); a covariate missing in a selected sample is skipped, and the result says so. (A blank in a
text column used to be a level of its own.)

When a categorical covariate's levels use up the residual degrees of freedom, the refusal names it
and how many design columns it takes.

Three consequences are reported on the result, and noted in the pane before the run, rather than
left to a rank-deficiency error or to nothing:

- **A categorical covariate nested in the groups** - every value in one arm only, as a patient ID is
  under a sex contrast - would absorb the contrast as a fixed effect, so it is dropped and named. A
  fixed covariate cannot account for a subject's repeated samples in a between-subject contrast.
- **A categorical covariate partly confounded with the groups** - some of its values in one arm only -
  can be fitted, and is. But each such value's indicator absorbs its samples' mean, so those samples
  do not inform the contrast, which is estimated only from the samples whose value occurs in both
  arms. The note names the values and counts the samples.
- **A column forced numeric although its inferred type is categorical** - its name and its repeating
  whole numbers say it labels groups - is fitted as asked, with a warning. A column inferred numeric
  gets no warning, whatever its name.

### Between subjects, with repeated samples: the blocked design

**Blocked by subject** (`--design blocked --subject COL`) is for a two-arm contrast whose groups are
constant within a subject - sex, onset site, a decline category - when subjects were sampled more
than once. Neither of the other two-arm designs is right there:

- **Unpaired** counts each of a subject's samples as independent evidence. Samples from one person
  are correlated, so the standard error is understated and the test reports more hits than the data
  support.
- **Paired** (a fixed subject block) cannot be fitted at all: the subject is nested in the group,
  so the block absorbs the contrast.

The blocked design models a subject's samples as correlated. It is limma's approach (Smyth, Michaud
& Scott 2005, *Bioinformatics* 21(9):2067-2075): one intra-subject correlation, shared by every
feature, estimated as the 15%-trimmed mean on the atanh scale of each feature's REML estimate
(`duplicateCorrelation`). Each feature is then fitted by generalized least squares at that
correlation (`lmFit(block=, correlation=)`) and moderated as usual, under any of the priors. The
result names the correlation and counts subjects as well as samples per arm, because the subjects are
what a between-subject contrast has to work with.

It is pinned to R's limma by `blocked.json`, generated by `generate_blocked.R` (inmoose has neither
function). PRISM replays statmod's REML iterations step for step - including the 20-iteration cap
and 1e-6 stopping rule limma passes it, which leave limma short of the REML optimum by up to 2.4e-3
on the atanh scale - so the per-feature correlations agree with limma's to about 4e-14 and
everything downstream to about 1e-12. On a 7,602-peptide, 63-sample, 15-patient ALS CSF cohort
(sex, F vs M, global prior) the whole result agreed with limma to 2e-14 on log2FC and 5e-12 on p,
at a correlation of 0.441. That cohort is not committed; the comparison is, as
`dotnet/tests/fixtures/differential/check_blocked_cohort.py`, and repeats on any PRISM output
directory and subject column (see the fixture README).

On simulated null data with a correlation of 0.5 and 3-6 samples per subject, the blocked design's
false-positive rate at 0.05 was 4.3% where the unpaired one's was 22.8%
(`BlockedDesignTests.UnderTheNull_BlockedHoldsItsFalsePositiveRate_AndUnpairedDoesNot`).

What it is not:

- **Not a per-feature mixed model.** One correlation serves every feature, and that has a cost.
  Hoffman & Roussos 2021 (dream, *Bioinformatics* 37(2):192-201,
  [doi:10.1093/bioinformatics/btaa687](https://doi.org/10.1093/bioinformatics/btaa687)) report a
  slight increase in `duplicateCorrelation`'s type I error at larger sample sizes in their
  simulations, and point out that a feature whose own intra-subject correlation is above the shared
  one is under-corrected, which raises its false-positive rate. They also show that methods which
  ignore the correlation do not control the false-positive rate at all. `proteomics-toolkit`'s mixed
  model (a per-feature `statsmodels` `mixedlm`) is a per-feature estimator of that other kind, and
  the two are not expected to agree.
- **The intensity-trend prior on a blocked fit has no external golden.** limma has no intensity
  trend and the toolkit has no blocked fit. The prior's shape comes from the controls (or design
  groups) as always, and its level and `d0` are calibrated to the GLS residual variances by the same
  step `toolkit_end_to_end.json` pins; the GLS itself is pinned by `blocked.json`.
- **Two-arm only.** A trend, or time by group with a subject block, is not offered yet; the GLS fit
  itself takes any design, and `blocked.json`'s `time_by_group` case already pins it on one.
- Samples with no subject are left out and counted. A run where no subject has two samples is the
  unpaired fit, at a correlation of 0, and says so.
- Each arm needs at least two subjects (`--min-per-group`), however many samples it holds. Between
  subjects the subjects are the replicates, and with one subject per arm the subject indicators are
  the group column itself: `duplicateCorrelation` then answers 0, and the "blocked" fit would be the
  unpaired one under another name. It is refused instead, with each arm's subject count.

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
`--trend-over`, `--restrict-to`, `--test`, `--prior`, `--prior-from-controls`, `--adjust-for`,
`--covariate-type`, `--correction`,
`--alpha`, `--raw-p`, `--min-log2fc` - and writes a results
CSV (`differential.csv` in the output directory unless `-o` says otherwise) whose header records the
contrast, its direction and the method that produced it. `prism differential --help` lists every
flag with its default.

Each arm takes several levels, and the arm is their **union**, so a three-level column can be
collapsed into a two-group contrast in one command:

```bash
prism differential -d output/ --group-by stage -a Control Mild -b Severe --adjust-for sex,age
```

Each `--adjust-for` column is fitted as the type the output names; `--covariate-type plate=numeric`
overrides one (see "Covariate types" above).

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

A between-subject contrast where subjects were sampled more than once takes the blocked design:

```bash
prism differential -d output/ --group-by Sex -a F -b M --design blocked --subject patient_id
```

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
  `prior: limma-trend`, `correction: bh`, and each arm as a list of its levels
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

The two tools use the same prior names - the toolkit's `config.moderation` takes PRISM's `--prior`
tokens, with underscores accepted for the hyphens - and, from toolkit 26.9.0, the same default,
chosen by level.

| prior | agreement |
|---|---|
| limma-trend-intensity-n-peptides (the default at protein level in both) | **one golden for both tools**: `limma_trend_intensity_n_peptides.json` is written byte for byte the same by twin R scripts in the two repositories, from limma 3.66.0's own pieces, and pins each tool to it (PRISM at 1e-9 on the prior df and p, 1e-10 on logFC and t). On the Verapamil serum cohort, paired week 0 vs 12, a default `prism differential --level protein` and the toolkit's default `run_moderated_linear_model` on the same matrix and `n_peptides` both chose this prior and agreed on all 3,595 proteins to **1.1e-12** relative on P.Value and 1e-14 on logFC |
| limma-trend (the default at peptide level in both) | each tool pinned to limma: PRISM by `moderated_t.json` (inmoose `squeezeVar`) and `ebayes.json` (R `eBayes(trend = TRUE)`), the toolkit by its `limma_ebayes.json` (R); the golden above also requires its own transcription of limma to reproduce limma-trend to 1e-12 before it writes a case |
| global (toolkit `moderation="global"`, formerly `"limma"`) | **~1e-13** on logFC, t and P.Value between PRISM, the toolkit and limma (inmoose) - three implementations of Smyth (2004) on the same numbers - except P.Value in toolkit 26.8.0 and earlier where limma's df cap binds (an infinite prior df, or a small feature set); see below |
| intensity trend (**deprecated** in both, `--prior intensity-trend`) | **PRISM and the toolkit are the same estimator**, pinned end to end by `toolkit_end_to_end.json` at 1e-9 on t and the prior, and ~1e-11 on a 3,595-protein serum cohort once the two conventions below are aligned. P.Value differs wherever the df cap binds, as for the global prior |
| intensity trend vs limma `trend=TRUE` | **median 0.9% on P.Value, at most 18%**, identical hit lists at p < 0.05 (the `moderated_t.json` trend case) |

**Two conventions can separate a toolkit run from PRISM**, and neither changes a conclusion. The
first applies to every prior when the toolkit is called through its top-level dispatcher; the
second to the intensity trend only:

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
leave the prior at its default in both (from toolkit 26.9.0), or name the same one in both.

The toolkit's own global prior differs from limma's `fitFDist` in two small ways that do not matter
on dense proteomics data and could on sparse: it drops zero and negative residual variances from
the prior fit where limma floors them at `1e-5 * median` and keeps them, and it has no
single-feature branch (limma returns `df_prior = 0` there).

**The moderated t's degrees of freedom are capped, as limma caps them.** limma's `eBayes` - and
PRISM from dotnet-v26.29.0, and the toolkit from 26.9.0 - never gives the t more degrees of freedom
than the residuals pooled over every feature tested:
`df.total = pmin(df.residual + df.prior, sum(df.residual))`. Toolkit 26.8.0 and earlier, and the
PRISM Differential Explorer, use `df.residual + df.prior` uncapped. The two agree whenever `df.prior` is finite and below
`(features tested - 1) x df.residual`, which on a real cohort of hundreds or thousands of features it
is. With an infinite prior df they always part, because the uncapped t is read off a normal: for
3,000 features at 8 residual df the capped p is higher by 0.1% at t = 3 and 4.5% at t = 8. On a
small feature set they part by far more: 8 features at 4 residual df with an infinite prior give a t
of 5.49 a p of 4.0e-8 uncapped and 4.8e-6 on limma's 32 df. `ebayes.json` pins PRISM to R limma's
own `eBayes`, and `toolkit_end_to_end.json` records both p-values for the same t.

## Cross-checking a PRISM result against the toolkit

A default PRISM run and a default toolkit run (26.9.0 or later) choose the same prior: on the
protein matrix both take limma-trend-intensity-n-peptides from its `n_peptides` column, and on the
peptide matrix both take limma-trend. So a cross-check of a default run needs no prior setting at
all - only the pseudocount, and the peptide counts in the data the toolkit is given:

```python
cfg.moderation = None                # the default: chosen by level, as PRISM chooses it
cfg.log_pseudocount = 0              # PRISM adds none
# Protein level: include corrected_proteins.parquet's n_peptides column in the data, or the
# toolkit has no counts and chooses limma-trend. config.peptide_count_column defaults to it.
```

For a run of PRISM's deprecated `--prior intensity-trend`:

```python
cfg.moderation = "intensity-trend"   # warns: deprecated in the toolkit too
cfg.log_pseudocount = 0
# For a prior on the QC/reference injections, as PRISM's intensity trend takes when the run has them:
cfg.variance_prior_group_column = "sample_type"
cfg.variance_prior_groups = ["reference", "qc"]
```

On toolkit 26.8.0 or earlier, compare t, not P.Value, when the prior df is infinite or few features
are tested: PRISM's p is on limma's capped df (`DifferentialResult.DfTotal`) and that toolkit's is
not (see the df cap above). Read the PRISM side from the same matrix PRISM reported on
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
