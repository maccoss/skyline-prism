using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using MathNet.Numerics.Distributions;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>One feature's differential-abundance result (a volcano point / hit-table row).</summary>
public sealed record DifferentialRow(
    string FeatureId,
    double LogFc,
    double Fc,
    double AveExpr,
    double T,
    double PValue,
    double AdjPValue,
    double MeanA,
    double MeanB);

/// <summary>A covariate to adjust the contrast for (Sex, PMI, batch, ...).</summary>
public abstract class Covariate
{
    protected Covariate(string name, string? typeReason)
    {
        Name = name;
        TypeReason = typeReason ?? "constructed directly";
    }

    /// <summary>Covariate name, used to label design columns.</summary>
    public string Name { get; }

    /// <summary>How the covariate enters the design.</summary>
    public abstract CovariateKind Kind { get; }

    /// <summary>
    /// Why it has that type: the inference rule that decided it (<see cref="CovariateTyping.Infer"/>),
    /// or that it was set explicitly.
    /// </summary>
    public string TypeReason { get; }

    /// <summary>The name with its type, as the outputs show it: <c>Patient (categorical)</c>.</summary>
    public string Describe() => $"{Name} ({CovariateTyping.Token(Kind)})";

    /// <summary>
    /// Build a covariate from raw metadata strings (null or blank = missing). With no
    /// <paramref name="kind"/> the type is <see cref="CovariateTyping.Infer"/>'s - numbers alone no
    /// longer make a column numeric; see there for why.
    /// </summary>
    /// <param name="name">The metadata column.</param>
    /// <param name="values">Its value per matrix column.</param>
    /// <param name="kind">The type to use, overriding the inferred one.</param>
    /// <param name="inferFrom">The values to infer the type from, when they are not
    /// <paramref name="values"/> - the whole column, when these are aligned to a subset of samples.</param>
    /// <exception cref="ArgumentException">Numeric was asked for, but a value is not a number.</exception>
    public static Covariate FromMetadata(string name, string?[] values, CovariateKind? kind = null,
        IReadOnlyList<string?>? inferFrom = null)
    {
        var guess = CovariateTyping.Infer(name, inferFrom ?? values);
        var chosen = kind ?? guess.Kind;
        var reason = kind is null
            ? "inferred: " + guess.Reason
            : kind == guess.Kind
                ? "set explicitly"
                : $"set explicitly; it would otherwise be {CovariateTyping.Token(guess.Kind)}, because {guess.Reason}";

        if (chosen == CovariateKind.Numeric)
        {
            var nums = new double[values.Length];
            for (var i = 0; i < values.Length; i++)
            {
                var v = values[i];
                if (CovariateTyping.IsMissing(v))
                    nums[i] = double.NaN;
                else if (CovariateTyping.TryNumber(v!.Trim(), out var x))
                    nums[i] = x;
                else
                    throw new ArgumentException(
                        $"'{name}' cannot be numeric: '{v}' is not a number. Fit it as categorical.");
            }

            return new NumericCovariate(name, nums, reason);
        }

        // A column of numbers used as categories keeps one spelling per level, so "1" and "1.0" - as a
        // spreadsheet export can mix them - are one level rather than two.
        var labels = new string?[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            var v = values[i];
            labels[i] = CovariateTyping.IsMissing(v) ? null
                : guess.AllNumeric && CovariateTyping.TryNumber(v!.Trim(), out var x)
                    ? x.ToString("R", CultureInfo.InvariantCulture)
                    : v;
        }

        return new CategoricalCovariate(name, labels, reason);
    }
}

/// <summary>
/// A continuous covariate. <see cref="Values"/> is indexed by matrix column (all samples), aligned to
/// the abundance matrix; <see cref="double.NaN"/> marks a missing value. A covariate missing in any
/// selected sample is skipped, as is a constant one.
/// </summary>
public sealed class NumericCovariate : Covariate
{
    public NumericCovariate(string name, double[] values, string? typeReason = null)
        : base(name, typeReason) => Values = values;

    /// <inheritdoc />
    public override CovariateKind Kind => CovariateKind.Numeric;

    /// <summary>Per-column values (aligned to the abundance matrix columns).</summary>
    public double[] Values { get; }
}

/// <summary>
/// A categorical covariate, dummy-encoded with the first level dropped (like limma's factor handling).
/// <see cref="Values"/> is indexed by matrix column; <c>null</c> marks a missing value (skips the
/// whole covariate). A dummy level collinear with the group contrast is dropped, and so is a
/// covariate nested in the groups.
/// </summary>
public sealed class CategoricalCovariate : Covariate
{
    public CategoricalCovariate(string name, string?[] values, string? typeReason = null)
        : base(name, typeReason) => Values = values;

    /// <inheritdoc />
    public override CovariateKind Kind => CovariateKind.Categorical;

    /// <summary>Per-column category labels (aligned to the abundance matrix columns).</summary>
    public string?[] Values { get; }
}

/// <summary>Result of <see cref="Differential.Run"/>: per-feature rows plus run-level summary.</summary>
public sealed class DifferentialResult
{
    internal DifferentialResult(IReadOnlyList<DifferentialRow> rows, int nA, int nB,
        int nFeaturesTotal, int nFeaturesTested, double dfResidual, double dfPrior,
        string variancePrior, IReadOnlyList<string> covariatesUsed, IReadOnlyList<string> messages,
        IReadOnlyList<string> warnings, double trendRange = double.NaN, int nSubjects = 0,
        double priorLevel = double.NaN, IReadOnlyList<string>? covariatesAdjusted = null)
    {
        CovariatesAdjusted = covariatesAdjusted ?? Array.Empty<string>();
        TrendRange = trendRange;
        NSubjects = nSubjects;
        PriorLevel = priorLevel;
        Rows = rows;
        NA = nA;
        NB = nB;
        NFeaturesTotal = nFeaturesTotal;
        NFeaturesTested = nFeaturesTested;
        DfResidual = dfResidual;
        DfPrior = dfPrior;
        VariancePrior = variancePrior;
        CovariatesUsed = covariatesUsed;
        Messages = messages;
        Warnings = warnings;
    }

    /// <summary>Tested features, ascending by <see cref="DifferentialRow.PValue"/>.</summary>
    public IReadOnlyList<DifferentialRow> Rows { get; }

    /// <summary>Group A / group B sample counts.</summary>
    public int NA { get; }

    /// <summary>Group A / group B sample counts.</summary>
    public int NB { get; }

    /// <summary>Total features supplied.</summary>
    public int NFeaturesTotal { get; }

    /// <summary>Features tested (complete across every selected sample).</summary>
    public int NFeaturesTested { get; }

    /// <summary>Features dropped for having a missing value in a selected sample.</summary>
    public int NFeaturesDropped => NFeaturesTotal - NFeaturesTested;

    /// <summary>Residual degrees of freedom, <c>n_samples - n_coef</c>.</summary>
    public double DfResidual { get; }

    /// <summary>Empirical-Bayes prior degrees of freedom. May be <see cref="double.PositiveInfinity"/>.</summary>
    public double DfPrior { get; }

    /// <summary>Which variance prior was fitted (<c>global</c>, <c>intensity_trend</c>, ...).</summary>
    public string VariancePrior { get; }

    /// <summary>
    /// The factor the intensity trend was scaled by to fit this design's residuals; NaN for every
    /// other prior.
    /// </summary>
    /// <remarks>
    /// The trend's source sets its shape only, and this is how far its own level was from the noise
    /// the design is tested against. Above 1 means the residuals are noisier than the trend's source:
    /// the usual case for control pools, which lack the study's biology. Below 1 is usual for design
    /// groups under a paired or within-subject model, whose subject block removes spread the groups
    /// still contain.
    /// </remarks>
    public double PriorLevel { get; }

    /// <summary>
    /// One line on how the intensity-trend prior was fitted, for a status line or a file header
    /// (<c>prior: intensity trend from the controls, scaled x2.01 to these residuals; prior df 3.2</c>);
    /// null for every other prior.
    /// </summary>
    public string? DescribePriorFit() => DescribePriorFitBody() is { } body ? "prior: " + body : null;

    /// <summary>
    /// <see cref="DescribePriorFit"/> without its <c>prior: </c> label, for a place that names the
    /// field itself, like the quant report's parameter table.
    /// </summary>
    public string? DescribePriorFitBody()
    {
        if (!double.IsFinite(PriorLevel))
            return null;
        var source = VariancePrior.EndsWith("controls", StringComparison.Ordinal) ? "controls" : "design groups";
        var df = double.IsPositiveInfinity(DfPrior) ? "inf" : Significant(DfPrior);
        return $"intensity trend from the {source}, scaled x{Significant(PriorLevel)} to these residuals; prior df {df}";
    }

    // Three significant figures, so a small factor is not rounded away: under a fixed "0.###" a
    // level below 0.0005 printed as "x0", which reads as a fit that failed. From 100 up, whole
    // numbers, rather than the exponent "G3" would give.
    private static string Significant(double v) =>
        Math.Abs(v) >= 100
            ? v.ToString("0", CultureInfo.InvariantCulture)
            : v.ToString("G3", CultureInfo.InvariantCulture);

    /// <summary>Design columns actually used, excluding the intercept and group term.</summary>
    public IReadOnlyList<string> CovariatesUsed { get; }

    /// <summary>
    /// The covariates that made it into the design, each with the type it was fitted as
    /// (<c>Patient (categorical)</c>, <c>Age (numeric)</c>) - what a reader needs to know what
    /// "adjusted for" meant, where <see cref="CovariatesUsed"/> lists design columns.
    /// </summary>
    public IReadOnlyList<string> CovariatesAdjusted { get; }

    /// <summary>Notes about covariates skipped or levels dropped during design construction.</summary>
    public IReadOnlyList<string> Messages { get; }

    /// <summary>Diagnostic messages from the variance-moderation step.</summary>
    public IReadOnlyList<string> Warnings { get; }

    /// <summary>
    /// The span of the trend column actually used, <c>x_max - x_min</c>; NaN for a two-arm contrast.
    /// </summary>
    /// <remarks>
    /// Recorded because <see cref="DifferentialRow.LogFc"/> on a trend is the modeled change across
    /// THIS span, not the raw slope - so the slope is <c>LogFc / TrendRange</c>, and without the
    /// span a reader cannot recover it. Reporting the change rather than the slope is what lets one
    /// effect-size threshold mean the same thing on both designs, whatever units the column is in.
    /// </remarks>
    public double TrendRange { get; }

    /// <summary>
    /// Subjects contributing to a within-subject trend; 0 for every other design.
    /// </summary>
    public int NSubjects { get; }

    /// <summary>Whether this result came from a trend design.</summary>
    public bool IsTrend => !double.IsNaN(TrendRange);
}

/// <summary>
/// Two-group differential abundance via a limma empirical-Bayes moderated t-test, ported from the
/// PRISM Differential Explorer's <c>differential</c> (prism_diff_explorer.py). Fits the shared design
/// [intercept, groupB] per feature (<see cref="LinearModel"/>), moderates the residual variances
/// (<see cref="EmpiricalBayes"/>, under the prior <see cref="DifferentialOptions.Prior"/> selects), and
/// reports the moderated t, its two-sided p-value on <c>df_residual + df_prior</c> degrees of freedom,
/// and the adjusted p-value under <see cref="DifferentialOptions.Correction"/>. B is the treatment arm,
/// so a positive <c>logFC</c> is higher in B.
///
/// <para>The simple tests (<see cref="SimpleTests"/>) and the paired designs route through the same
/// <see cref="Run(double[,], IReadOnlyList{string}, IReadOnlyList{int}, IReadOnlyList{int}, DifferentialOptions)"/>
/// entry point, so every caller gets the same row shape whichever estimator ran.</para>
/// </summary>
public static class Differential
{
    /// <summary>
    /// Run the moderated-t contrast on a LOG2 abundance matrix. <paramref name="groupAColumns"/> and
    /// <paramref name="groupBColumns"/> are disjoint column indices into
    /// <paramref name="exprLog2FeaturesBySamples"/>; only features observed (non-NaN) in every selected
    /// sample are tested. Optional <paramref name="covariates"/> adjust the contrast; the intensity-trend
    /// variance prior is not yet supported.
    /// </summary>
    public static DifferentialResult Run(
        double[,] exprLog2FeaturesBySamples,
        IReadOnlyList<string> featureIds,
        IReadOnlyList<int> groupAColumns,
        IReadOnlyList<int> groupBColumns,
        int minPerGroup = 2,
        IReadOnlyList<Covariate>? covariates = null,
        bool trend = false)
        => Run(exprLog2FeaturesBySamples, featureIds, groupAColumns, groupBColumns,
            new DifferentialOptions
            {
                MinPerGroup = minPerGroup,
                Covariates = covariates,
                // The historical meaning of this flag: limma's trend, not the toolkit's. Callers that
                // want the lab's default prior pass DifferentialOptions instead.
                Prior = trend ? VariancePrior.LimmaTrend : VariancePrior.Global,
            });

    /// <summary>
    /// The moderated-t contrast under an explicit <see cref="DifferentialOptions"/>.
    /// </summary>
    public static DifferentialResult Run(
        double[,] exprLog2FeaturesBySamples,
        IReadOnlyList<string> featureIds,
        IReadOnlyList<int> groupAColumns,
        IReadOnlyList<int> groupBColumns,
        DifferentialOptions options)
    {
        // A trend has no arms, so it cannot come through the two-arm entry point. Refused rather
        // than fallen through: only Paired is handled below, so a trend request would otherwise run
        // an ordinary two-arm contrast while Describe() reported a linear trend.
        if (options.Design is DifferentialDesign.LinearTrend
            or DifferentialDesign.LinearTrendWithinSubject)
            throw new ArgumentException(
                "A linear-trend design has no A and B arms - call Differential.RunTrend with the "
                + "sample columns and their trend values instead.");

        var minPerGroup = options.MinPerGroup;
        var covariates = options.Covariates;
        var pairingMessages = new List<string>();
        IReadOnlyList<SamplePair>? pairs = null;

        if (options.Design == DifferentialDesign.Paired)
        {
            if (options.SubjectLabels is null)
                // Worded for the person reading it, not the caller: both front ends show this
                // message as it stands, and it used to name a property of this options type.
                throw new ArgumentException(
                    "A paired design needs a subject column: the metadata column that identifies "
                    + "each subject, so their two samples can be matched.");

            var resolved = PairedSamples.Resolve(options.SubjectLabels, groupAColumns, groupBColumns);
            pairs = resolved.Pairs;
            pairingMessages.AddRange(resolved.Messages);
            if (pairs.Count < minPerGroup)
                throw new ArgumentException(
                    $"A paired design needs at least {minPerGroup} matched subjects; {pairs.Count} "
                    + "matched. " + string.Join(" ", resolved.Messages));

            // From here the arms ARE the matched pairs, in a common subject order. Everything
            // downstream - the design's subject block, the complete-case filter, the per-group means
            // - then lines up by position, and an unmatched sample cannot leak into one arm only.
            groupAColumns = pairs.Select(pair => pair.AColumn).ToList();
            groupBColumns = pairs.Select(pair => pair.BColumn).ToList();
        }

        if (options.Test is DifferentialTest.PairedT or DifferentialTest.Wilcoxon)
        {
            if (pairs is null)
                throw new ArgumentException(
                    $"{options.Test} is a paired test and needs the Paired design.");
            return SimpleTests.RunPaired(exprLog2FeaturesBySamples, featureIds, pairs, options.Test,
                options.Correction, minPerGroup, Messages(options, pairingMessages));
        }

        if (options.Test != DifferentialTest.ModeratedT)
            return RunSimple(exprLog2FeaturesBySamples, featureIds, groupAColumns, groupBColumns,
                options, pairingMessages);
        var nFeatures = exprLog2FeaturesBySamples.GetLength(0);
        var nColumns = exprLog2FeaturesBySamples.GetLength(1);
        if (featureIds.Count != nFeatures)
            throw new ArgumentException(
                $"featureIds has {featureIds.Count} entries but the matrix has {nFeatures} features.",
                nameof(featureIds));

        var nA = groupAColumns.Count;
        var nB = groupBColumns.Count;
        if (nA < minPerGroup || nB < minPerGroup)
            throw new ArgumentException(
                $"Each group needs at least {minPerGroup} samples (A has {nA}, B has {nB}).");

        var seen = new HashSet<int>();
        foreach (var c in groupAColumns.Concat(groupBColumns))
        {
            if (c < 0 || c >= nColumns)
                throw new ArgumentException($"Sample column index {c} is out of range.");
            if (!seen.Add(c))
                throw new ArgumentException("Groups A and B must be disjoint (a sample is in both).");
        }

        // Selected columns in order [A..., B...]; design is [intercept, groupB].
        var cols = new int[nA + nB];
        for (var i = 0; i < nA; i++)
            cols[i] = groupAColumns[i];
        for (var i = 0; i < nB; i++)
            cols[nA + i] = groupBColumns[i];
        var nSamples = cols.Length;

        var (design, covariatesUsed, messages) = TwoArmDesign(nA, nB, cols, covariates, pairs?.Count);
        messages.InsertRange(0, pairingMessages);
        return Moderate(exprLog2FeaturesBySamples, featureIds, cols, design, options,
            PriorGroups(options, cols, nA, nB), covariatesUsed, messages,
            nA, nB, trendRange: double.NaN, nSubjects: 0);
    }

    /// <summary>
    /// Run a LINEAR TREND contrast: fit a slope against a numeric column and test whether it
    /// differs from zero.
    /// </summary>
    /// <remarks>
    /// <para><paramref name="xValues"/> is indexed by MATRIX COLUMN, like a metadata column, not by
    /// position in <paramref name="sampleColumns"/>.</para>
    ///
    /// <para><b>The reported effect is the modeled change across the observed range of x</b>, not
    /// the raw slope - see <see cref="DifferentialRow.LogFc"/> and
    /// <see cref="DifferentialResult.TrendRange"/>. A slope is in log2 per unit of x, so a threshold
    /// on it would mean something different for a column in days than for the same column in hours;
    /// the change across the range is unit-free and, for a two-level x coded 0/1, is exactly the
    /// log2 fold change. That is what lets one effect-size cut serve both designs.</para>
    ///
    /// <para>Under <see cref="DifferentialDesign.LinearTrendWithinSubject"/> a fixed-effect subject
    /// block is added, so the slope is estimated within subject. Without it, repeated measures on
    /// one subject are treated as independent and the standard error is understated.</para>
    /// </remarks>
    public static DifferentialResult RunTrend(
        double[,] exprLog2FeaturesBySamples,
        IReadOnlyList<string> featureIds,
        IReadOnlyList<int> sampleColumns,
        double[] xValues,
        DifferentialOptions options)
    {
        if (options.Design is not (DifferentialDesign.LinearTrend
            or DifferentialDesign.LinearTrendWithinSubject))
            throw new ArgumentException("RunTrend needs a linear-trend design.");

        var withinSubject = options.Design == DifferentialDesign.LinearTrendWithinSubject;
        var selection = TrendSamples.Resolve(
            sampleColumns, xValues, options.SubjectLabels?.ToArray(), withinSubject);
        var messages = new List<string>(selection.Messages);
        var cols = selection.Columns.ToArray();
        var x = selection.X.ToArray();

        if (cols.Length < Math.Max(3, options.MinPerGroup))
            throw new ArgumentException(
                $"A trend needs at least {Math.Max(3, options.MinPerGroup)} usable samples; "
                + $"{cols.Length} remained. " + string.Join(" ", messages));

        var range = x.Max() - x.Min();
        if (!(range > 0))
            throw new ArgumentException(
                "The trend column takes only one value across the selected samples, so there is no "
                + "slope to fit.");

        // [intercept, x, covariates...]. x is centered for the same reason a numeric covariate is:
        // it leaves the intercept meaning the abundance at the MEAN of x rather than at x = 0,
        // which for a column like year-of-birth is far outside the data.
        var (design, covariatesUsed, designMessages) =
            TrendDesign(x, cols, options.Covariates, withinSubject ? selection : null);
        messages.AddRange(designMessages);

        return Moderate(exprLog2FeaturesBySamples, featureIds, cols, design, options,
            PriorGroups(options, cols, nA: 0, nB: 0), covariatesUsed, messages,
            nA: cols.Length, nB: 0, trendRange: range, nSubjects: selection.SubjectCount);
    }

    /// <summary>
    /// Fit the shared design, moderate the variances and build the rows. Everything after the
    /// design matrix is the same for every design, so it lives here once.
    /// </summary>
    /// <remarks>
    /// <paramref name="trendRange"/> being finite is what marks a trend: the coefficient is then
    /// scaled to the change across that range before it is reported, and the two per-arm means
    /// become the fitted ends of the line rather than group averages. Everything else - the fit,
    /// the prior, the moderated t, the correction, the ordering - is identical, which is the point.
    /// </remarks>
    private static DifferentialResult Moderate(
        double[,] exprLog2FeaturesBySamples,
        IReadOnlyList<string> featureIds,
        int[] cols,
        double[,] design,
        DifferentialOptions options,
        IReadOnlyList<IReadOnlyList<int>> priorGroups,
        List<string> covariatesUsed,
        List<string> messages,
        int nA,
        int nB,
        double trendRange,
        int nSubjects)
    {
        const int coefIdx = 1; // the tested term is always the second design column
        var isTrend = !double.IsNaN(trendRange);
        var nFeatures = exprLog2FeaturesBySamples.GetLength(0);
        var nSamples = cols.Length;
        var nParams = design.GetLength(1);
        if (nSamples - nParams < 1)
            throw new ArgumentException(
                $"Not enough residual degrees of freedom (n={nSamples}, params={nParams}). "
                + ColumnsPerCovariate(options.Covariates, covariatesUsed)
                + "Use more samples or fewer covariates.");
        if (LinAlg.MatrixRank(design) < nParams)
            throw new ArgumentException(
                "Design matrix is rank-deficient (covariates collinear with each other or with group).");

        // Complete-case filter: a feature is tested only if observed in every selected sample.
        var tested = new List<int>(nFeatures);
        for (var f = 0; f < nFeatures; f++)
        {
            var complete = true;
            for (var s = 0; s < nSamples; s++)
                if (double.IsNaN(exprLog2FeaturesBySamples[f, cols[s]]))
                {
                    complete = false;
                    break;
                }

            if (complete)
                tested.Add(f);
        }

        if (tested.Count == 0)
            throw new InvalidOperationException("No feature is observed across every selected sample.");

        var nTested = tested.Count;
        var mk = new double[nTested, nSamples];
        for (var i = 0; i < nTested; i++)
        for (var s = 0; s < nSamples; s++)
            mk[i, s] = exprLog2FeaturesBySamples[tested[i], cols[s]];

        var fit = LinearModel.Fit(mk, design);
        var variances = new double[nTested];
        for (var i = 0; i < nTested; i++)
            variances[i] = fit.Sigma[i] * fit.Sigma[i];

        var (squeezed, variancePrior, priorLevel) = FitPrior(
            options, exprLog2FeaturesBySamples, priorGroups, variances, fit, tested, messages, cols);

        var dfTotal = fit.DfResidual + squeezed.DfPrior;
        var stdevUnscaled = fit.StdevUnscaled[coefIdx];

        var pValues = new double[nTested];
        var rows = new DifferentialRow[nTested];
        var groupB = new double[nB];
        var groupA = new double[isTrend ? 0 : nA];
        for (var i = 0; i < nTested; i++)
        {
            var coef = fit.Coefficients[i, coefIdx];
            var t = coef / (stdevUnscaled * Math.Sqrt(squeezed.VarPost[i]));
            var p = ModeratedPValue(t, dfTotal);
            pValues[i] = p;

            // The reported effect. On a trend the coefficient is log2 per unit of x, so it is
            // scaled to the span actually measured; on a two-arm contrast the span is 1 by
            // construction and this is the coefficient itself.
            var effect = isTrend ? coef * trendRange : coef;

            double meanA, meanB;
            if (isTrend)
            {
                // The fitted line's two ends, centered on the feature's own mean - so their
                // difference is exactly the reported effect. An intercept-based pair would be the
                // REFERENCE subject's trajectory under a within-subject design, not the average one.
                meanA = fit.Amean[i] - effect / 2.0;
                meanB = fit.Amean[i] + effect / 2.0;
            }
            else
            {
                for (var s = 0; s < nA; s++)
                    groupA[s] = mk[i, s];
                for (var s = 0; s < nB; s++)
                    groupB[s] = mk[i, nA + s];
                meanA = NumpyMath.Mean(groupA);
                meanB = NumpyMath.Mean(groupB);
            }

            rows[i] = new DifferentialRow(
                featureIds[tested[i]], effect, Math.Pow(2.0, effect), fit.Amean[i], t, p,
                double.NaN, meanA, meanB);
        }

        var adj = Fdr.Adjust(pValues, options.Correction);
        for (var i = 0; i < nTested; i++)
            rows[i] = rows[i] with { AdjPValue = adj[i] };

        var ordered = OrderByPValue(rows);

        return new DifferentialResult(ordered, nA, nB, nFeatures, nTested, fit.DfResidual,
            squeezed.DfPrior, variancePrior, covariatesUsed, messages, squeezed.Warnings,
            trendRange, nSubjects, priorLevel, DescribeAdjusted(options.Covariates, covariatesUsed));
    }

    /// <summary>
    /// Which covariates the design columns came from, for the residual-df refusal - a categorical
    /// covariate takes one column per level after the first, so one with many levels can use up the
    /// samples on its own, and the refusal has to say which.
    /// </summary>
    private static string ColumnsPerCovariate(IReadOnlyList<Covariate>? covariates, List<string> used)
    {
        if (covariates is not { Count: > 0 })
            return string.Empty;
        var inDesign = new HashSet<string>(used, StringComparer.Ordinal);
        var parts = covariates
            .Select(c => (c, n: c switch
            {
                NumericCovariate => inDesign.Contains(c.Name) ? 1 : 0,
                CategoricalCovariate cat => cat.Values.Where(v => v is not null).Distinct()
                    .Count(v => inDesign.Contains($"{c.Name}_{v}")),
                _ => 0,
            }))
            .Where(x => x.n > 0)
            .Select(x => $"{x.c.Describe()} takes {x.n} column{(x.n == 1 ? "" : "s")}")
            .ToList();
        if (parts.Count == 0)
            return string.Empty;
        var hint = covariates.Any(c => c.Kind == CovariateKind.Categorical)
            ? " A categorical covariate takes a column per level after the first; one with many levels may be meant as numeric."
            : string.Empty;
        return string.Join("; ", parts) + "." + hint + " ";
    }

    /// <summary>
    /// Each covariate that contributed at least one design column, named with its type. A numeric
    /// covariate's column carries its name; a categorical one's columns are <c>name_level</c>.
    /// </summary>
    private static List<string> DescribeAdjusted(IReadOnlyList<Covariate>? covariates, List<string> used)
    {
        var inDesign = new HashSet<string>(used, StringComparer.Ordinal);
        return covariates?
            .Where(c => c switch
            {
                NumericCovariate => inDesign.Contains(c.Name),
                CategoricalCovariate cat => cat.Values.Any(v => v is not null && inDesign.Contains($"{c.Name}_{v}")),
                _ => false,
            })
            .Select(c => c.Describe())
            .ToList() ?? new List<string>();
    }

    /// <summary>
    /// Order rows by p-value with NaN LAST.
    /// </summary>
    /// <remarks>
    /// A plain <c>OrderBy(r =&gt; r.PValue)</c> puts them FIRST: .NET's double comparer ranks NaN
    /// below every number, so the features a test could not evaluate at all - a constant row, an
    /// empty arm - lead the hit table, the CSV and anything that takes the top n. The tests return
    /// NaN specifically to avoid an infinity sorting to the top; sorting has to agree with that
    /// intent rather than quietly undo it.
    /// </remarks>
    internal static DifferentialRow[] OrderByPValue(IEnumerable<DifferentialRow> rows) =>
        rows.OrderBy(r => double.IsNaN(r.PValue) ? 1 : 0)
            .ThenBy(r => r.PValue)
            .ToArray();

    /// <summary>
    /// Build the design matrix [intercept, groupB, covariate columns...]. Numeric covariates are
    /// mean-centered; categorical covariates are dummy-encoded with the first (sorted) level dropped.
    /// A covariate missing in any selected sample, or constant, is skipped; a dummy level collinear
    /// with the group is dropped. Every skip/drop is recorded in the returned messages.
    /// </summary>

    /// <summary>
    /// Append a fixed-effect subject block: one indicator per subject after the first, which the
    /// intercept already spans.
    /// </summary>
    /// <remarks>
    /// The general form of <see cref="WithSubjectBlock"/>, which can assume exactly two samples per
    /// subject in a known order. Here a subject has any number of samples, so the block is built
    /// from an explicit subject index per sample.
    /// </remarks>
    private static double[,] WithSubjectDummies(double[,] design, IReadOnlyList<int> subjectOf, int nSubjects)
    {
        var nSamples = design.GetLength(0);
        var nParams = design.GetLength(1);
        if (nSubjects < 2)
            return design;

        var widened = new double[nSamples, nParams + nSubjects - 1];
        for (var s = 0; s < nSamples; s++)
        {
            for (var c = 0; c < nParams; c++)
                widened[s, c] = design[s, c];
            // Subject 0 is the reference level and gets no column.
            if (subjectOf[s] > 0)
                widened[s, nParams + subjectOf[s] - 1] = 1.0;
        }

        return widened;
    }

    /// <summary>
    /// Drop covariate columns a subject block would make redundant, naming them.
    /// </summary>
    /// <remarks>
    /// A covariate constant within every subject - sex, genotype, birth year, the usual things to
    /// tick - is exactly a linear combination of the subject block, so leaving it in makes the
    /// design rank-deficient and the run dies on a check that names neither the covariate nor a
    /// block the user never asked for. This is not a limitation of the implementation: a
    /// within-subject slope cannot estimate a between-subject effect, because the block has already
    /// absorbed it.
    /// </remarks>
    private static (double[,] Design, List<string> Used) DropWithinSubjectCollinear(
        double[,] design, List<string> covariatesUsed, IReadOnlyList<int> subjectOf,
        List<string> messages)
    {
        var nSamples = design.GetLength(0);
        var nParams = design.GetLength(1);
        const int firstCovariate = 2; // [intercept, term, covariates...]
        if (nParams <= firstCovariate)
            return (design, covariatesUsed);

        var firstRowOfSubject = new Dictionary<int, int>();
        for (var s = 0; s < nSamples; s++)
            if (!firstRowOfSubject.ContainsKey(subjectOf[s]))
                firstRowOfSubject[subjectOf[s]] = s;

        var keep = new List<int>();
        var kept = new List<string>();
        var dropped = new List<string>();
        for (var c = 0; c < nParams; c++)
        {
            if (c < firstCovariate)
            {
                keep.Add(c);
                continue;
            }

            var constantWithinSubject = true;
            for (var s = 0; s < nSamples && constantWithinSubject; s++)
                if (design[s, c] != design[firstRowOfSubject[subjectOf[s]], c])
                    constantWithinSubject = false;

            var name = covariatesUsed[c - firstCovariate];
            if (constantWithinSubject)
                dropped.Add(name);
            else
            {
                keep.Add(c);
                kept.Add(name);
            }
        }

        if (dropped.Count == 0)
            return (design, covariatesUsed);

        messages.Add(
            $"{dropped.Count} covariate(s) are constant within every subject, so a within-subject "
            + $"slope cannot estimate them, and they were dropped ({string.Join(", ", dropped)}).");

        var reduced = new double[nSamples, keep.Count];
        for (var s = 0; s < nSamples; s++)
        for (var c = 0; c < keep.Count; c++)
            reduced[s, c] = design[s, keep[c]];

        return (reduced, kept);
    }

    /// <summary>
    /// Drop covariate columns that the subject block will make redundant, naming them.
    /// </summary>
    /// <remarks>
    /// A covariate constant within every subject carries no information a within-subject contrast
    /// can use - the subject block absorbs it entirely - so its column is exactly a linear
    /// combination of the block. Detected by the definition rather than by a rank test, because the
    /// definition is what can be explained to a reader.
    /// </remarks>
    private static (double[,] Design, List<string> Used) DropSubjectCollinear(
        double[,] design, List<string> covariatesUsed, int nPairs, List<string> messages)
    {
        var nSamples = design.GetLength(0);
        var nParams = design.GetLength(1);
        const int firstCovariate = 2; // [intercept, group, covariates...]
        if (nParams <= firstCovariate || nPairs < 1)
            return (design, covariatesUsed);

        var keep = new List<int>();
        var dropped = new List<string>();
        for (var c = 0; c < nParams; c++)
        {
            if (c < firstCovariate)
            {
                keep.Add(c);
                continue;
            }

            var constantWithinSubject = true;
            for (var j = 0; j < nPairs && constantWithinSubject; j++)
                if (design[j, c] != design[nPairs + j, c])
                    constantWithinSubject = false;

            if (constantWithinSubject)
            {
                var name = c - firstCovariate < covariatesUsed.Count
                    ? covariatesUsed[c - firstCovariate]
                    : $"column {c}";
                dropped.Add(name);
            }
            else
            {
                keep.Add(c);
            }
        }

        if (dropped.Count == 0)
            return (design, covariatesUsed);

        messages.Add($"Covariate(s) {string.Join(", ", dropped)} are constant within each subject, so "
            + "the paired design's subject block already accounts for them - they were dropped. A "
            + "within-subject contrast cannot estimate a between-subject effect.");

        var reduced = new double[nSamples, keep.Count];
        for (var r = 0; r < nSamples; r++)
            for (var k = 0; k < keep.Count; k++)
                reduced[r, k] = design[r, keep[k]];

        var used = covariatesUsed.Where(n => !dropped.Contains(n)).ToList();
        return (reduced, used);
    }

    /// <summary>
    /// Append a fixed-effect subject block to a paired design: one indicator per subject after the
    /// first, each marking that subject's two samples.
    /// </summary>
    /// <remarks>
    /// <para>This is what makes the contrast a WITHIN-subject one. Without it every subject's overall
    /// level is part of the residual, so between-subject variation - which a paired design exists to
    /// remove - inflates the variance and costs the test its power.</para>
    /// <para>A FIXED effect, not a random one, matching the lab's toolkit
    /// (<c>statistical_analysis.py:1321</c>). A random intercept would be a mixed model, which is a
    /// different estimator and is not implemented. The first subject is dropped, as always with
    /// indicator coding, because the intercept already spans it.</para>
    /// <para>The columns are positional: the caller has already reduced the arms to matched pairs in
    /// a common subject order, so subject j is at row j in arm A and row <c>nPairs + j</c> in arm B.</para>
    /// </remarks>
    private static double[,] WithSubjectBlock(double[,] design, int nPairs)
    {
        var nSamples = design.GetLength(0);
        var nParams = design.GetLength(1);
        var extra = nPairs - 1;
        if (extra <= 0)
            return design;

        var widened = new double[nSamples, nParams + extra];
        for (var r = 0; r < nSamples; r++)
        for (var c = 0; c < nParams; c++)
            widened[r, c] = design[r, c];

        for (var j = 1; j < nPairs; j++)
        {
            widened[j, nParams + j - 1] = 1.0;          // arm A half
            widened[nPairs + j, nParams + j - 1] = 1.0; // arm B half
        }

        return widened;
    }

    /// <summary>Pairing notes plus whatever the options themselves make unusable.</summary>
    private static List<string> Messages(DifferentialOptions options, List<string> pairingMessages)
    {
        var messages = new List<string>(pairingMessages);
        if (options.Covariates is { Count: > 0 } cov)
            messages.Add($"{options.Describe()} cannot adjust for covariates "
                + $"({string.Join(", ", cov.Select(c => c.Name))}) - it has no design matrix to put "
                + "them in. Use the moderated t for an adjusted contrast.");
        return messages;
    }

    /// <summary>
    /// What the design will do with <paramref name="covariate"/> in a two-arm contrast over these
    /// arms - skip it, drop it, warn about it - before anything is run: the messages
    /// <see cref="Run(double[,], IReadOnlyList{string}, IReadOnlyList{int}, IReadOnlyList{int}, DifferentialOptions)"/>
    /// would report for it, and none when it goes in as it is.
    /// </summary>
    /// <remarks>
    /// Produced by the same design builder the run uses, on the covariate alone, so a note shown
    /// beside a covariate cannot disagree with what the run then does with it. Alone, because the
    /// point is to say what is wrong with THIS column; a clash between two covariates still surfaces
    /// when the contrast runs.
    /// </remarks>
    public static IReadOnlyList<string> CovariateNotes(Covariate covariate,
        IReadOnlyList<int> groupAColumns, IReadOnlyList<int> groupBColumns, DifferentialOptions options)
    {
        // The same pairing the run does, so a paired design's preview sees the matched subjects and
        // the subject block - which is what drops a covariate constant within subject.
        int? nPairs = null;
        if (options.Design == DifferentialDesign.Paired)
        {
            if (options.SubjectLabels is null)
                return Array.Empty<string>();
            var pairs = PairedSamples.Resolve(options.SubjectLabels, groupAColumns, groupBColumns).Pairs;
            if (pairs.Count == 0)
                return Array.Empty<string>();
            groupAColumns = pairs.Select(p => p.AColumn).ToList();
            groupBColumns = pairs.Select(p => p.BColumn).ToList();
            nPairs = pairs.Count;
        }

        var cols = groupAColumns.Concat(groupBColumns).ToArray();
        return TwoArmDesign(groupAColumns.Count, groupBColumns.Count, cols, new[] { covariate }, nPairs).Messages;
    }

    /// <summary>
    /// <see cref="CovariateNotes(Covariate, IReadOnlyList{int}, IReadOnlyList{int}, DifferentialOptions)"/>
    /// for a trend over <paramref name="sampleColumns"/>, resolved as
    /// <see cref="RunTrend"/> resolves it (<paramref name="xValues"/> indexed by matrix column).
    /// </summary>
    public static IReadOnlyList<string> CovariateNotes(Covariate covariate, IReadOnlyList<int> sampleColumns,
        double[] xValues, DifferentialOptions options)
    {
        var withinSubject = options.Design == DifferentialDesign.LinearTrendWithinSubject;
        var selection = TrendSamples.Resolve(sampleColumns, xValues, options.SubjectLabels?.ToArray(), withinSubject);
        if (selection.Columns.Count < 2 || !(selection.X.Max() - selection.X.Min() > 0))
            return Array.Empty<string>();
        return TrendDesign(selection.X.ToArray(), selection.Columns.ToArray(), new[] { covariate },
            withinSubject ? selection : null).Messages;
    }

    /// <summary>
    /// The two-arm design as the run fits it: <c>[1, group, covariates]</c>, then - for a paired
    /// design over <paramref name="nPairs"/> matched subjects - covariates constant within subject
    /// dropped and the subject block added. One path for the run and for its preview.
    /// </summary>
    private static (double[,] Design, List<string> CovariatesUsed, List<string> Messages) TwoArmDesign(
        int nA, int nB, int[] cols, IReadOnlyList<Covariate>? covariates, int? nPairs)
    {
        var (design, covariatesUsed, messages) = BuildDesign(nA, nB, cols, covariates);
        if (nPairs is { } n)
        {
            // Sex, age, genotype, diagnosis - the most natural things to tick in a paired design -
            // are all constant within a subject, and therefore exactly collinear with the subject
            // block. Left in, the design is rank-deficient and the run dies on a check that names
            // neither the covariate nor a block the user never asked for. Dropping them here, and
            // saying so, matches how BuildDesign already handles a dummy collinear with the group.
            // Note this is not a limitation of the implementation: a within-subject contrast cannot
            // estimate a between-subject effect, because the subject block has already absorbed it.
            (design, covariatesUsed) = DropSubjectCollinear(design, covariatesUsed, n, messages);
            design = WithSubjectBlock(design, n);
        }

        return (design, covariatesUsed, messages);
    }

    /// <summary>
    /// The trend design as the run fits it: <c>[1, x - mean(x), covariates]</c>, plus - within
    /// subject - covariates constant within subject dropped and the subject indicators added.
    /// </summary>
    private static (double[,] Design, List<string> CovariatesUsed, List<string> Messages) TrendDesign(
        double[] x, int[] cols, IReadOnlyList<Covariate>? covariates, TrendSelection? withinSubject)
    {
        var mean = x.Average();
        var (design, covariatesUsed, messages) =
            BuildTrendDesign(x.Select(v => v - mean).ToArray(), cols, covariates);
        if (withinSubject is { } selection)
        {
            (design, covariatesUsed) = DropWithinSubjectCollinear(
                design, covariatesUsed, selection.SubjectOf, messages);
            design = WithSubjectDummies(design, selection.SubjectOf, selection.SubjectCount);
        }

        return (design, covariatesUsed, messages);
    }

    private static (double[,] Design, List<string> CovariatesUsed, List<string> Messages) BuildDesign(
        int nA, int nB, int[] cols, IReadOnlyList<Covariate>? covariates)
    {
        var nSamples = nA + nB;
        var grp = new double[nSamples];
        for (var s = nA; s < nSamples; s++)
            grp[s] = 1.0;
        return BuildDesign(grp, "group", cols, covariates);
    }

    /// <summary>
    /// <c>[intercept, x, covariates...]</c> for a trend, where x is the CENTERED trend value.
    /// </summary>
    /// <remarks>
    /// The covariate handling is shared with the two-arm design rather than copied, because it is
    /// the part with the rules worth keeping identical - missing values skipped, constants skipped,
    /// numerics centered, categoricals dummy-coded dropping the first sorted level.
    /// </remarks>
    private static (double[,] Design, List<string> CovariatesUsed, List<string> Messages)
        BuildTrendDesign(double[] centeredX, int[] cols, IReadOnlyList<Covariate>? covariates)
        => BuildDesign(centeredX, "the trend", cols, covariates);

    /// <summary>
    /// Build <c>[intercept, term, covariates...]</c>. <paramref name="term"/> is the column being
    /// tested - a group indicator for a two-arm contrast, a centered numeric column for a trend -
    /// and <paramref name="termName"/> names it in the "confounded with" message.
    /// </summary>
    private static (double[,] Design, List<string> CovariatesUsed, List<string> Messages) BuildDesign(
        double[] grp, string termName, int[] cols, IReadOnlyList<Covariate>? covariates)
    {
        var nSamples = grp.Length;
        var extra = new List<double[]>();
        var names = new List<string>();
        var messages = new List<string>();
        // A two-arm contrast's term is a 0/1 group indicator; a trend's is a centered numeric column,
        // which nothing can be nested in.
        var twoArm = grp.All(g => g is 0.0 or 1.0);

        if (covariates != null)
        {
            foreach (var cov in covariates)
            {
                if (cov is NumericCovariate num)
                {
                    var v = new double[nSamples];
                    var missing = false;
                    for (var s = 0; s < nSamples; s++)
                    {
                        v[s] = num.Values[cols[s]];
                        if (double.IsNaN(v[s]))
                            missing = true;
                    }

                    if (missing)
                    {
                        messages.Add($"Covariate '{num.Name}' has missing values in selected samples - skipped.");
                        continue;
                    }

                    if (Distinct(v) < 2)
                    {
                        messages.Add($"Covariate '{num.Name}' is constant - skipped.");
                        continue;
                    }

                    var mean = NumpyMath.Mean(v);
                    var centered = new double[nSamples];
                    for (var s = 0; s < nSamples; s++)
                        centered[s] = v[s] - mean;
                    extra.Add(centered);
                    names.Add(num.Name);

                    // Only reachable by asking for it: the inferred type of such a column is
                    // categorical. Fitted anyway, as asked, but not in silence - an ID fitted as a
                    // number is a straight-line effect of an arbitrary label.
                    if (CovariateTyping.CategoryWord(num.Name) is { } word)
                        messages.Add($"Covariate '{num.Name}' is fitted as numeric - one straight-line effect of "
                            + $"its value - although its name contains '{word}', which usually labels groups. If "
                            + "its values are IDs or batch labels, fit it as categorical. If it identifies "
                            + "subjects sampled more than once, no fixed covariate accounts for those "
                            + "repeated samples.");
                }
                else if (cov is CategoricalCovariate cat)
                {
                    var missing = false;
                    for (var s = 0; s < nSamples; s++)
                        if (cat.Values[cols[s]] is null)
                        {
                            missing = true;
                            break;
                        }

                    if (missing)
                    {
                        messages.Add($"Covariate '{cat.Name}' has missing values in selected samples - skipped.");
                        continue;
                    }

                    var vals = new string[nSamples];
                    for (var s = 0; s < nSamples; s++)
                        vals[s] = cat.Values[cols[s]]!;

                    // get_dummies(drop_first=True): sorted levels, drop the first, one indicator each.
                    var levels = vals.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToArray();
                    if (levels.Length < 2)
                    {
                        messages.Add($"Covariate '{cat.Name}' is constant - skipped.");
                        continue;
                    }

                    // Nested in the groups: every level lies within one arm. Its indicators then
                    // span the group column between them, so as a fixed effect it absorbs the very
                    // contrast being tested and the design cannot be fitted. Two levels nested is one
                    // indicator equal to the group, which the confounding check below already drops;
                    // three or more used to reach the generic rank-deficiency error, naming nothing.
                    // Dropped and named instead, the same treatment that confounded indicator gets.
                    if (twoArm && levels.Length > 2 && levels.All(level =>
                            Enumerable.Range(0, nSamples).Where(s => vals[s] == level)
                                .Select(s => grp[s]).Distinct().Count() == 1))
                    {
                        messages.Add($"Covariate '{cat.Name}' is nested within the groups - each of its values "
                            + "occurs in one arm only - so as a fixed effect it would absorb the contrast, "
                            + "which could then not be estimated. It is dropped. If it identifies subjects "
                            + "sampled more than once, no fixed covariate accounts for those repeated samples.");
                        continue;
                    }
                    for (var li = 1; li < levels.Length; li++)
                    {
                        var level = levels[li];
                        var col = new double[nSamples];
                        for (var s = 0; s < nSamples; s++)
                            col[s] = vals[s] == level ? 1.0 : 0.0;

                        if (Distinct(col) < 2)
                            continue;
                        if (AllClose(col, grp, complement: false) || AllClose(col, grp, complement: true))
                        {
                            messages.Add(
                                $"Covariate level '{cat.Name}_{level}' is confounded with {termName} - dropped.");
                            continue;
                        }

                        extra.Add(col);
                        names.Add($"{cat.Name}_{level}");
                    }
                }
            }
        }

        var nCoef = 2 + extra.Count;
        var design = new double[nSamples, nCoef];
        for (var s = 0; s < nSamples; s++)
        {
            design[s, 0] = 1.0;
            design[s, 1] = grp[s];
            for (var c = 0; c < extra.Count; c++)
                design[s, 2 + c] = extra[c][s];
        }

        return (design, names, messages);
    }

    /// <summary>Number of exactly-distinct values (numpy.unique semantics), enough to spot a constant.</summary>
    private static int Distinct(double[] v)
    {
        var set = new HashSet<double>();
        foreach (var x in v)
            set.Add(x);
        return set.Count;
    }

    /// <summary>
    /// numpy.allclose(a, complement ? 1-grp : grp) with numpy's default tolerances (rtol 1e-5, atol
    /// 1e-8): true when the dummy column exactly tracks (or complements) the group assignment.
    /// </summary>
    private static bool AllClose(double[] a, double[] grp, bool complement)
    {
        for (var i = 0; i < a.Length; i++)
        {
            var b = complement ? 1.0 - grp[i] : grp[i];
            if (Math.Abs(a[i] - b) > 1e-8 + 1e-5 * Math.Abs(b))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Two-sided moderated-t p-value: <c>2 * cdf(-|t|)</c> on <paramref name="dfTotal"/> degrees of
    /// freedom. An infinite prior gives an infinite total df, where the t-distribution is the standard
    /// normal (matching scipy's <c>t.cdf(df=inf)</c>).
    /// </summary>
    /// <summary>
    /// The non-moderated tests, which share this entry point but none of the linear-model machinery.
    /// </summary>
    /// <remarks>
    /// Covariates are reported as ignored rather than dropped in silence: a t-test has no design
    /// matrix to hold one, so a ticked covariate simply cannot be honored, and a reader who ticked
    /// one is entitled to know the contrast in front of them is unadjusted.
    /// </remarks>
    private static DifferentialResult RunSimple(
        double[,] exprLog2FeaturesBySamples,
        IReadOnlyList<string> featureIds,
        IReadOnlyList<int> groupAColumns,
        IReadOnlyList<int> groupBColumns,
        DifferentialOptions options,
        IReadOnlyList<string> pairingMessages)
    {
        var messages = new List<string>(pairingMessages);
        if (options.Covariates is { Count: > 0 } cov)
            messages.Add($"{options.Describe()} cannot adjust for covariates "
                + $"({string.Join(", ", cov.Select(c => c.Name))}) - it has no design matrix to put "
                + "them in. Use the moderated t for an adjusted contrast.");

        return SimpleTests.Run(exprLog2FeaturesBySamples, featureIds, groupAColumns, groupBColumns,
            options.Test, options.Correction, options.MinPerGroup, messages);
    }

    /// <summary>
    /// The variance prior the options asked for, with the name to report it under. Falls back to the
    /// global prior - saying so in <paramref name="messages"/> - whenever the requested one cannot be
    /// fitted, because a silently substituted prior is a silently different p-value.
    /// </summary>
    private static (SqueezeVarResult Squeezed, string Name, double Level) FitPrior(
        DifferentialOptions options, double[,] expr, IReadOnlyList<IReadOnlyList<int>> priorGroups,
        double[] variances, LinearModelFit fit, List<int> tested, List<string> messages,
        IReadOnlyCollection<int> fitColumns)
    {
        // Peptide counts arrive per FEATURE of the input matrix; the fit is over the tested subset,
        // so they have to be gathered in the same order or the trend would pair each variance with
        // another feature's count.
        double[]? counts = null;
        if (options.PeptideCounts is { } supplied)
        {
            counts = new double[tested.Count];
            for (var i = 0; i < tested.Count; i++)
                counts[i] = tested[i] < supplied.Count ? supplied[tested[i]] : double.NaN;
        }

        var needsCounts = options.Prior is VariancePrior.PeptideCount;
        if (needsCounts && counts is null)
        {
            messages.Add("This prior needs a peptide count per feature, which only the protein-level "
                + "matrix carries - the global prior was used instead.");
            return (EmpiricalBayes.SqueezeVarGlobal(variances, fit.DfResidual), "global", double.NaN);
        }

        switch (options.Prior)
        {
            case VariancePrior.LimmaTrend when fit.Amean.All(double.IsFinite):
                return (EmpiricalBayes.SqueezeVarTrend(variances, fit.DfResidual, fit.Amean),
                    "limma-trend", double.NaN);

            case VariancePrior.LimmaTrend:
                messages.Add("Mean expression has non-finite values - the limma-trend prior is "
                    + "unavailable, so the global prior was used.");
                break;

            case VariancePrior.IntensityTrend when priorGroups.Count == 0:
                // A trend design has no groups to take a within-group variance from, and the
                // estimator is only defined on groups. Inventing one - pooling every sample, say -
                // would fold the trend itself into the "noise" it is meant to describe, inflating
                // the prior for exactly the features that have a real slope and over-shrinking
                // them. So it degrades to the global prior and says how to get a real one back.
                messages.Add("The intensity-trend prior needs sample groups to take a within-group "
                    + "variance from, and a trend design has none - the global prior was used. Fit "
                    + "the prior on the QC and reference replicates to use it here.");
                break;

            case VariancePrior.IntensityTrend:
            {
                // The prior is cleanest when its samples take no part in the fit - the usual case,
                // since control injections carry no timepoint and no study condition. They CAN
                // overlap: a contrast whose own arm is the QC samples, or a trend over something
                // every sample has, like injection order. That is not an error and not corrected
                // here - the prior is a smoothed curve over thousands of features, so one sample's
                // influence on it is slight - but it is said out loud, because "fitted on the
                // controls" then means something weaker than it usually does.
                var shared = priorGroups.SelectMany(g => g).Distinct().Intersect(fitColumns).Count();
                if (shared > 0)
                    messages.Add($"{shared} sample(s) are both in the fit and in the variance "
                        + "prior's groups, so the prior is not independent of the data it moderates.");

                var shape = VariancePriors.IntensityTrend(expr, tested, priorGroups);
                if (shape is not null)
                {
                    // The groups give the trend its SHAPE. Its level and the prior df are fitted to
                    // this design's residuals, because neither source measures the noise the
                    // design is tested against - see WithCalibratedLevel. The source is still named:
                    // it decides the shape, and two results are comparable only on the same one.
                    var squeezed = WithCalibratedLevel(variances, fit.DfResidual, shape, out var level);
                    return (squeezed,
                        options.PriorGroupColumns is not null
                            ? "intensity-trend from controls"
                            : "intensity-trend from design groups",
                        level);
                }

                messages.Add("Too few usable (feature, group) points to fit the intensity trend - "
                    + "the global prior was used instead.");
                break;
            }

            case VariancePrior.PeptideCount:
            {
                var prior = VariancePriors.PeptideCountTrend(variances, counts!);
                if (prior is not null)
                    return (WithGlobalDf(variances, fit.DfResidual, prior), "peptide-count", double.NaN);
                messages.Add("Too few features carry a usable peptide count to fit the trend - the "
                    + "global prior was used instead.");
                break;
            }
        }

        return (EmpiricalBayes.SqueezeVarGlobal(variances, fit.DfResidual), "global", double.NaN);
    }

    /// <summary>
    /// The groups the variance prior is fitted over: the two contrast arms by default, or whichever
    /// replicates the caller nominated.
    /// </summary>
    /// <remarks>
    /// The override takes the intensity trend's SHAPE from dedicated QC or reference injections,
    /// which trace noise against intensity with no biology mixed in. Its level is fitted to the
    /// residuals either way (<see cref="WithCalibratedLevel"/>). Those replicates take no part in the
    /// contrast, so their columns exist only in the FULL matrix, which is why these indices are
    /// absolute.
    /// </remarks>
    private static IReadOnlyList<IReadOnlyList<int>> PriorGroups(
        DifferentialOptions options, int[] cols, int nA, int nB)
    {
        if (options.PriorGroupColumns is not null)
            return options.PriorGroupColumns;

        // No arms means no groups. Returning empty rather than one pooled group is deliberate: see
        // the IntensityTrend case in FitPrior for why a pooled "group" would be worse than none.
        return nA == 0 && nB == 0
            ? Array.Empty<IReadOnlyList<int>>()
            : new IReadOnlyList<int>[]
            {
                cols.Take(nA).ToArray(),
                cols.Skip(nA).Take(nB).ToArray(),
            };
    }

    /// <summary>
    /// A per-feature prior scale paired with the GLOBAL prior degrees of freedom - the DEqMS
    /// peptide-count prior, kept exactly as DEqMS defines it.
    /// </summary>
    /// <remarks>
    /// DEqMS is limma with one change: the prior scale follows a LOWESS of log residual variance on
    /// log peptide count, and limma's own global <c>d0</c> is kept. So this is NOT calibrated the way
    /// the intensity trend is (<see cref="WithCalibratedLevel"/>): the fitted curve keeps the low bias
    /// of a smoothed log variance, and <c>d0</c> is measured around the global mean rather than around
    /// the curve. That is deliberate. The estimator is DEqMS's, reproducing it is the requirement
    /// (<c>peptide_count_prior.json</c>), and a calibrated variant would be a different method under
    /// DEqMS's name.
    /// </remarks>
    private static SqueezeVarResult WithGlobalDf(double[] variances, double dfResidual, double[] prior)
    {
        var global = EmpiricalBayes.SqueezeVarGlobal(variances, dfResidual);
        return EmpiricalBayes.SqueezeVarWithScale(
            variances, dfResidual, prior, global.DfPrior, global.Warnings);
    }

    /// <summary>
    /// An intensity trend used for its SHAPE, with its level and the prior degrees of freedom fitted
    /// to the design's own residual variances.
    /// </summary>
    /// <remarks>
    /// <para>The trend is fitted on within-group variances, and no source of groups measures the noise
    /// a design is tested against. Control pools carry instrument noise only, while a study residual
    /// also carries biology. On the Verapamil serum cohort the residuals ran about 2x above the pools'
    /// trend. Design groups under a paired or within-subject model still contain the between-subject
    /// spread the subject block removes, about 0.7x on the same cohort. And a LOWESS of log variance
    /// is biased low by the log of a chi-square draw: a factor of 0.81 at six samples per group, 0.28
    /// at two.</para>
    /// <para>Up to dotnet-v26.27.0 the trend was used at its source's level, with the prior df taken
    /// from the residuals around their own global mean - a df measured against one level and applied
    /// to another. With controls, a large df made the posterior the pools' technical variance outright.
    /// On simulated nulls that put 40% of p-values under 0.05 for an unpaired contrast, and 0.17% for a
    /// paired one fitted on its design groups.</para>
    /// <para>The fix is limma's fitFDist with the trend as a known covariate offset: the moment fit
    /// runs on <c>s^2 / trend</c>, so its scale is the factor the trend is multiplied by and its df
    /// measures how tightly the residuals follow the scaled trend. It is the same estimator
    /// <c>proteomics-toolkit</c> uses from v26.8.0 (<c>_calibrate_trend_to_design</c>), and the
    /// composition is pinned to it by <c>toolkit_end_to_end.json</c>.</para>
    /// </remarks>
    /// <param name="level">The factor the trend was multiplied by.</param>
    private static SqueezeVarResult WithCalibratedLevel(
        double[] variances, double dfResidual, double[] shape, out double level)
    {
        var ratio = new double[variances.Length];
        for (var i = 0; i < ratio.Length; i++)
            ratio[i] = variances[i] / shape[i];
        var calibrated = EmpiricalBayes.SqueezeVarGlobal(ratio, dfResidual);
        level = calibrated.VarPrior[0];

        var prior = new double[shape.Length];
        for (var i = 0; i < prior.Length; i++)
            prior[i] = level * shape[i];
        return EmpiricalBayes.SqueezeVarWithScale(
            variances, dfResidual, prior, calibrated.DfPrior, calibrated.Warnings);
    }

    private static double ModeratedPValue(double t, double dfTotal)
    {
        return Distributions.TwoSidedT(t, dfTotal);
    }
}
