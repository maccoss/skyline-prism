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
    protected Covariate(string name) => Name = name;

    /// <summary>Covariate name, used to label design columns.</summary>
    public string Name { get; }

    /// <summary>
    /// Build a covariate from raw metadata strings, inferring the type the way pandas dtype inference
    /// does: numeric if every non-null value parses as an invariant-culture number (so an integer-coded
    /// batch is centered, not dummy-coded), otherwise categorical. Null entries are missing.
    /// </summary>
    public static Covariate FromMetadata(string name, string?[] values)
    {
        var anyNonNull = false;
        foreach (var v in values)
        {
            if (v is null)
                continue;
            anyNonNull = true;
            if (!double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out _))
                return new CategoricalCovariate(name, values);
        }

        if (!anyNonNull)
            return new CategoricalCovariate(name, values);

        var nums = new double[values.Length];
        for (var i = 0; i < values.Length; i++)
            nums[i] = values[i] is null
                ? double.NaN
                : double.Parse(values[i]!, NumberStyles.Float, CultureInfo.InvariantCulture);
        return new NumericCovariate(name, nums);
    }
}

/// <summary>
/// A continuous covariate. <see cref="Values"/> is indexed by matrix column (all samples), aligned to
/// the abundance matrix; <see cref="double.NaN"/> marks a missing value. A covariate missing in any
/// selected sample is skipped, as is a constant one.
/// </summary>
public sealed class NumericCovariate : Covariate
{
    public NumericCovariate(string name, double[] values) : base(name) => Values = values;

    /// <summary>Per-column values (aligned to the abundance matrix columns).</summary>
    public double[] Values { get; }
}

/// <summary>
/// A categorical covariate, dummy-encoded with the first level dropped (like limma's factor handling).
/// <see cref="Values"/> is indexed by matrix column; <c>null</c> marks a missing value (skips the
/// whole covariate). A dummy level collinear with the group contrast is dropped.
/// </summary>
public sealed class CategoricalCovariate : Covariate
{
    public CategoricalCovariate(string name, string?[] values) : base(name) => Values = values;

    /// <summary>Per-column category labels (aligned to the abundance matrix columns).</summary>
    public string?[] Values { get; }
}

/// <summary>Result of <see cref="Differential.Run"/>: per-feature rows plus run-level summary.</summary>
public sealed class DifferentialResult
{
    internal DifferentialResult(IReadOnlyList<DifferentialRow> rows, int nA, int nB,
        int nFeaturesTotal, int nFeaturesTested, double dfResidual, double dfPrior,
        string variancePrior, IReadOnlyList<string> covariatesUsed, IReadOnlyList<string> messages,
        IReadOnlyList<string> warnings)
    {
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

    /// <summary>Variance-prior mode; currently always <c>global</c>.</summary>
    public string VariancePrior { get; }

    /// <summary>Design columns actually used, excluding the intercept and group term.</summary>
    public IReadOnlyList<string> CovariatesUsed { get; }

    /// <summary>Notes about covariates skipped or levels dropped during design construction.</summary>
    public IReadOnlyList<string> Messages { get; }

    /// <summary>Diagnostic messages from the variance-moderation step.</summary>
    public IReadOnlyList<string> Warnings { get; }
}

/// <summary>
/// Two-group differential abundance via a limma empirical-Bayes moderated t-test, ported from the
/// PRISM Differential Explorer's <c>differential</c> (prism_diff_explorer.py). Fits the shared design
/// [intercept, groupB] per feature (<see cref="LinearModel"/>), moderates the residual variances
/// (<see cref="EmpiricalBayes"/>, global prior), and reports the moderated t, its two-sided p-value on
/// <c>df_residual + df_prior</c> degrees of freedom, and the Benjamini-Hochberg adjusted p-value. B is
/// the treatment arm, so a positive <c>logFC</c> is higher in B.
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
    {
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

        var (design, covariatesUsed, messages) = BuildDesign(nA, nB, cols, covariates);
        const int coefIdx = 1; // groupB is the second design column
        var nParams = design.GetLength(1);
        if (nSamples - nParams < 1)
            throw new ArgumentException(
                $"Not enough residual degrees of freedom (n={nSamples}, params={nParams}). " +
                "Use more samples or fewer covariates.");
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

        SqueezeVarResult squeezed;
        string variancePrior;
        if (trend && fit.Amean.All(double.IsFinite))
        {
            squeezed = EmpiricalBayes.SqueezeVarTrend(variances, fit.DfResidual, fit.Amean);
            variancePrior = "intensity-trend";
        }
        else
        {
            if (trend)
                messages.Add("Mean expression has non-finite values - intensity-trend prior "
                    + "unavailable, fell back to the global prior.");
            squeezed = EmpiricalBayes.SqueezeVarGlobal(variances, fit.DfResidual);
            variancePrior = "global";
        }

        var dfTotal = fit.DfResidual + squeezed.DfPrior;
        var stdevUnscaled = fit.StdevUnscaled[coefIdx];

        var pValues = new double[nTested];
        var rows = new DifferentialRow[nTested];
        var groupB = new double[nB];
        var groupA = new double[nA];
        for (var i = 0; i < nTested; i++)
        {
            var coef = fit.Coefficients[i, coefIdx];
            var t = coef / (stdevUnscaled * Math.Sqrt(squeezed.VarPost[i]));
            var p = ModeratedPValue(t, dfTotal);
            pValues[i] = p;

            for (var s = 0; s < nA; s++)
                groupA[s] = mk[i, s];
            for (var s = 0; s < nB; s++)
                groupB[s] = mk[i, nA + s];

            rows[i] = new DifferentialRow(
                featureIds[tested[i]], coef, Math.Pow(2.0, coef), fit.Amean[i], t, p,
                double.NaN, NumpyMath.Mean(groupA), NumpyMath.Mean(groupB));
        }

        var adj = Fdr.BenjaminiHochberg(pValues);
        for (var i = 0; i < nTested; i++)
            rows[i] = rows[i] with { AdjPValue = adj[i] };

        var ordered = rows.OrderBy(r => r.PValue).ToArray();

        return new DifferentialResult(ordered, nA, nB, nFeatures, nTested, fit.DfResidual,
            squeezed.DfPrior, variancePrior, covariatesUsed, messages, squeezed.Warnings);
    }

    /// <summary>
    /// Build the design matrix [intercept, groupB, covariate columns...]. Numeric covariates are
    /// mean-centered; categorical covariates are dummy-encoded with the first (sorted) level dropped.
    /// A covariate missing in any selected sample, or constant, is skipped; a dummy level collinear
    /// with the group is dropped. Every skip/drop is recorded in the returned messages.
    /// </summary>
    private static (double[,] Design, List<string> CovariatesUsed, List<string> Messages) BuildDesign(
        int nA, int nB, int[] cols, IReadOnlyList<Covariate>? covariates)
    {
        var nSamples = nA + nB;
        var grp = new double[nSamples];
        for (var s = nA; s < nSamples; s++)
            grp[s] = 1.0;

        var extra = new List<double[]>();
        var names = new List<string>();
        var messages = new List<string>();

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
                            messages.Add($"Covariate level '{cat.Name}_{level}' is confounded with group - dropped.");
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
    private static double ModeratedPValue(double t, double dfTotal)
    {
        var absT = Math.Abs(t);
        return double.IsInfinity(dfTotal)
            ? 2.0 * Normal.CDF(0.0, 1.0, -absT)
            : 2.0 * new StudentT(0.0, 1.0, dfTotal).CumulativeDistribution(-absT);
    }
}
