using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// The one writer of a differential results table. <c>prism differential</c>'s <c>differential.csv</c>
/// and the quant report's <c>quant/differential.csv</c> are the same file shape because both come from
/// here - there used to be two writers, and the report's copy dropped the provenance header and the
/// gene/protein/accession columns while the docs described the two as the same file.
/// </summary>
public static class DifferentialCsv
{
    /// <summary>
    /// Write the result table, in the row order the test produced (most significant first).
    /// </summary>
    /// <remarks>
    /// The header carries the contrast and the method as <c>#</c> comment lines, because a results
    /// file outlives the shell it was produced in and "which way round is the fold change" is the
    /// first question anyone asks of one. Invariant culture throughout, as every other PRISM writer:
    /// a decimal comma would make the file unreadable as CSV.
    /// </remarks>
    /// <param name="groupBy">The contrast column, or the trend column under a trend design.</param>
    /// <param name="aLabel">Arm A's label, or the trend column's low end (<see cref="TrendEndpoints"/>).</param>
    /// <param name="bLabel">Arm B's label, or the trend column's high end.</param>
    /// <param name="restrictions">The samples a trend was fitted on, when not all of them; recorded in
    /// the header, since the rows below mean something different over a subset.</param>
    public static void Write(
        string path, DifferentialResult result, DifferentialDataset dataset,
        DifferentialOptions options, SignificanceRule rule, string groupBy, string aLabel, string bLabel,
        string effectName = "log2FC", IReadOnlyList<QuantRestriction>? restrictions = null)
    {
        var dirName = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dirName))
            Directory.CreateDirectory(dirName);

        using var w = new StreamWriter(path);
        w.WriteLine(result.IsTrend
            ? $"# trend: {groupBy} from {aLabel} to {bLabel} (span {result.TrendRange.ToString("0.####", CultureInfo.InvariantCulture)}); "
              + "log2fc is the modeled change ACROSS that span, slope = log2fc / span"
            : $"# contrast: {groupBy} = {bLabel} vs {aLabel} (positive log2FC is higher in {bLabel})");
        if (result.IsTrend && restrictions is { Count: > 0 } && QuantRestriction.Describe(restrictions) is { Length: > 0 } kept)
            w.WriteLine($"# restricted to: {kept} (only those samples were fitted)");
        w.WriteLine($"# method: {options.Describe(result.VariancePrior)}, {CorrectionName(options.Correction)}");
        if (result.DescribePriorFit() is { } priorFit)
            w.WriteLine($"# {priorFit}");
        if (result.DescribeBlocking() is { } blocking)
            w.WriteLine($"# blocked by subject: {blocking}");
        w.WriteLine(result.IsTrend
            ? $"# n: {result.NA} samples"
              + (result.NSubjects > 0 ? $" in {result.NSubjects} subjects" : string.Empty)
              + $"; tested {result.NFeaturesTested} of {result.NFeaturesTotal}"
            : result.IsBlocked
                ? $"# n: {result.NA} ({result.SubjectsA} subjects) vs {result.NB} ({result.SubjectsB} subjects); "
                  + $"tested {result.NFeaturesTested} of {result.NFeaturesTotal}"
                : $"# n: {result.NA} vs {result.NB}; tested {result.NFeaturesTested} of {result.NFeaturesTotal}");
        // EVERY tested feature is in this file, not just the hits - so the rule is recorded as the
        // one the run reported against, not as a filter that was applied to the rows below.
        w.WriteLine($"# hit rule (rows are NOT filtered by it): {rule.Describe(effectName)}");
        w.WriteLine("feature_id,label,gene,protein,accession,log2fc,fc,ave_expr,statistic,"
            + "p_value,adj_p_value,mean_a,mean_b");
        foreach (var r in result.Rows)
        {
            // A feature the matrix carried no annotation for keeps its id and leaves the rest
            // empty, rather than repeating the id into columns that mean something else.
            var identity = dataset.IdentityOf(r.FeatureId);
            w.WriteLine(string.Join(',', new[]
            {
                Csv(r.FeatureId),
                Csv(identity?.Label ?? r.FeatureId),
                Csv(Join(identity?.Genes)),
                Csv(Join(identity?.ProteinNames)),
                Csv(Join(identity?.Accessions)),
                Num(r.LogFc), Num(r.Fc), Num(r.AveExpr), Num(r.T),
                Num(r.PValue), Num(r.AdjPValue), Num(r.MeanA), Num(r.MeanB),
            }));
        }
    }

    /// <summary>The correction's full name, as the CSV header and the CLI's summary line print it.</summary>
    public static string CorrectionName(MultipleTesting correction) => correction switch
    {
        MultipleTesting.BenjaminiHochberg => "Benjamini-Hochberg",
        MultipleTesting.BenjaminiYekutieli => "Benjamini-Yekutieli",
        MultipleTesting.Bonferroni => "Bonferroni",
        MultipleTesting.Holm => "Holm",
        _ => "uncorrected",
    };

    /// <summary>
    /// A trend column's observed range as the two labels <see cref="Write"/> takes for a trend, over
    /// the finite values only; <c>(none)</c> for an end with no usable value.
    /// </summary>
    /// <remarks>
    /// Kept as numbers until formatted here. Formatting them and then splitting the formatted string
    /// back apart lost precision, and turned the no-usable-values case into two copies of the literal
    /// "(none)".
    /// </remarks>
    public static (string Low, string High) TrendEndpoints(IReadOnlyList<double> x)
    {
        var used = x.Where(double.IsFinite).ToList();
        var xMin = used.Count > 0 ? used.Min() : double.NaN;
        var xMax = used.Count > 0 ? used.Max() : double.NaN;
        static string Endpoint(double v) =>
            double.IsNaN(v) ? "(none)" : v.ToString("0.###", CultureInfo.InvariantCulture);
        return (Endpoint(xMin), Endpoint(xMax));
    }

    /// <summary>A shared feature's several groups, semicolon-joined so the cell stays one field.</summary>
    private static string Join(IReadOnlyList<string>? values) =>
        values is null ? string.Empty : string.Join(';', values.Where(v => !string.IsNullOrEmpty(v)));

    private static string Num(double v) =>
        double.IsNaN(v) ? string.Empty : v.ToString("G17", CultureInfo.InvariantCulture);

    private static string Csv(string v) =>
        v.Contains(',') || v.Contains('"') || v.Contains('\n')
            ? '"' + v.Replace("\"", "\"\"") + '"'
            : v;
}
