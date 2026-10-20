using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace SkylinePrism.Core.Library;

/// <summary>
/// One peptide's precision in one pool material: the CV its library-assisted values actually reached
/// across the pool's injections, against the CV a sum of fragments reaches at the signal the fit kept.
/// </summary>
/// <param name="SignalMean">Mean total area of the peptide's fragments across the pool's injections.</param>
/// <param name="RetainedMean">Mean area of the fragments the fit estimated its scale from.</param>
/// <param name="KeptMean">Mean number of fragments the fit kept, summed over charge states.</param>
/// <param name="ObservedCv">CV (%, LINEAR) of the peptide's library-assisted abundance across the pool.</param>
/// <param name="ExpectedCv">The expected-CV curve at <paramref name="RetainedMean"/>.</param>
public sealed record PrecisionRow(
    string Group, int Injections, double SignalMean, double RetainedMean, double KeptMean,
    double ObservedCv, double ExpectedCv)
{
    /// <summary>
    /// Observed / expected. Near 1: as precise as the signal the fit kept allows - a high CV near the
    /// limit of detection is then the honest price of removing interference. Well above 1: the fit
    /// added variation of its own.
    /// </summary>
    public double Ratio => double.IsFinite(ObservedCv) && double.IsFinite(ExpectedCv) && ExpectedCv > 0
        ? ObservedCv / ExpectedCv
        : double.NaN;
}

/// <summary>
/// The per-peptide precision check for library assist, written as
/// <c>library_assist_precision.csv</c> and summarized in the QC report.
/// <para>
/// Why "expected at the retained signal": a library-assisted value is the fitted scale times the
/// library's total, so a fragment left out of the scale still contributes its library-predicted area -
/// what it does NOT contribute is its measurement. The scale's precision therefore rests on the
/// fragments that were kept, and that is the signal the comparison is made at.
/// </para>
/// </summary>
public static class LibraryAssistPrecision
{
    public const string FileName = "library_assist_precision.csv";

    /// <summary>Fewest usable injections an observed CV is computed from.</summary>
    public const int MinInjections = 3;

    /// <summary>
    /// The row for one group. <paramref name="abundance"/> (LINEAR), <paramref name="retained"/>,
    /// <paramref name="kept"/> and <paramref name="signal"/> are per sample of the run; only the
    /// <paramref name="indices"/> are read. The observed CV uses the finite, positive abundances (NaN
    /// when fewer than <see cref="MinInjections"/>); the means are over every index.
    /// </summary>
    public static PrecisionRow Row(
        string group, IReadOnlyList<int> indices, double[] abundance, double[] retained, double[] kept,
        double[] signal, FragmentNoiseCurve curve)
    {
        var values = new List<double>(indices.Count);
        foreach (var j in indices)
            if (double.IsFinite(abundance[j]) && abundance[j] > 0)
                values.Add(abundance[j]);
        var observed = double.NaN;
        if (values.Count >= MinInjections)
        {
            double sum = 0;
            foreach (var v in values)
                sum += v;
            var mean = sum / values.Count;
            double ss = 0;
            foreach (var v in values)
                ss += (v - mean) * (v - mean);
            observed = 100.0 * Math.Sqrt(ss / (values.Count - 1)) / mean;
        }

        var n = indices.Count;
        double r = 0, k = 0, s = 0;
        foreach (var j in indices)
            r += retained[j];
        foreach (var j in indices)
            k += kept[j];
        foreach (var j in indices)
            s += signal[j];
        var retainedMean = r / n;
        return new PrecisionRow(
            group, n, s / n, retainedMean, k / n, observed, curve.ExpectedCvAt(retainedMean));
    }

    private static readonly string[] Header =
    {
        "peptide", "group", "injections", "matched_fragments", "kept_fragments_mean", "signal_mean",
        "retained_signal_mean", "observed_cv", "expected_cv", "observed_over_expected",
    };

    /// <summary>Write the rows, one per peptide and pool material, in peptide order.</summary>
    public static void Write(string path, IEnumerable<(string Peptide, int Matched, PrecisionRow Row)> rows)
    {
        var sb = new StringBuilder();
        sb.Append(string.Join(",", Header)).Append('\n');
        foreach (var (peptide, matched, row) in rows)
        {
            sb.Append(Escape(peptide)).Append(',')
              .Append(Escape(row.Group)).Append(',')
              .Append(row.Injections.ToString(CultureInfo.InvariantCulture)).Append(',')
              .Append(matched.ToString(CultureInfo.InvariantCulture)).Append(',')
              .Append(F(row.KeptMean)).Append(',')
              .Append(F(row.SignalMean)).Append(',')
              .Append(F(row.RetainedMean)).Append(',')
              .Append(F(row.ObservedCv)).Append(',')
              .Append(F(row.ExpectedCv)).Append(',')
              .Append(F(row.Ratio)).Append('\n');
        }
        File.WriteAllText(path, sb.ToString());
    }

    /// <summary>The rows of a written file, or an empty list when it is absent or unreadable.</summary>
    public static List<(string Peptide, int Matched, PrecisionRow Row)> Read(string path)
    {
        var rows = new List<(string, int, PrecisionRow)>();
        try
        {
            if (!File.Exists(path))
                return rows;
            using var reader = new StreamReader(IO.ParquetColumnIo.OpenRead(path));
            var header = reader.ReadLine();
            if (header is null || !header.StartsWith("peptide,group,", StringComparison.Ordinal))
                return rows;
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                var f = Split(line);
                if (f.Count < Header.Length)
                    continue;
                var row = new PrecisionRow(
                    f[1], int.Parse(f[2], CultureInfo.InvariantCulture), P(f[5]), P(f[6]), P(f[4]), P(f[7]), P(f[8]));
                rows.Add((f[0], int.Parse(f[3], CultureInfo.InvariantCulture), row));
            }
        }
        catch (Exception ex) when (ex is IOException or FormatException or OverflowException
                                       or UnauthorizedAccessException)
        {
            rows.Clear();
        }
        return rows;
    }

    private static string F(double v) => double.IsNaN(v) ? "NaN" : v.ToString("R", CultureInfo.InvariantCulture);

    private static double P(string s) => s == "NaN" ? double.NaN : double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);

    private static string Escape(string field) =>
        field.Contains(',') || field.Contains('"') || field.Contains('\n')
            ? "\"" + field.Replace("\"", "\"\"") + "\""
            : field;

    private static List<string> Split(string line)
    {
        var fields = new List<string>();
        var sb = new StringBuilder();
        var quoted = false;
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (quoted)
            {
                if (c == '"' && i + 1 < line.Length && line[i + 1] == '"')
                {
                    sb.Append('"');
                    i++;
                }
                else if (c == '"')
                {
                    quoted = false;
                }
                else
                {
                    sb.Append(c);
                }
            }
            else if (c == '"')
            {
                quoted = true;
            }
            else if (c == ',')
            {
                fields.Add(sb.ToString());
                sb.Clear();
            }
            else
            {
                sb.Append(c);
            }
        }
        fields.Add(sb.ToString());
        return fields;
    }
}
