using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Library;
using SkylinePrism.Core.Numerics;
using static System.FormattableString;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// The library-assist precision section of the QC report: whether each peptide's library-assisted
/// values are as precise as the fragment signal the fit kept allows. See
/// <see cref="LibraryAssistPrecision"/> for the comparison and <see cref="FragmentNoiseCurve"/> for
/// both curves.
/// </summary>
public static partial class QcReport
{
    private sealed record LibraryAssistSection(
        FragmentNoiseCurve? Noise, List<(string Peptide, int Matched, PrecisionRow Row)> Rows);

    /// <summary>
    /// A file read, or null when this run did not use the noise-gated library fit. The run's own config
    /// decides, not the files' presence: they could be left from an earlier run with other settings.
    /// </summary>
    private static LibraryAssistSection? RenderLibraryAssistSection(string outputDir, PrismConfig runConfig)
    {
        if (!FragmentNoise.AppliesTo(runConfig))
            return null;
        var noise = FragmentNoiseCurve.Load(Path.Combine(outputDir, FragmentNoiseCurve.FileName));
        var rows = LibraryAssistPrecision.Read(Path.Combine(outputDir, LibraryAssistPrecision.FileName));
        return noise is null && rows.Count == 0 ? null : new LibraryAssistSection(noise, rows);
    }

    private static void AppendLibraryAssist(StringBuilder sb, LibraryAssistSection section)
    {
        sb.Append("<div class=\"section-header\">Library-Assist Precision</div>");
        sb.Append("<p class=\"note\">Library assist leaves a fragment out of a peptide's scale when it reads "
            + "well above its library prediction - interference - and fills it in from the library at that "
            + "scale. Removing signal costs precision, most near the limit of detection; this checks that "
            + "the cost is no more than the signal kept allows. <em>Expected CV</em> is what a plain sum of "
            + "fragments achieves in these pool injections at the same total area as the fragments the fit "
            + "kept; <em>observed / expected</em> near 1 means a peptide is as precise as its kept signal "
            + "allows, and well above 1 means the fit added variation of its own. Per-peptide values are in "
            + HtmlEncode(LibraryAssistPrecision.FileName) + ".</p>");

        if (section.Noise is { } noise)
        {
            var source = noise.Source switch
            {
                FragmentNoiseCurve.PoolReplicates => "within "
                    + string.Join(" + ", noise.Groups.Select(g => Invariant($"{g.Injections} {g.Name}")))
                    + " pool injections",
                FragmentNoiseCurve.AllSamples => "across all samples, because no reference or QC pool has "
                    + "3 injections (biological variation then counts as noise, so fewer fragments are removed)",
                _ => "nowhere: too few samples, so the noise test was off",
            };
            sb.Append("<p class=\"note\">Fragment noise, which decides whether an excess is interference or "
                + $"noise, was measured {HtmlEncode(source)}");
            if (noise.HasNoise)
                sb.Append(Invariant($": a fragment's SD is {noise.SdAt(1_000):N0} at an expected area of 1,000 and ")
                    + Invariant($"{noise.SdAt(10_000):N0} at 10,000"));
            sb.Append(".</p>");
        }

        if (section.Rows.Count == 0)
        {
            sb.Append("<p class=\"note\">No precision check: it needs a reference or QC pool with at least "
                + "3 injections.</p>");
            return;
        }

        sb.Append(PrecisionByIntensityTable(section.Rows));
        sb.Append(PrecisionByKeptTable(section.Rows));
    }

    private static string F(double v, string format) =>
        double.IsFinite(v) ? v.ToString(format, CultureInfo.InvariantCulture) : "-";

    private static double MedianOf(IEnumerable<double> values)
    {
        var finite = values.Where(double.IsFinite).ToArray();
        return finite.Length > 0 ? Stats.Median(finite) : double.NaN;
    }

    /// <summary>One row per pool and fifth of peptides by total fragment signal in that pool.</summary>
    private static string PrecisionByIntensityTable(List<(string Peptide, int Matched, PrecisionRow Row)> rows)
    {
        var sb = new StringBuilder();
        sb.Append("<table><tr><th>Pool</th><th>Peptides, by fragment signal</th><th>Peptides</th>"
            + "<th>Signal kept (median %)</th><th>Fragments kept (median)</th><th>Observed CV (median %)</th>"
            + "<th>Expected CV (median %)</th><th>Observed / expected (median)</th></tr>");
        foreach (var group in rows.GroupBy(r => r.Row.Group))
        {
            var ordered = group.Where(r => double.IsFinite(r.Row.SignalMean))
                .OrderBy(r => r.Row.SignalMean).Select(r => r.Row).ToList();
            var labels = new[] { "dimmest fifth", "2nd", "3rd", "4th", "brightest fifth" };
            for (var q = 0; q < 5; q++)
            {
                var lo = ordered.Count * q / 5;
                var hi = ordered.Count * (q + 1) / 5;
                var slice = ordered.GetRange(lo, hi - lo);
                if (slice.Count == 0)
                    continue;
                sb.Append($"<tr><td>{HtmlEncode(group.Key)}</td><td>{labels[q]}</td>"
                    + Invariant($"<td>{slice.Count:N0}</td>")
                    + $"<td>{F(MedianOf(slice.Select(r => r.SignalMean > 0 ? 100.0 * r.RetainedMean / r.SignalMean : double.NaN)), "0")}</td>"
                    + $"<td>{F(MedianOf(slice.Select(r => r.KeptMean)), "0.0")}</td>"
                    + $"<td>{F(MedianOf(slice.Select(r => r.ObservedCv)), "0.0")}</td>"
                    + $"<td>{F(MedianOf(slice.Select(r => r.ExpectedCv)), "0.0")}</td>"
                    + $"<td>{F(MedianOf(slice.Select(r => r.Ratio)), "0.00")}</td></tr>");
            }
            var all = group.Select(r => r.Row).ToList();
            sb.Append($"<tr><td>{HtmlEncode(group.Key)}</td><td><em>all</em></td>"
                + Invariant($"<td>{all.Count:N0}</td>")
                + $"<td>{F(MedianOf(all.Select(r => r.SignalMean > 0 ? 100.0 * r.RetainedMean / r.SignalMean : double.NaN)), "0")}</td>"
                + $"<td>{F(MedianOf(all.Select(r => r.KeptMean)), "0.0")}</td>"
                + $"<td>{F(MedianOf(all.Select(r => r.ObservedCv)), "0.0")}</td>"
                + $"<td>{F(MedianOf(all.Select(r => r.ExpectedCv)), "0.0")}</td>"
                + $"<td>{F(MedianOf(all.Select(r => r.Ratio)), "0.00")}</td></tr>");
        }
        sb.Append("</table>");
        return sb.ToString();
    }

    /// <summary>Observed / expected by how many fragments the fit kept - the low-coverage values.</summary>
    private static string PrecisionByKeptTable(List<(string Peptide, int Matched, PrecisionRow Row)> rows)
    {
        static string Bin(double kept) => kept < 3.5 ? "3 or fewer" : kept < 4.5 ? "4" : kept < 5.5 ? "5" : "6 or more";
        var bins = new[] { "3 or fewer", "4", "5", "6 or more" };
        var sb = new StringBuilder();
        sb.Append("<table><tr><th>Pool</th><th>Fragments kept (mean over injections)</th><th>Peptides</th>"
            + "<th>Observed / expected (median)</th><th>Peptides above 1.5</th></tr>");
        foreach (var group in rows.GroupBy(r => r.Row.Group))
        {
            foreach (var bin in bins)
            {
                var slice = group.Where(r => Bin(r.Row.KeptMean) == bin).Select(r => r.Row).ToList();
                if (slice.Count == 0)
                    continue;
                var ratios = slice.Select(r => r.Ratio).Where(double.IsFinite).ToList();
                var above = ratios.Count > 0 ? 100.0 * ratios.Count(x => x > 1.5) / ratios.Count : double.NaN;
                sb.Append($"<tr><td>{HtmlEncode(group.Key)}</td><td>{bin}</td>"
                    + Invariant($"<td>{slice.Count:N0}</td>")
                    + $"<td>{F(MedianOf(ratios), "0.00")}</td>"
                    + $"<td>{F(above, "0")}%</td></tr>");
            }
        }
        sb.Append("</table>");
        return sb.ToString();
    }
}
