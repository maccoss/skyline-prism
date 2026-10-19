using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Visualization;
using static System.FormattableString;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// The peak-quality section of the QC report: Skyline's own peak metrics per run, in acquisition
/// order. See <see cref="PeakQuality"/> for the score.
/// </summary>
public static partial class QcReport
{
    /// <summary>What the section shows: the measurement and its plots, or why there is none.</summary>
    private sealed record PeakQualitySection(
        PeakQualityResult? Result, List<PlotSection> Plots, string? Failure);

    /// <summary>
    /// Measure and plot, or null when this directory cannot have the section (no merged data, or
    /// none of the metrics) - which <see cref="PeakQuality.Measure"/> has already logged.
    /// </summary>
    /// <remarks>
    /// Unlike the ion-accounting section this is not a file read: it is computed from
    /// <c>merged_data/</c> every time - about 20 s for 30 million transition rows, most of it
    /// DuckDB's grouped median (see <see cref="IO.MergedParquetReader.StreamPrecursorQuality"/>) -
    /// so there is no cache to go stale. A directory whose merged data has been deleted simply has
    /// no section.
    /// </remarks>
    private static PeakQualitySection? RenderPeakQualitySection(
        string outputDir, PrismConfig runConfig, IReadOnlyDictionary<string, string> sampleTypes,
        bool savePlots, string plotsDir, Action<string>? log)
    {
        PeakQualityResult? result;
        try
        {
            result = PeakQuality.Measure(
                outputDir, runConfig.Data.ToOverrides(), log, runConfig.Processing.MergeMemoryMb);
        }
        catch (Exception ex)
        {
            // Not swallowed: the section says it failed and why, in the report and in the log. What
            // it does not do is take the rest of the report down with it.
            var reason = $"{ex.GetType().Name}: {ex.Message}";
            log?.Invoke($"  WARNING: peak quality could not be measured ({reason}).");
            return new PeakQualitySection(null, new List<PlotSection>(), reason);
        }
        if (result is null || result.Runs.Count == 0)
            return null;

        var flagged = result.Flagged;
        log?.Invoke(flagged.Count == 0
            ? "  Peak quality: no run's median score exceeds twice the cohort's."
            : $"  Peak quality: {flagged.Count} run(s) exceed twice the cohort's median score: "
              + string.Join(", ", flagged.Select(r => PlotRenderer.RunLabel(r, result))) + ".");

        var score = new List<PlotImage>();
        Render(score, "", "Peak quality score per run, in acquisition order",
            "peak_quality_by_run.png", savePlots, plotsDir,
            () => PlotRenderer.PeakQualityScorePng(result, sampleTypes));
        var metrics = new List<PlotImage>();
        Render(metrics, "", "Per-run median of each Skyline peak-quality metric, in acquisition order",
            "peak_quality_metrics_by_run.png", savePlots, plotsDir,
            () => PlotRenderer.PeakQualityMetricsPng(result, sampleTypes));

        return new PeakQualitySection(result, new List<PlotSection>
        {
            new("Peak Quality Score by Run", score),
            new("Peak Quality Metrics by Run", metrics),
        }, null);
    }

    private static void AppendPeakQuality(
        StringBuilder sb, PeakQualitySection section, IReadOnlyDictionary<string, string> sampleTypes)
    {
        sb.Append("<div class=\"section-header\">Peak Quality by Acquisition Order</div>");
        if (section.Result is not { } result)
        {
            sb.Append("<div class=\"warnings\"><strong>WARNING</strong>: peak quality could not be "
                + $"measured: {HtmlEncode(section.Failure ?? "unknown error")}</div>");
            return;
        }

        // What the score is, once, since neither plot can say it. The precursor count and the
        // metrics the data had are facts about THIS run, so they go here too.
        sb.Append("<p class=\"note\">Skyline's own peak-quality columns for every precursor in "
            + "every run - " + HtmlEncode(string.Join(", ", result.Metrics)) + " - each compared "
            + "with the same precursor's median across all runs as a robust z (the deviation over "
            + "MAD/0.6745). A low shape correlation or isotope dot product counts; retention time and "
            + "FWHM count in either direction. A precursor's score in a run is the root mean square "
            + "of its z's, and each run is shown by the median and 10-90% range of its precursors' "
            + Invariant($"scores ({result.Precursors:N0} precursors). Runs whose median is more than ")
            + Invariant($"{PeakQuality.FlagMultiple:0} times the cohort's are labeled, and marked in every panel ")
            + "of the second plot so the metric that moved can be read straight down. The score axis "
            + "is set by the medians, so a run whose worst precursors score far higher has its range "
            + "run off the top.</p>");
        if (!result.OrderedByAcquiredTime)
        {
            sb.Append("<p class=\"note\">The runs are in sample order, not acquisition order: "
                + Invariant($"{result.TimedRuns:N0} of {result.Runs.Count:N0} have an acquired time. Export the ")
                + "Acquired Time column in the transition report to order them.</p>");
        }

        if (result.Flagged.Count > 0)
            sb.Append(FlaggedRunsTable(result, sampleTypes));
        AppendSections(sb, section.Plots);
    }

    /// <summary>
    /// The labeled runs beside the cohort's typical values, so "0.63 against 0.81" is a number to
    /// read rather than a distance to estimate off a plot.
    /// </summary>
    private static string FlaggedRunsTable(PeakQualityResult result, IReadOnlyDictionary<string, string> sampleTypes)
    {
        static string F(double v, string format) =>
            double.IsFinite(v) ? v.ToString(format, CultureInfo.InvariantCulture) : "-";
        static double Typical(IEnumerable<double> values)
        {
            var finite = values.Where(double.IsFinite).ToArray();
            return finite.Length > 0 ? Stats.Median(finite) : double.NaN;
        }

        var sb = new StringBuilder();
        sb.Append("<table><tr><th>Run</th><th>Sample type</th><th>Median score</th>"
            + "<th>Shape correlation</th><th>Isotope dot product</th><th>RT vs median (min)</th>"
            + "<th>FWHM vs median</th></tr>");
        foreach (var run in result.Flagged)
        {
            sb.Append($"<tr><td>{HtmlEncode(PlotRenderer.RunLabel(run, result))}</td>"
                + $"<td>{HtmlEncode(sampleTypes.GetValueOrDefault(run.SampleId, "unknown"))}</td>"
                + $"<td>{F(run.ScoreMedian, "0.00")}</td><td>{F(run.ShapeCorrelation, "0.000")}</td>"
                + $"<td>{F(run.IsotopeDotProduct, "0.000")}</td><td>{F(run.RtDeviation, "+0.000;-0.000;0.000")}</td>"
                + $"<td>{F(run.FwhmRatio, "0.000")}</td></tr>");
        }
        var runs = result.Runs;
        sb.Append("<tr><td><em>Median of all runs</em></td><td></td>"
            + $"<td>{F(result.CohortMedian, "0.00")}</td>"
            + $"<td>{F(Typical(runs.Select(r => r.ShapeCorrelation)), "0.000")}</td>"
            + $"<td>{F(Typical(runs.Select(r => r.IsotopeDotProduct)), "0.000")}</td>"
            + $"<td>{F(Typical(runs.Select(r => r.RtDeviation)), "+0.000;-0.000;0.000")}</td>"
            + $"<td>{F(Typical(runs.Select(r => r.FwhmRatio)), "0.000")}</td></tr>");
        sb.Append("</table>");
        return sb.ToString();
    }
}
