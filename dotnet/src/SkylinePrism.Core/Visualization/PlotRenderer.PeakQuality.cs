using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Core.Visualization;

/// <summary>
/// The peak-quality plots: Skyline's own peak metrics per run, in the order the runs were acquired.
/// </summary>
public static partial class PlotRenderer
{
    // Wider than the report's other plots: the x axis is every run in the cohort, and a degraded run
    // has to be readable as one point among them.
    private const int PeakQualityWidth = 1600;
    private const int PeakQualityScoreHeight = 780;
    private const int PeakQualityPanelHeight = 380;

    // Sample types in the order they are drawn: the cohort first, so the controls sit on top of it.
    private static readonly string[] PeakQualityTypeOrder = { "experimental", "qc", "reference" };

    /// <summary>PNG of <see cref="DrawPeakQualityScore"/>.</summary>
    public static byte[] PeakQualityScorePng(
        PeakQualityResult result, IReadOnlyDictionary<string, string> sampleTypes)
    {
        var plt = new Plot();
        DrawPeakQualityScore(plt, result, sampleTypes);
        return plt.GetImageBytes(PeakQualityWidth, PeakQualityScoreHeight, ImageFormat.Png);
    }

    /// <summary>
    /// Each run's median peak-quality score with the 10-90% range of its precursors' scores, in
    /// acquisition order and colored by sample type. The cohort's median and the labeling threshold
    /// are drawn across, and every run above the threshold is labeled.
    /// </summary>
    public static void DrawPeakQualityScore(
        Plot plt, PeakQualityResult result, IReadOnlyDictionary<string, string> sampleTypes)
    {
        var runs = result.Runs.Where(r => double.IsFinite(r.ScoreMedian)).ToList();
        if (runs.Count == 0)
        {
            DrawEmptyState(plt, "No run has a peak-quality score");
            return;
        }

        // Behind everything: what "typical" is, and the line a run has to cross to be labeled.
        var median = plt.Add.HorizontalLine(result.CohortMedian);
        median.Color = Color.FromHex("#7f7f7f");
        median.LineWidth = 3;
        median.LinePattern = LinePattern.Dashed;
        median.LegendText = $"cohort median {result.CohortMedian:0.00}";
        var threshold = plt.Add.HorizontalLine(result.FlagThreshold);
        threshold.Color = Color.FromHex("#333333");
        threshold.LineWidth = 2;
        threshold.LinePattern = LinePattern.Dotted;
        threshold.LegendText = $"{PeakQuality.FlagMultiple:0}x cohort median";

        var markerSize = RunMarkerSize(runs.Count);
        foreach (var (type, group) in ByType(runs, sampleTypes))
        {
            var color = Color.FromHex(TypeColors.GetValueOrDefault(type, TypeColors["unknown"]));
            var xs = group.Select(r => (double)r.RunOrder).ToArray();
            var ys = group.Select(r => r.ScoreMedian).ToArray();
            var zeros = new double[xs.Length];
            var range = new ScottPlot.Plottables.ErrorBar(
                xs, ys, zeros, zeros,
                group.Select(r => Math.Max(0, r.ScoreP90 - r.ScoreMedian)).ToArray(),
                group.Select(r => Math.Max(0, r.ScoreMedian - r.ScoreP10)).ToArray())
            {
                Color = color.WithAlpha(0.55),
                LineWidth = 2,
                CapSize = 0,
            };
            plt.Add.Plottable(range);
            var dots = plt.Add.Markers(xs, ys);
            dots.Color = color;
            dots.MarkerSize = markerSize;
            dots.LegendText = type;
        }

        var last = runs.Max(r => r.RunOrder);
        foreach (var run in result.Flagged)
        {
            var label = plt.Add.Text(RunLabel(run, result), run.RunOrder, run.ScoreMedian);
            // To the left of a run near the right edge, so the label is not cut off by the frame.
            var leftward = run.RunOrder > 0.75 * last;
            label.LabelAlignment = leftward ? Alignment.MiddleRight : Alignment.MiddleLeft;
            label.OffsetX = leftward ? -12 : 12;
            StyleTextLabel(label, 22, bold: true);
        }

        var flagged = result.Flagged.Count;
        plt.Title(flagged == 0
            ? $"No run stands out on peak quality ({runs.Count} runs)"
            : $"{flagged} of {runs.Count} runs stand out on peak quality");
        plt.XLabel(RunAxisLabel(result));
        plt.YLabel("Peak quality score (robust z)");
        StyleQcPlot(plt);
        plt.Axes.SetLimits(0, last + 1, 0, ScoreAxisTop(runs, result.FlagThreshold));
        // Outside the data: every corner of a run-order plot has runs in it.
        plt.ShowLegend(Edge.Right);
    }

    /// <summary>
    /// Top of the score axis, set by the medians and the typical run's 90th percentile rather than
    /// by the tallest range. A few runs whose worst precursors score 10 or more would otherwise squeeze
    /// every median - the thing being read - into the bottom quarter of the plot; their ranges run off
    /// the top instead, which still reads as "far above".
    /// </summary>
    internal static double ScoreAxisTop(IReadOnlyList<PeakQualityRun> runs, double flagThreshold)
    {
        var medianTop = runs.Max(r => r.ScoreMedian);
        var p90 = runs.Select(r => r.ScoreP90).Where(double.IsFinite).ToArray();
        var typicalTail = p90.Length > 0 ? Numerics.Stats.PercentileLinear(p90, 90) : medianTop;
        return Math.Max(Math.Max(medianTop * 1.15, typicalTail * 1.1), flagThreshold * 1.2);
    }

    /// <summary>
    /// The per-run median of each raw metric, one panel per metric the data had, stacked on one
    /// acquisition-order axis. Runs labeled on the score plot are marked in every panel, so which
    /// metric moved is read straight down.
    /// </summary>
    public static byte[] PeakQualityMetricsPng(
        PeakQualityResult result, IReadOnlyDictionary<string, string> sampleTypes)
    {
        var panels = PeakQualityPanels
            .Where(p => result.Runs.Any(r => double.IsFinite(p.Value(r))))
            .ToList();
        var multiplot = new Multiplot();
        while (multiplot.Subplots.Count < Math.Max(1, panels.Count))
            multiplot.AddPlot();
        multiplot.Layout = new ScottPlot.MultiplotLayouts.Rows();

        if (panels.Count == 0)
        {
            DrawEmptyState(multiplot.GetPlot(0), "No run has a peak-quality metric");
            return multiplot.Render(PeakQualityWidth, PeakQualityPanelHeight)
                .GetImageBytes(ImageFormat.Png);
        }

        var runs = result.Runs;
        var last = runs.Count == 0 ? 1 : runs.Max(r => r.RunOrder);
        var markerSize = RunMarkerSize(runs.Count);
        for (var i = 0; i < panels.Count; i++)
        {
            var (title, yLabel, value, reference) = panels[i];
            var plt = multiplot.GetPlot(i);

            if (reference is { } at)
            {
                var line = plt.Add.HorizontalLine(at);
                line.Color = Color.FromHex("#7f7f7f");
                line.LineWidth = 2;
                line.LinePattern = LinePattern.Dashed;
            }
            foreach (var run in result.Flagged)
            {
                var mark = plt.Add.VerticalLine(run.RunOrder);
                mark.Color = Color.FromHex("#333333").WithAlpha(0.6);
                mark.LineWidth = 2;
                mark.LinePattern = LinePattern.Dotted;
            }

            // A thin trace in run order under the points, so a drift reads as a drift.
            var ordered = runs.Where(r => double.IsFinite(value(r))).ToList();
            var trace = plt.Add.ScatterLine(
                ordered.Select(r => (double)r.RunOrder).ToArray(),
                ordered.Select(value).ToArray());
            trace.Color = Color.FromHex("#bdbdbd");
            trace.LineWidth = 1.5f;

            foreach (var (type, group) in ByType(ordered, sampleTypes))
            {
                var dots = plt.Add.Markers(
                    group.Select(r => (double)r.RunOrder).ToArray(),
                    group.Select(value).ToArray());
                dots.Color = Color.FromHex(TypeColors.GetValueOrDefault(type, TypeColors["unknown"]));
                dots.MarkerSize = markerSize;
                if (i == 0)
                    dots.LegendText = type;
            }

            plt.Title(title);
            plt.YLabel(yLabel);
            if (i == panels.Count - 1)
                plt.XLabel(RunAxisLabel(result));
            StyleQcPlot(plt, fontScale: 0.75);
            // Every panel's data area starts at the same x, whatever its tick labels' widths, so a
            // run sits at the same horizontal position in all of them.
            plt.Axes.Left.MinimumSize = 170;
            plt.Axes.SetLimitsX(0, last + 1);
            if (i == 0)
                plt.ShowLegend(ChooseLegendCorner(
                    ordered.Select(r => (double)r.RunOrder).ToList(), ordered.Select(value).ToList()));
        }

        return multiplot.Render(PeakQualityWidth, PeakQualityPanelHeight * panels.Count + 60)
            .GetImageBytes(ImageFormat.Png);
    }

    /// <summary>The raw-metric panels: title, y label, the per-run value, and a reference level.</summary>
    private static readonly (string Title, string YLabel, Func<PeakQualityRun, double> Value, double? Reference)[]
        PeakQualityPanels =
        {
            ("Shape correlation (median over precursors)", "Correlation", r => r.ShapeCorrelation, null),
            ("Isotope dot product (median over precursors)", "Dot product", r => r.IsotopeDotProduct, null),
            ("Retention time minus each precursor's median", "Minutes", r => r.RtDeviation, 0),
            ("FWHM over each precursor's median", "Ratio", r => r.FwhmRatio, 1),
        };

    /// <summary>Runs grouped by sample type, the known types first in a fixed order.</summary>
    private static IEnumerable<(string Type, List<PeakQualityRun> Runs)> ByType(
        IEnumerable<PeakQualityRun> runs, IReadOnlyDictionary<string, string> sampleTypes)
    {
        var groups = runs
            .GroupBy(r => sampleTypes.GetValueOrDefault(r.SampleId, "unknown"))
            .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.Ordinal);
        foreach (var type in PeakQualityTypeOrder.Concat(
                     groups.Keys.Except(PeakQualityTypeOrder).OrderBy(t => t, StringComparer.Ordinal)))
            if (groups.TryGetValue(type, out var list))
                yield return (type, list);
    }

    /// <summary>Smaller points as the cohort grows, so neighboring runs stay distinguishable.</summary>
    private static float RunMarkerSize(int runs) => (float)Math.Clamp(1200.0 / Math.Max(1, runs), 5, 14);

    private static string RunAxisLabel(PeakQualityResult result) => result.OrderedByAcquiredTime
        ? "Run (acquisition order)"
        : "Run (sample order - not every run has an acquired time)";

    /// <summary>"run 11: WBC-IC-TS-001-011" - the replicate, without its document suffix.</summary>
    internal static string RunLabel(PeakQualityRun run, PeakQualityResult result)
    {
        var sep = run.SampleId.IndexOf("__@__", StringComparison.Ordinal);
        var replicate = sep >= 0 ? run.SampleId[..sep] : run.SampleId;
        return result.OrderedByAcquiredTime ? $"run {run.RunOrder}: {replicate}" : replicate;
    }
}
