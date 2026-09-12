using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Core.Visualization;

/// <summary>
/// The ion-accounting plots: how many ions reached the detector, and what share of them a peptide
/// sequence explains - per replicate, and across the gradient.
/// </summary>
public static partial class PlotRenderer
{
    /// <summary>Which MS level a plot shows. MS1 and MS2 are never mixed on one axis.</summary>
    /// <remarks>
    /// Measured over a whole Astral run, the two levels differ in both totals and fraction: MS2
    /// acquired 1.228e12 ions against MS1's 3.745e11 - 167 MS2 scans per survey scan - while the
    /// assigned FRACTION goes the other way, 40.5% at MS1 against 3.4% at MS2. MS1 is dominated by
    /// the precursors that were identified; each 3 Th MS2 window fragments everything co-isolated
    /// in it. One axis would flatten one of the two, and the fractions are what the plot is for.
    /// </remarks>
    public enum IonLevel
    {
        Ms1 = 1,
        Ms2 = 2,
    }

    /// <summary>
    /// Per replicate: the ions acquired, and the part of them assigned to a peptide sequence.
    /// </summary>
    /// <remarks>
    /// <para>Same shape as <see cref="DrawMs2Accounting"/> - a full-width neutral background bar for
    /// acquired with the colored assigned bar drawn on top of it - because the two totals NEST. At
    /// 192 replicates a bar is about 4 px, so nested widths differ by under a pixel while nested
    /// heights read at any density.</para>
    /// <para><b>A fraction over 1 is never drawn.</b> It is impossible, so it means a defect - a
    /// units mismatch, a scheme that does not match the acquisition, or claims merged too loosely -
    /// and clamping it to 100% would turn a visible bug into a plausible reading. The earlier version
    /// of this feature reported a fraction built from mismatched units for exactly that kind of
    /// reason, and it looked entirely plausible.</para>
    /// </remarks>
    public static void DrawIonAccounting(
        Plot plt, IonAccountingResult result, IonLevel level, string? title = null,
        double fontScale = 1.0)
    {
        // Start from an empty plot. ScottPlot's Add methods APPEND - they do not replace, and
        // neither the plottables nor the legend entries they carry go away on their own - so a
        // caller that draws twice on one Plot stacks the second render on the first. That is what
        // the GUI does every time the view, level or replicate changes, and the visible symptom was
        // a legend that grew another copy of every series each time, with the earlier renders' data
        // still underneath. Clearing HERE rather than at the call site because three of the four
        // call sites remembered and one did not; DrawEmptyState has always done it. Safe because
        // every path below sets its own axis limits, so nothing stale survives.
        plt.Clear();

        var rows = result.Rows;
        if (rows.Count == 0)
        {
            DrawEmptyState(plt, title ?? "No ion accounting to show", fontScale);
            return;
        }

        var acquiredOf = Selector(level, acquired: true);
        var assignedOf = Selector(level, acquired: false);

        // Drawn only where it exists and can differ: MS2, and a cache that actually measured it. An
        // export with no precursor charge column measures none, and a bar of height zero would read
        // as "these peptides account for nothing" rather than "this was never asked".
        var showExplained = level == IonLevel.Ms2 && rows.Any(r => r.HasExplained);
        var tallest = rows.Max(r => Math.Max(Finite(acquiredOf(r)), Finite(assignedOf(r))));
        var (scale, unit) = SignalScale(tallest);

        // Acquired first, so the assigned bars land on top of it.
        var acquiredBars = new List<Bar>(rows.Count);
        for (var i = 0; i < rows.Count; i++)
        {
            acquiredBars.Add(new Bar
            {
                Position = i,
                // Finite() gives 0 for a replicate with no data file, so its background bar simply
                // does not appear - the honest rendering of an unknown denominator.
                Value = Finite(acquiredOf(rows[i])) / scale,
                FillColor = AcquiredBarColor,
                LineWidth = 0,
                Size = 0.85,
            });
        }
        plt.Add.Bars(acquiredBars);

        var withDenominator = rows.Count(r => Finite(acquiredOf(r)) > 0);
        var acquiredKey = plt.Add.Marker(double.NaN, double.NaN);
        acquiredKey.MarkerStyle.Shape = MarkerShape.FilledSquare;
        acquiredKey.MarkerStyle.Size = 14;
        acquiredKey.MarkerStyle.FillColor = AcquiredBarColor;
        acquiredKey.MarkerStyle.LineWidth = 0;
        acquiredKey.LegendText = withDenominator == rows.Count
            ? $"acquired {level.ToString().ToUpperInvariant()} ions"
            : $"acquired {level.ToString().ToUpperInvariant()} ions "
              + $"({withDenominator:N0} of {rows.Count:N0})";

        // Between the two, and BEFORE the assigned bars so the shorter one lands on top. The three
        // totals nest - acquired >= explained >= assigned - so they are drawn back to front rather
        // than stacked: stacking would partition, and these do not partition.
        if (showExplained)
        {
            var explainedBars = new List<Bar>(rows.Count);
            for (var i = 0; i < rows.Count; i++)
            {
                // A replicate with no explained total is NOT drawn as zero: a zero-height bar is
                // indistinguishable from one that was measured and explained nothing, and a cohort
                // measured partly before this feature is exactly the mixed case that produces both.
                // Skipping it leaves the acquired background bar alone, and the legend says how many
                // of the replicates carry the series.
                if (!rows[i].HasExplained)
                    continue;

                explainedBars.Add(new Bar
                {
                    Position = i,
                    Value = Finite(rows[i].Ms2Explained) / scale,
                    FillColor = ExplainedBarColor,
                    LineWidth = 0,
                    Size = 0.85,
                });
            }
            plt.Add.Bars(explainedBars);

            var measured = rows.Count(r => r.HasExplained);
            var explainedKey = plt.Add.Marker(double.NaN, double.NaN);
            explainedKey.MarkerStyle.Shape = MarkerShape.FilledSquare;
            explainedKey.MarkerStyle.Size = 14;
            explainedKey.MarkerStyle.FillColor = ExplainedBarColor;
            explainedKey.MarkerStyle.LineWidth = 0;
            explainedKey.LegendText = measured == rows.Count
                ? "explained by any b/y or precursor ion"
                : $"explained by any b/y or precursor ion ({measured:N0} of {rows.Count:N0})";
        }

        var assignedBars = new List<Bar>(rows.Count);
        for (var i = 0; i < rows.Count; i++)
        {
            assignedBars.Add(new Bar
            {
                Position = i,
                Value = Finite(assignedOf(rows[i])) / scale,
                // GroupColor cycles a palette for an unrecognized type, which is right when the
                // colors mean something and wrong here: a cohort with no sample types is one
                // category, and a rainbow across it reads as several. Cycle only on a type that is
                // present and unknown to GroupColor.
                FillColor = string.IsNullOrWhiteSpace(rows[i].SampleType)
                    ? Color.FromHex(TypeColors["experimental"])
                    : GroupColor(rows[i].SampleType, i),
                LineWidth = 0,
                Size = 0.85,
            });
        }
        plt.Add.Bars(assignedBars);

        foreach (var type in rows.Select(r => r.SampleType)
                     .Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var key = plt.Add.Marker(double.NaN, double.NaN);
            key.MarkerStyle.Shape = MarkerShape.FilledSquare;
            key.MarkerStyle.Size = 14;
            key.MarkerStyle.FillColor = string.IsNullOrWhiteSpace(type)
                ? Color.FromHex(TypeColors["experimental"])
                : GroupColor(type, 0);
            key.MarkerStyle.LineWidth = 0;
            // "quantified" only once there is an explained series to tell it apart from; on its
            // own the old wording is what every existing report says and means the same thing.
            var assignedLabel = showExplained ? "quantified" : "assigned to a peptide";
            key.LegendText = string.IsNullOrWhiteSpace(type)
                ? assignedLabel
                : $"{(showExplained ? "quantified" : "assigned")} ({type})";
        }

        plt.ShowLegend(Alignment.UpperRight);
        plt.XLabel($"Replicate (n = {rows.Count:N0})");
        plt.YLabel($"{level.ToString().ToUpperInvariant()} ions{unit}");
        LabelCategoryTicks(plt, rows.Select(r => r.Sample).ToArray());
        StyleQcPlot(plt, fontScale);
        SetPlotTitle(plt, WithIonFraction(title, result, level), fontScale);
        plt.Axes.SetLimits(-0.7, rows.Count - 0.3, 0, tallest > 0 ? tallest / scale * 1.15 : 1);
    }

    /// <summary>
    /// One replicate across the gradient: ions acquired per cycle, and the part of them assigned.
    /// </summary>
    /// <remarks>
    /// <para><b>A cycle, not a fixed time bin.</b> One cycle is one sweep of the isolation scheme -
    /// on the cohort this was written for, 167 MS2 scans and one MS1 - so a cycle is the natural unit
    /// of "what the instrument did once". Cycles are grouped into bins of
    /// <paramref name="binMinutes"/> only to keep the trace readable: an hour-long run has thousands
    /// of them and a point per cycle is noise.</para>
    /// </remarks>
    public static void DrawIonProfile(
        Plot plt, IReadOnlyList<IonCycleRow> cycles, IonLevel level, double binMinutes = 1.0,
        string? title = null, double fontScale = 1.0)
    {
        plt.Clear();   // see DrawIonAccounting: these append, so a redraw stacks without this

        var binned = BinCycles(cycles, level, binMinutes);
        if (binned.Count == 0)
        {
            DrawEmptyState(plt, title ?? "No cycles to profile", fontScale);
            return;
        }

        var x = binned.Select(b => b.RtMin).ToArray();
        var acquired = binned.Select(b => b.Acquired).ToArray();
        var assigned = binned.Select(b => b.Assigned).ToArray();
        var explained = binned.Select(b => b.Explained).ToArray();
        var showExplained = explained.Any(v => v > 0);
        var tallest = Math.Max(acquired.Max(), assigned.Max());
        var (scale, unit) = SignalScale(tallest);

        // Filled acquired with the assigned line over it: the shape a reader already knows how to
        // read as "of that, this much". Stacking would be wrong - the totals nest, they do not
        // partition.
        var band = plt.Add.FillY(
            x, new double[x.Length], acquired.Select(v => v / scale).ToArray());
        band.FillColor = Color.FromHex("#c8ccd4").WithAlpha((byte)140);
        band.LineWidth = 0;
        band.MarkerSize = 0;
        band.LegendText = $"acquired {level.ToString().ToUpperInvariant()} ions";

        if (showExplained)
        {
            var explainedLine = plt.Add.Scatter(x, explained.Select(v => v / scale).ToArray());
            explainedLine.Color = ExplainedBarColor;
            explainedLine.LineWidth = 3;
            explainedLine.MarkerSize = 0;
            explainedLine.LegendText = "explained by any b/y or precursor ion";
        }

        var line = plt.Add.Scatter(x, assigned.Select(v => v / scale).ToArray());
        line.Color = Color.FromHex(TypeColors["experimental"]);
        line.LineWidth = 3;
        line.MarkerSize = 0;
        line.LegendText = showExplained ? "quantified" : "assigned to a peptide";

        plt.ShowLegend(Alignment.UpperRight);
        plt.XLabel("Retention time (min)");
        plt.YLabel($"{level.ToString().ToUpperInvariant()} ions{unit} per {binMinutes:0.##} min");
        StyleQcPlot(plt, fontScale);
        SetPlotTitle(plt, title, fontScale);
        plt.Axes.SetLimits(
            x[0] - binMinutes, x[^1] + binMinutes, 0, tallest > 0 ? tallest / scale * 1.15 : 1);
    }

    /// <summary>
    /// The assigned FRACTION across the gradient, which is where the two totals stop being
    /// interchangeable with their ratio.
    /// </summary>
    /// <remarks>
    /// <para>The absolute traces are dominated by the elution envelope - both rise and fall together
    /// - so a stretch where the analysis explains little of what was acquired is invisible in them
    /// and obvious here.</para>
    /// <para><b>The axis starts at zero and fits the data above it.</b> Pinning it to 0-100% was
    /// tried first, on the reasoning that a fraction is bounded and an autoscaled 4% maximum makes a
    /// bad run look like a full one. On real data that reasoning fails its own test: at 3.4% the
    /// trace sits on the baseline and no structure is visible, which is the one thing this plot
    /// exists to show. What deceives is a non-zero ORIGIN, not a fitted top - so the origin is fixed
    /// at zero, the tick labels state the scale, and the dashed whole-run line carries the figure.</para>
    /// </remarks>
    public static void DrawIonFractionProfile(
        Plot plt, IReadOnlyList<IonCycleRow> cycles, IonLevel level, double binMinutes = 1.0,
        string? title = null, double fontScale = 1.0)
    {
        plt.Clear();   // see DrawIonAccounting: these append, so a redraw stacks without this

        var binned = BinCycles(cycles, level, binMinutes);
        if (binned.Count == 0)
        {
            DrawEmptyState(plt, title ?? "No cycles to profile", fontScale);
            return;
        }

        // Only bins that acquired something: a bin with no ions has no fraction, and plotting zero
        // there would draw a dip that means "nothing was acquired", not "nothing was assigned".
        var points = binned
            .Where(b => b.Acquired > 0)
            .Select(b => (b.RtMin, Fraction: b.Assigned / b.Acquired * 100.0))
            .ToArray();
        if (points.Length == 0)
        {
            DrawEmptyState(plt, title ?? "No acquired ions to take a fraction of", fontScale);
            return;
        }

        // The SAME filter as the quantified line above - acquired only. Excluding bins whose
        // explained total is zero would drop real points, and ScottPlot joins across an omission:
        // the line would sail over exactly the stretch where nothing was explained while the
        // quantified line dipped to zero beneath it. A bin with no acquired ions has no fraction at
        // all and is the only thing either line may skip.
        var explainedPoints = binned
            .Where(b => b.Acquired > 0)
            .Select(b => (b.RtMin, Fraction: b.Explained / b.Acquired * 100.0))
            .ToArray();
        var showExplained = binned.Any(b => b.Explained > 0);

        if (showExplained)
        {
            var explainedLine = plt.Add.Scatter(
                explainedPoints.Select(p => p.RtMin).ToArray(),
                explainedPoints.Select(p => p.Fraction).ToArray());
            explainedLine.Color = ExplainedBarColor;
            explainedLine.LineWidth = 3;
            explainedLine.MarkerSize = 0;
            explainedLine.LegendText = "explained share";
        }

        var line = plt.Add.Scatter(
            points.Select(p => p.RtMin).ToArray(), points.Select(p => p.Fraction).ToArray());
        line.Color = Color.FromHex(TypeColors["experimental"]);
        line.LineWidth = 3;
        line.MarkerSize = 0;
        line.LegendText = showExplained
            ? "quantified share"
            : $"assigned share of acquired {level.ToString().ToUpperInvariant()}";

        var overallAcquired = binned.Sum(b => b.Acquired);
        var overall = overallAcquired > 0 ? binned.Sum(b => b.Assigned) / overallAcquired * 100 : 0;
        var mean = plt.Add.HorizontalLine(overall);
        mean.Color = Colors.Gray.WithAlpha(0.6);
        mean.LineWidth = 2;
        mean.LinePattern = LinePattern.Dashed;
        // Named for the series it belongs to once there are two, or it reads as the average of
        // whichever line the eye landed on first.
        mean.LegendText = showExplained
            ? $"whole run, quantified ({overall:0.#}%)"
            : $"whole run ({overall:0.#}%)";

        plt.ShowLegend(Alignment.UpperRight);
        plt.XLabel("Retention time (min)");
        plt.YLabel(
            showExplained
                ? $"Share of acquired {level.ToString().ToUpperInvariant()} ions (%)"
                : $"Assigned share of acquired {level.ToString().ToUpperInvariant()} ions (%)");
        StyleQcPlot(plt, fontScale);
        SetPlotTitle(plt, title, fontScale);
        // Zero origin always; the top fits the data, with a floor so a near-zero run does not get an
        // absurdly magnified axis, and a ceiling because a fraction cannot exceed 100%.
        var tallest = Math.Max(points.Max(p => p.Fraction), overall);
        if (showExplained)
            tallest = Math.Max(tallest, explainedPoints.Max(p => p.Fraction));
        var top = Math.Min(100.0, Math.Max(MinimumFractionAxisTop, tallest * 1.25));
        plt.Axes.SetLimits(
            points[0].RtMin - binMinutes, points[^1].RtMin + binMinutes, 0, top);
    }

    /// <summary>PNG of <see cref="DrawIonAccounting"/>, for the QC report.</summary>
    public static byte[] IonAccountingPng(
        IonAccountingResult result, IonLevel level, string? title = null,
        int width = Width, int height = Height, double fontScale = 1.0)
    {
        var plt = new Plot();
        DrawIonAccounting(plt, result, level, title, fontScale);
        return plt.GetImageBytes(width, height, ImageFormat.Png);
    }

    /// <summary>PNG of <see cref="DrawIonProfile"/>, for the QC report.</summary>
    public static byte[] IonProfilePng(
        IReadOnlyList<IonCycleRow> cycles, IonLevel level, double binMinutes = 1.0,
        string? title = null, int width = Width, int height = Height, double fontScale = 1.0)
    {
        var plt = new Plot();
        DrawIonProfile(plt, cycles, level, binMinutes, title, fontScale);
        return plt.GetImageBytes(width, height, ImageFormat.Png);
    }

    /// <summary>PNG of <see cref="DrawIonFractionProfile"/>, for the QC report.</summary>
    public static byte[] IonFractionProfilePng(
        IReadOnlyList<IonCycleRow> cycles, IonLevel level, double binMinutes = 1.0,
        string? title = null, int width = Width, int height = Height, double fontScale = 1.0)
    {
        var plt = new Plot();
        DrawIonFractionProfile(plt, cycles, level, binMinutes, title, fontScale);
        return plt.GetImageBytes(width, height, ImageFormat.Png);
    }

    /// <summary>One retention-time bin's totals. Internal so the conservation rule below can be
    /// asserted on its own numbers rather than read back off a rendered plot.</summary>
    /// <summary>
    /// The smallest top the share axis will use, in percent. Below this the plot magnifies noise: a
    /// run assigning a fraction of a percent has nothing to show and should look like it.
    /// </summary>
    private const double MinimumFractionAxisTop = 5.0;

    internal readonly record struct CycleBin(
        double RtMin, double Acquired, double Assigned, double Explained = 0);

    /// <summary>
    /// The explained series' color: between the neutral acquired background and the saturated
    /// assigned bar, because the quantity it shows nests between them.
    /// </summary>
    private static readonly Color ExplainedBarColor = Color.FromHex("#8fa8c8");

    /// <summary>
    /// Group cycles into retention-time bins. Bin membership is by the cycle's START time, so a
    /// cycle belongs to exactly one bin and no signal is counted twice.
    /// </summary>
    internal static List<CycleBin> BinCycles(
        IReadOnlyList<IonCycleRow> cycles, IonLevel level, double binMinutes)
    {
        var bins = new List<CycleBin>();
        if (cycles is null || cycles.Count == 0)
            return bins;

        var width = binMinutes > 0 ? binMinutes : 1.0;
        var usable = cycles
            .Where(c => double.IsFinite(c.RtStartMin))
            .OrderBy(c => c.RtStartMin)
            .ToArray();
        if (usable.Length == 0)
            return bins;

        var acquiredOf = level == IonLevel.Ms1
            ? new Func<IonCycleRow, double>(c => c.Ms1Acquired)
            : c => c.Ms2Acquired;
        var assignedOf = level == IonLevel.Ms1
            ? new Func<IonCycleRow, double>(c => c.Ms1Assigned)
            : c => c.Ms2Assigned;

        // MS1 has no explained series: the theoretical MS1 claim IS the precursor isotope envelope
        // Skyline already extracts, so the two totals cannot differ and one of them would be a line
        // drawn exactly on top of another.
        var explainedOf = level == IonLevel.Ms1
            ? new Func<IonCycleRow, double>(_ => 0)
            : c => c.Ms2Explained;

        var origin = usable[0].RtStartMin;
        var current = -1;
        var acquired = 0.0;
        var assigned = 0.0;
        var explained = 0.0;

        foreach (var cycle in usable)
        {
            var bin = (int)Math.Floor((cycle.RtStartMin - origin) / width);
            if (bin != current)
            {
                if (current >= 0)
                    bins.Add(new CycleBin(
                        origin + (current + 0.5) * width, acquired, assigned, explained));
                current = bin;
                acquired = 0;
                assigned = 0;
                explained = 0;
            }
            acquired += Finite(acquiredOf(cycle));
            assigned += Finite(assignedOf(cycle));
            explained += Finite(explainedOf(cycle));
        }
        bins.Add(new CycleBin(origin + (current + 0.5) * width, acquired, assigned, explained));
        return bins;
    }

    private static Func<IonAccountingRow, double> Selector(IonLevel level, bool acquired) =>
        (level, acquired) switch
        {
            (IonLevel.Ms1, true) => r => r.Ms1Acquired,
            (IonLevel.Ms1, false) => r => r.Ms1Assigned,
            (_, true) => r => r.Ms2Acquired,
            _ => r => r.Ms2Assigned,
        };

    /// <summary>
    /// Add the median assigned fraction to the title, and refuse to when it is not physical.
    /// </summary>
    private static string? WithIonFraction(
        string? title, IonAccountingResult result, IonLevel level)
    {
        var usable = result.Rows.Where(r => r.IsUsable).ToArray();
        if (usable.Length == 0)
            return title;

        // A replicate that assigned more than it acquired taints the median it is in, so the whole
        // figure is withheld and named as a defect rather than quietly excluded.
        if (usable.Any(r => r.Exceeded))
        {
            var suffix = "assigned exceeds acquired in "
                + $"{usable.Count(r => r.Exceeded):N0} replicate(s); fraction not shown";
            return string.IsNullOrEmpty(title) ? suffix : $"{title}{NewLine}{suffix}";
        }

        var fractions = usable
            .Select(r => level == IonLevel.Ms1 ? r.Ms1Fraction : r.Ms2Fraction)
            .Where(double.IsFinite)
            .OrderBy(f => f)
            .ToArray();
        if (fractions.Length == 0)
            return title;

        var name = level.ToString().ToUpperInvariant();
        var median = Median(fractions);

        // Both medians when both exist. The title is the line a reader takes away from the page, and
        // one number out of two invites the reading that it is the whole answer.
        var explained = usable
            .Where(r => r.HasExplained)
            .Select(r => r.Ms2ExplainedFraction)
            .Where(double.IsFinite)
            .OrderBy(f => f)
            .ToArray();

        var text = level == IonLevel.Ms2 && explained.Length > 0
            ? $"median {median:P1} of acquired {name} ions quantified, "
              + $"{Median(explained):P1} explained by any b/y or precursor ion"
            : $"median {median:P1} of acquired {name} ions assigned to a peptide";
        return string.IsNullOrEmpty(title) ? text : $"{title}{NewLine}{text}";
    }

    /// <summary>
    /// Name the bars, or label nothing at all - but never leave the numeric ticks.
    /// </summary>
    /// <remarks>
    /// A bar chart's x positions are category indices, so the default generator prints -0.5, 0, 0.5
    /// and so on: numbers that look like data and mean nothing. Few enough replicates get their own
    /// names; too many get an empty axis, because a 192-replicate cohort cannot show a name per bar
    /// and overlapping labels are worse than none.
    /// </remarks>
    /// <summary>Median of an ALREADY SORTED array.</summary>
    private static double Median(IReadOnlyList<double> sorted) =>
        sorted.Count % 2 == 1
            ? sorted[sorted.Count / 2]
            : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2;

    private static void LabelCategoryTicks(Plot plt, IReadOnlyList<string> samples)
    {
        if (samples.Count > MaxNamedCategories)
        {
            plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.EmptyTickGenerator();
            return;
        }

        var ticks = new ScottPlot.Tick[samples.Count];
        for (var i = 0; i < samples.Count; i++)
            ticks[i] = new ScottPlot.Tick(i, ShortSampleName(samples[i]));
        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(ticks);
        plt.Axes.Bottom.TickLabelStyle.Rotation = -45;
        plt.Axes.Bottom.TickLabelStyle.Alignment = Alignment.UpperRight;
        // Rotated labels do not enlarge the axis on their own, so without this they are drawn up
        // into the data area and across the bars they name.
        plt.Axes.Bottom.MinimumSize = 150;
    }

    /// <summary>Beyond this many bars a name per bar cannot be read.</summary>
    private const int MaxNamedCategories = 16;

    /// <summary>
    /// The replicate half of a PRISM sample id. The batch half is the same for every replicate of a
    /// document and is what makes these labels too long to read.
    /// </summary>
    private static string ShortSampleName(string sample)
    {
        var i = sample.IndexOf("__@__", StringComparison.Ordinal);
        return i < 0 ? sample : sample[..i];
    }
}
