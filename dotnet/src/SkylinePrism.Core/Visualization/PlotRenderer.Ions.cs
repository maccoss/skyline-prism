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
    /// of this feature reported a fraction 7x too large for exactly that kind of reason.</para>
    /// </remarks>
    public static void DrawIonAccounting(
        Plot plt, IonAccountingResult result, IonLevel level, string? title = null,
        double fontScale = 1.0)
    {
        var rows = result.Rows;
        if (rows.Count == 0)
        {
            DrawEmptyState(plt, title ?? "No ion accounting to show", fontScale);
            return;
        }

        var acquiredOf = Selector(level, acquired: true);
        var assignedOf = Selector(level, acquired: false);
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

        var assignedBars = new List<Bar>(rows.Count);
        for (var i = 0; i < rows.Count; i++)
        {
            assignedBars.Add(new Bar
            {
                Position = i,
                Value = Finite(assignedOf(rows[i])) / scale,
                FillColor = GroupColor(rows[i].SampleType, i),
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
            key.MarkerStyle.FillColor = GroupColor(type, 0);
            key.MarkerStyle.LineWidth = 0;
            key.LegendText = string.IsNullOrWhiteSpace(type)
                ? "assigned to a peptide"
                : $"assigned ({type})";
        }

        plt.ShowLegend(Alignment.UpperRight);
        plt.XLabel($"Replicate (n = {rows.Count:N0})");
        plt.YLabel($"{level.ToString().ToUpperInvariant()} ions{unit}");
        HideCategoryTicks(plt, rows.Count);
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
        var binned = BinCycles(cycles, level, binMinutes);
        if (binned.Count == 0)
        {
            DrawEmptyState(plt, title ?? "No cycles to profile", fontScale);
            return;
        }

        var x = binned.Select(b => b.RtMin).ToArray();
        var acquired = binned.Select(b => b.Acquired).ToArray();
        var assigned = binned.Select(b => b.Assigned).ToArray();
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

        var line = plt.Add.Scatter(x, assigned.Select(v => v / scale).ToArray());
        line.Color = Color.FromHex(TypeColors["experimental"]);
        line.LineWidth = 3;
        line.MarkerSize = 0;
        line.LegendText = "assigned to a peptide";

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
    /// The absolute traces are dominated by the elution envelope - both rise and fall together - so a
    /// stretch where the analysis explains little of what was acquired is invisible in them and
    /// obvious here. The axis is fixed to 0-100%: a fraction is bounded, and letting the axis
    /// autoscale to a 4% maximum makes a bad run look like a full one.
    /// </remarks>
    public static void DrawIonFractionProfile(
        Plot plt, IReadOnlyList<IonCycleRow> cycles, IonLevel level, double binMinutes = 1.0,
        string? title = null, double fontScale = 1.0)
    {
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

        var line = plt.Add.Scatter(
            points.Select(p => p.RtMin).ToArray(), points.Select(p => p.Fraction).ToArray());
        line.Color = Color.FromHex(TypeColors["experimental"]);
        line.LineWidth = 3;
        line.MarkerSize = 0;
        line.LegendText = $"assigned share of acquired {level.ToString().ToUpperInvariant()}";

        var overallAcquired = binned.Sum(b => b.Acquired);
        var overall = overallAcquired > 0 ? binned.Sum(b => b.Assigned) / overallAcquired * 100 : 0;
        var mean = plt.Add.HorizontalLine(overall);
        mean.Color = Colors.Gray.WithAlpha(0.6);
        mean.LineWidth = 2;
        mean.LinePattern = LinePattern.Dashed;
        mean.LegendText = $"whole run ({overall:0.#}%)";

        plt.ShowLegend(Alignment.UpperRight);
        plt.XLabel("Retention time (min)");
        plt.YLabel($"Assigned share of acquired {level.ToString().ToUpperInvariant()} ions (%)");
        StyleQcPlot(plt, fontScale);
        SetPlotTitle(plt, title, fontScale);
        plt.Axes.SetLimits(
            points[0].RtMin - binMinutes, points[^1].RtMin + binMinutes, 0, 100);
    }

    /// <summary>PNG of <see cref="DrawIonAccounting"/>, for the QC report.</summary>
    public static byte[] IonAccountingPng(
        IonAccountingResult result, IonLevel level, string? title = null,
        int width = Width, int height = Height, double fontScale = 1.0)
    {
        var plt = new Plot();
        DrawIonAccounting(plt, result, level, title, fontScale);
        return plt.GetImageBytes(width, height);
    }

    /// <summary>PNG of <see cref="DrawIonProfile"/>, for the QC report.</summary>
    public static byte[] IonProfilePng(
        IReadOnlyList<IonCycleRow> cycles, IonLevel level, double binMinutes = 1.0,
        string? title = null, int width = Width, int height = Height, double fontScale = 1.0)
    {
        var plt = new Plot();
        DrawIonProfile(plt, cycles, level, binMinutes, title, fontScale);
        return plt.GetImageBytes(width, height);
    }

    /// <summary>PNG of <see cref="DrawIonFractionProfile"/>, for the QC report.</summary>
    public static byte[] IonFractionProfilePng(
        IReadOnlyList<IonCycleRow> cycles, IonLevel level, double binMinutes = 1.0,
        string? title = null, int width = Width, int height = Height, double fontScale = 1.0)
    {
        var plt = new Plot();
        DrawIonFractionProfile(plt, cycles, level, binMinutes, title, fontScale);
        return plt.GetImageBytes(width, height);
    }

    private readonly record struct CycleBin(double RtMin, double Acquired, double Assigned);

    /// <summary>
    /// Group cycles into retention-time bins. Bin membership is by the cycle's START time, so a
    /// cycle belongs to exactly one bin and no signal is counted twice.
    /// </summary>
    private static List<CycleBin> BinCycles(
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

        var origin = usable[0].RtStartMin;
        var current = -1;
        var acquired = 0.0;
        var assigned = 0.0;

        foreach (var cycle in usable)
        {
            var bin = (int)Math.Floor((cycle.RtStartMin - origin) / width);
            if (bin != current)
            {
                if (current >= 0)
                    bins.Add(new CycleBin(origin + (current + 0.5) * width, acquired, assigned));
                current = bin;
                acquired = 0;
                assigned = 0;
            }
            acquired += Finite(acquiredOf(cycle));
            assigned += Finite(assignedOf(cycle));
        }
        bins.Add(new CycleBin(origin + (current + 0.5) * width, acquired, assigned));
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

        var median = fractions.Length % 2 == 1
            ? fractions[fractions.Length / 2]
            : (fractions[fractions.Length / 2 - 1] + fractions[fractions.Length / 2]) / 2;
        var text = $"median {median:P1} of acquired {level.ToString().ToUpperInvariant()} ions "
            + $"assigned to a peptide";
        return string.IsNullOrEmpty(title) ? text : $"{title}{NewLine}{text}";
    }

    /// <summary>
    /// Category axes with more entries than fit: the labels are dropped rather than overlapped. A
    /// 192-replicate cohort cannot show a name per bar, and the hover readout names them instead.
    /// </summary>
    private static void HideCategoryTicks(Plot plt, int count)
    {
        if (count <= 24)
            return;
        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.EmptyTickGenerator();
    }
}
