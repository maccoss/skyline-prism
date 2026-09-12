using System;
using System.Linq;
using ScottPlot;
using ScottPlot.Plottables;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The three places the SECOND quantity had to be carried through and was not.
///
/// <para>All three are unit errors, and the project's own history says those cannot be caught by
/// looking at a fraction - both sides carry the same weighting, so the ratio cancels. They are
/// caught by checking the number against the quantity it claims to be.</para>
/// </summary>
public class IonQuantityGuardTests
{
    /// <summary>
    /// A row whose ion and signal totals differ by a constant injection time, so any place that
    /// reaches for the wrong one is off by that factor rather than subtly wrong.
    /// </summary>
    private static IonAccountingRow Row(
        string sample = "s1",
        double ms2Acquired = 1000, double ms2Assigned = 100, double ms2Explained = 300,
        double ms2Signal = 50_000, double ms2SignalAssigned = 5_000,
        double ms2SignalExplained = 15_000,
        bool hasSignal = true) =>
        new(sample, "experimental", $"{sample}.raw", Ms2ReadStatus.Ok, "pwiz-sharp",
            10, 100, ms2Acquired, ms2Acquired, ms2Assigned, ms2Assigned,
            ms2Explained, true, 0, 60, 5, 0, 0, 10,
            Array.Empty<double>(), Array.Empty<double>(), null,
            ms2Signal, ms2Signal, ms2SignalAssigned, ms2SignalAssigned,
            ms2SignalExplained, hasSignal);

    private static IonAccountingResult Result(params IonAccountingRow[] rows) =>
        new("k", "+/-10 ppm", "+/-10 ppm", "167 windows",
            Array.Empty<string>(), 100, true, rows, Array.Empty<IonCycleRow>());

    /// <summary>
    /// THE defect: the explained bar was drawn from the ion count against a scale derived from the
    /// TIC totals, so in Signal mode it came out short by the injection time - below the quantified
    /// bar it is required to nest ABOVE. The plot then said "these peptides explain less than the
    /// run quantifies", which the report elsewhere calls impossible, while the title on the same
    /// image quoted the correct share.
    /// </summary>
    [Fact]
    public void TheExplainedBarIsDrawnInTheQuantityTheAxisNames()
    {
        var result = Result(Row());

        var signal = new Plot();
        PlotRenderer.DrawIonAccounting(
            signal, result, PlotRenderer.IonLevel.Ms2, null, 1.0, PlotRenderer.IonQuantity.Signal);

        // Three bar series: acquired, explained, quantified. The explained one must sit between the
        // other two, which is only true if it was taken from the signal column.
        var series = signal.GetPlottables<BarPlot>().Select(b => b.Bars.First().Value).ToArray();
        Assert.Equal(3, series.Length);
        var acquired = series[0];
        var explained = series[1];
        var quantified = series[2];

        Assert.True(
            explained >= quantified,
            $"explained {explained} fell below quantified {quantified} - drawn in the wrong quantity");
        Assert.True(explained <= acquired, $"explained {explained} exceeded acquired {acquired}");
    }

    /// <summary>
    /// An impossibility in the ion totals does not imply one in the TIC totals, or the reverse: the
    /// ion totals weight each scan by its injection time and the TIC totals do not, so a fault
    /// confined to short-injection scans shows in one and not the other. Whichever quantity is on
    /// screen is the one that has to be checked, or an impossible fraction is drawn with no warning.
    /// </summary>
    [Fact]
    public void TheImpossibilityCheckFollowsTheQuantityOnScreen()
    {
        // Sound as ions, impossible as signal.
        var signalBroken = Row(ms2Assigned: 100, ms2SignalAssigned: 90_000);
        Assert.False(signalBroken.Exceeded);
        Assert.True(signalBroken.SignalExceeded);
        Assert.False(signalBroken.ExceededIn(signal: false));
        Assert.True(signalBroken.ExceededIn(signal: true));

        // And the reverse, so neither is merely a stricter version of the other.
        var ionBroken = Row(ms2Assigned: 5_000, ms2SignalAssigned: 5_000);
        Assert.True(ionBroken.Exceeded);
        Assert.False(ionBroken.SignalExceeded);
    }

    /// <summary>The plot title withholds the median on the quantity it is drawing, not on ions.</summary>
    [Fact]
    public void TheMedianIsWithheldForTheQuantityDrawn()
    {
        var result = Result(Row(ms2Assigned: 100, ms2SignalAssigned: 90_000));

        var ions = new Plot();
        PlotRenderer.DrawIonAccounting(
            ions, result, PlotRenderer.IonLevel.Ms2, "t", 1.0, PlotRenderer.IonQuantity.Ions);
        Assert.DoesNotContain("exceeds acquired", ions.Axes.Title.Label.Text);

        var signal = new Plot();
        PlotRenderer.DrawIonAccounting(
            signal, result, PlotRenderer.IonLevel.Ms2, "t", 1.0, PlotRenderer.IonQuantity.Signal);
        Assert.Contains("exceeds acquired", signal.Axes.Title.Label.Text);
    }

    /// <summary>
    /// The explained impossibility is separate from the assigned one for signal exactly as it is for
    /// ions - a fault confined to the theoretical claim set must not blank a quantified fraction
    /// that is perfectly sound.
    /// </summary>
    [Fact]
    public void ASignalExplainedFaultDoesNotBlankTheQuantifiedFraction()
    {
        var row = Row(ms2SignalExplained: 1_000);   // below the 5,000 quantified: impossible

        Assert.True(row.SignalExplainedImpossible);
        Assert.True(row.ExplainedImpossibleIn(signal: true));
        Assert.False(row.SignalExceeded);
        Assert.False(row.ExplainedImpossibleIn(signal: false));
    }

    /// <summary>A cache with no signal cannot be impossible in it - unknown is not a violation.</summary>
    [Fact]
    public void WithoutSignalThereIsNoSignalImpossibility()
    {
        var row = Row(ms2Signal: 0, ms2SignalAssigned: 0, ms2SignalExplained: 0, hasSignal: false);

        Assert.False(row.SignalExceeded);
        Assert.False(row.SignalExplainedImpossible);
    }
}
