using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The order the ion accounting bars are drawn in.
///
/// <para>The plot has one bar per replicate and no room to label them, so the order IS the axis -
/// it is the only thing that says which bar is which, and it decides what the plot can show.</para>
/// </summary>
public class IonRowOrderTests
{
    private static IonAccountingRow Row(
        string sample, string type = "experimental", string? file = null, DateTime? acquired = null) =>
        new(sample, type, file ?? $"{sample}.raw", Ms2ReadStatus.Ok, "pwiz-sharp",
            100, 1000, 1e6, 3e6, 4e5, 1e5, 2e5, true, 0, 60, 10, 0, 0, 100,
            Array.Empty<double>(), Array.Empty<double>(), acquired);

    private static string[] Names(IEnumerable<IonAccountingRow> rows) =>
        rows.Select(r => r.Sample).ToArray();

    /// <summary>
    /// THE case this exists for on a plate: ordinal comparison puts a 48-well plate in the order
    /// A1, A10, A11, A12, A2 - which on a plot whose entire axis is the order is not a cosmetic
    /// problem, it is the axis being wrong.
    /// </summary>
    [Fact]
    public void FileNameOrderReadsTheDigitsAsNumbers()
    {
        var rows = new[] { Row("A10"), Row("A2"), Row("A1"), Row("A12"), Row("B1") };

        Assert.Equal(
            new[] { "A1", "A2", "A10", "A12", "B1" },
            Names(IonRowOrder.Sort(rows, IonRowOrder.By.FileName)));
    }

    /// <summary>Zero padding is a spelling, not a value: 007 and 7 are the same injection.</summary>
    [Fact]
    public void ZeroPaddingDoesNotChangeTheOrder()
    {
        var rows = new[] { Row("run-010"), Row("run-2"), Row("run-007") };

        Assert.Equal(
            new[] { "run-2", "run-007", "run-010" },
            Names(IonRowOrder.Sort(rows, IonRowOrder.By.FileName)));
    }

    /// <summary>
    /// Run order comes from the data file's own start timestamp, which is the only honest source
    /// for it - neither the file name nor the order the files were read in is the order they were
    /// acquired in.
    /// </summary>
    [Fact]
    public void RunOrderFollowsTheAcquisitionTimestampNotTheName()
    {
        var t0 = new DateTime(2026, 5, 20, 8, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row("A1", acquired: t0.AddHours(3)),
            Row("A2", acquired: t0),
            Row("A3", acquired: t0.AddHours(1)),
        };

        Assert.True(IonRowOrder.CanOrderByRun(rows));
        Assert.Equal(
            new[] { "A2", "A3", "A1" },
            Names(IonRowOrder.Sort(rows, IonRowOrder.By.RunOrder)));
    }

    /// <summary>
    /// A cache measured before the timestamp was recorded cannot state run order. It falls back to
    /// file name rather than interleaving the rows that cannot answer - which would present an
    /// arbitrary order as an acquisition one, and drift is exactly what a reader would take from it.
    /// </summary>
    [Fact]
    public void WithoutTimestampsRunOrderFallsBackToFileNameAndSaysItCannot()
    {
        var rows = new[] { Row("A10"), Row("A2"), Row("A1") };

        Assert.False(IonRowOrder.CanOrderByRun(rows));
        Assert.Equal(
            new[] { "A1", "A2", "A10" },
            Names(IonRowOrder.Sort(rows, IonRowOrder.By.RunOrder)));
    }

    /// <summary>A partly-stamped cache is not a run order either - one missing row is enough.</summary>
    [Fact]
    public void OneUnstampedRowIsEnoughToRefuseRunOrder()
    {
        var t0 = new DateTime(2026, 5, 20, 8, 0, 0, DateTimeKind.Utc);
        var rows = new[] { Row("A1", acquired: t0), Row("A2"), Row("A3", acquired: t0.AddHours(1)) };

        Assert.False(IonRowOrder.CanOrderByRun(rows));
    }

    /// <summary>
    /// Grouped by type, so the controls can be seen sitting where the experimental samples do - or
    /// not. Within a group the plate order still holds.
    /// </summary>
    [Fact]
    public void SampleTypeGroupsAndThenOrdersWithinTheGroup()
    {
        var rows = new[]
        {
            Row("A10", "qc"), Row("A2", "experimental"), Row("A1", "reference"),
            Row("A3", "experimental"), Row("A4", "qc"),
        };

        Assert.Equal(
            new[] { "A2", "A3", "A1", "A4", "A10" },
            Names(IonRowOrder.Sort(rows, IonRowOrder.By.SampleType)));
    }

    /// <summary>
    /// A type PRISM does not name sorts after the ones it does, rather than being dropped or folded
    /// into a group it does not belong to.
    /// </summary>
    [Fact]
    public void AnUnknownSampleTypeSortsLastRatherThanVanishing()
    {
        var rows = new[] { Row("A1", "blank"), Row("A2", "experimental"), Row("A3", "") };

        var sorted = IonRowOrder.Sort(rows, IonRowOrder.By.SampleType);

        Assert.Equal(3, sorted.Count);
        Assert.Equal("A2", sorted[0].Sample);
        Assert.Equal(new[] { "A1", "A2", "A3" }, Names(sorted).OrderBy(n => n).ToArray());
    }

    /// <summary>
    /// A replicate with no data file cannot state when it was acquired and never will - reference
    /// and QC injections named identically in every plate are routinely left unpaired, which is a
    /// documented normal case. Counting their silence disabled run order for the whole cohort
    /// forever, and the fallback message blamed a cache that a re-measure cannot fix.
    /// </summary>
    [Fact]
    public void AnUnpairedReplicateDoesNotDisableRunOrderForTheCohort()
    {
        var t0 = new DateTime(2026, 5, 20, 8, 0, 0, DateTimeKind.Utc);
        var rows = new[]
        {
            Row("A1", acquired: t0.AddHours(2)),
            Unpaired("QC-shared"),
            Row("A2", acquired: t0),
        };

        Assert.True(IonRowOrder.CanOrderByRun(rows));

        // And it sorts LAST rather than at the epoch: it has no place in an acquisition order, and
        // first would read as one.
        Assert.Equal(
            new[] { "A2", "A1", "QC-shared" },
            Names(IonRowOrder.Sort(rows, IonRowOrder.By.RunOrder)));
    }

    /// <summary>A cohort of nothing but unpaired replicates still cannot state a run order.</summary>
    [Fact]
    public void WithNoPairedReplicateThereIsNoRunOrder()
    {
        Assert.False(IonRowOrder.CanOrderByRun(new[] { Unpaired("a"), Unpaired("b") }));
    }

    private static IonAccountingRow Unpaired(string sample) =>
        new(sample, "qc", "", Ms2ReadStatus.NotFound, "none",
            0, 0, 0, 0, 0, 0, 0, false, double.NaN, double.NaN, 0, 0, 0, 0,
            Array.Empty<double>(), Array.Empty<double>(), null);

    /// <summary>Sorting never loses or duplicates a replicate, whichever key is used.</summary>
    [Theory]
    [InlineData(IonRowOrder.By.RunOrder)]
    [InlineData(IonRowOrder.By.FileName)]
    [InlineData(IonRowOrder.By.SampleType)]
    public void EveryReplicateSurvivesEveryOrdering(IonRowOrder.By by)
    {
        var t0 = new DateTime(2026, 5, 20, 8, 0, 0, DateTimeKind.Utc);
        var rows = new List<IonAccountingRow>();
        for (var i = 0; i < 25; i++)
        {
            rows.Add(Row(
                $"A{i + 1}",
                i % 5 == 0 ? "qc" : "experimental",
                acquired: i % 3 == 0 ? t0.AddMinutes(i * 7) : null));
        }

        var sorted = IonRowOrder.Sort(rows, by);

        Assert.Equal(rows.Count, sorted.Count);
        Assert.Equal(
            rows.Select(r => r.Sample).OrderBy(s => s).ToArray(),
            sorted.Select(r => r.Sample).OrderBy(s => s).ToArray());
    }

    /// <summary>Nothing to sort is not an error - an empty or single-row cache still draws.</summary>
    [Fact]
    public void EmptyAndSingleRowAreLeftAlone()
    {
        Assert.Empty(IonRowOrder.Sort(Array.Empty<IonAccountingRow>(), IonRowOrder.By.RunOrder));
        Assert.Single(IonRowOrder.Sort(new[] { Row("A1") }, IonRowOrder.By.SampleType));
        Assert.False(IonRowOrder.CanOrderByRun(Array.Empty<IonAccountingRow>()));
    }
}
