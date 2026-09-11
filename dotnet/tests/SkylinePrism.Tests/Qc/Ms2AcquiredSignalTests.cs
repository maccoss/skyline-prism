using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The acquired denominator: persisted separately from the accounting, joined at render time, and
/// absent far more often than not. What these pin is that "no denominator" stays visibly distinct
/// from "a denominator of zero" all the way to the plot - the two are opposite findings, and the
/// whole point of the fraction is lost if they render alike.
/// </summary>
public class Ms2AcquiredSignalTests
{
    private static string NewDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_acq_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static Ms2AcquiredSignal.Entry Entry(
        string sample, double total, string status = nameof(Ms2ReadStatus.Ok)) =>
        new(sample, $@"C:\raw\{sample}.raw", status, "test-reader",
            nameof(Ms2SignalSource.ReportedTic), 1000, 9000, total, 0.5, 60.0,
            nameof(Ms2CycleModel.Ms1Bounded));

    private static Ms2SignalAccounting.Row Row(string sample, double assigned) =>
        new(sample, "experimental", assigned, assigned * 1.2, Array.Empty<double>(),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    private static Ms2SignalAccounting.Result Result(params Ms2SignalAccounting.Row[] rows) =>
        new(rows, Array.Empty<string>(), Array.Empty<string>(), Array.Empty<int>(),
            0, "10 ppm", "scheme", true);

    [Fact]
    public void EntriesRoundTripThroughParquet()
    {
        var dir = NewDir();
        try
        {
            var entries = new[] { Entry("A__@__p1", 1.5e9), Entry("B__@__p1", 2.5e9) };
            Ms2AcquiredSignal.Write(dir, entries);

            var read = Ms2AcquiredSignal.Read(dir);
            Assert.Equal(2, read.Count);
            Assert.Equal("A__@__p1", read[0].Sample);
            Assert.Equal(1.5e9, read[0].TotalMs2Signal, 3);
            Assert.Equal("test-reader", read[0].Reader);
            Assert.Equal(9000, read[1].Ms2Count);
            Assert.Equal(nameof(Ms2CycleModel.Ms1Bounded), read[1].CycleModel);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    /// <summary>
    /// A file that could not be read is still WRITTEN, with its status - so the report can say which
    /// replicates have no denominator. But it must never reach the totals, or an unreadable file
    /// would become a denominator of zero and the fraction would divide by it.
    /// </summary>
    [Fact]
    public void AFailedReadIsRecordedButIsNotATotal()
    {
        var dir = NewDir();
        try
        {
            Ms2AcquiredSignal.Write(dir, new[]
            {
                Entry("A__@__p1", 1.5e9),
                Entry("B__@__p1", double.NaN, nameof(Ms2ReadStatus.NotFound)),
                Entry("C__@__p1", 0, nameof(Ms2ReadStatus.Ok)),   // read fine, genuinely empty
            });

            Assert.Equal(3, Ms2AcquiredSignal.Read(dir).Count);

            var totals = Ms2AcquiredSignal.ReadTotals(dir);
            Assert.Single(totals);
            Assert.Equal(1.5e9, totals["A__@__p1"], 3);
            Assert.False(totals.ContainsKey("B__@__p1"));
            // Zero acquired signal is not a usable denominator either - dividing by it is not a
            // fraction, it is an infinity.
            Assert.False(totals.ContainsKey("C__@__p1"));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    /// <summary>
    /// Cycles are captured on the same pass as the totals, for every replicate, because going back
    /// for them means re-reading the cohort - ~1.1 TB on the one this was written against. So the
    /// round trip has to hold, and it has to keep replicates separable.
    /// </summary>
    [Fact]
    public void CyclesRoundTripAndStaySeparatedByReplicate()
    {
        var dir = NewDir();
        try
        {
            var cycles = new Dictionary<string, IReadOnlyList<Ms2Cycle>>
            {
                ["A__@__p1"] = new[]
                {
                    new Ms2Cycle(0, 0.0, 0.5, 12, 1e6),
                    new Ms2Cycle(1, 0.5, 1.0, 11, 2e6),
                },
                ["B__@__p1"] = new[] { new Ms2Cycle(0, 0.0, 0.5, 9, 3e6) },
            };
            Ms2AcquiredSignal.WriteCycles(dir, new[] { "A__@__p1", "B__@__p1" }, cycles);

            var a = Ms2AcquiredSignal.ReadCycles(dir, "A__@__p1");
            Assert.Equal(2, a.Count);
            Assert.Equal(0, a[0].Index);
            Assert.Equal(0.5, a[0].RtStopMin, 6);
            Assert.Equal(2e6, a[1].Ms2Signal, 3);
            Assert.Equal(11, a[1].Ms2Count);

            Assert.Single(Ms2AcquiredSignal.ReadCycles(dir, "B__@__p1"));
            Assert.Empty(Ms2AcquiredSignal.ReadCycles(dir, "C__@__p1"));

            Assert.Equal(
                new[] { "A__@__p1", "B__@__p1" },
                Ms2AcquiredSignal.SamplesWithCycles(dir).OrderBy(x => x, StringComparer.Ordinal));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void AbsentCyclesReadAsEmptyRatherThanThrowing()
    {
        var dir = NewDir();
        try
        {
            Assert.Empty(Ms2AcquiredSignal.ReadCycles(dir, "A__@__p1"));
            Assert.Empty(Ms2AcquiredSignal.SamplesWithCycles(dir));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void AnAbsentFileReadsAsNothingRatherThanThrowing()
    {
        var dir = NewDir();
        try
        {
            Assert.Empty(Ms2AcquiredSignal.Read(dir));
            Assert.Empty(Ms2AcquiredSignal.ReadTotals(dir));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void WithAcquiredFillsOnlyTheRowsItHas()
    {
        var result = Result(Row("A__@__p1", 4e8), Row("B__@__p1", 6e8));
        Assert.False(result.HasAcquired);
        Assert.All(result.Rows, r => Assert.False(double.IsFinite(r.AcquiredFraction)));

        var joined = result.WithAcquired(new Dictionary<string, double> { ["A__@__p1"] = 1e9 });

        Assert.True(joined.HasAcquired);
        Assert.Equal(0.4, joined.Rows[0].AcquiredFraction, 6);
        // B had no data file. NaN, not 0: "not measured" and "none assigned" are opposite findings.
        Assert.False(double.IsFinite(joined.Rows[1].AcquiredFraction));
        Assert.False(double.IsFinite(joined.Rows[1].AcquiredArea));

        // The original is untouched - Result is a record and the join returns a copy, so a caller
        // holding the cached accounting does not silently acquire a denominator.
        Assert.False(result.HasAcquired);
    }

    [Fact]
    public void TheMedianFractionIgnoresReplicatesWithNoDenominator()
    {
        var result = Result(Row("A__@__p1", 1e8), Row("B__@__p1", 3e8), Row("C__@__p1", 9e8))
            .WithAcquired(new Dictionary<string, double>
            {
                ["A__@__p1"] = 1e9,   // 0.1
                ["C__@__p1"] = 3e9,   // 0.3
            });

        // Median of {0.1, 0.3}, with B excluded rather than counted as zero - which would have
        // dragged the reported fraction down to 0.1 and made the analysis look worse than it is.
        Assert.Equal(0.2, result.MedianAcquiredFraction(), 6);
    }

    [Fact]
    public void NoDenominatorAnywhereGivesNoFraction()
    {
        var result = Result(Row("A__@__p1", 1e8));
        Assert.False(double.IsFinite(result.MedianAcquiredFraction()));

        // An empty map is a no-op rather than an exception or a wipe.
        Assert.False(result.WithAcquired(new Dictionary<string, double>()).HasAcquired);
    }
}
