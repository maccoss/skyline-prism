using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The ion accounting: the geometry loaded out of <c>merged_data/</c>, the window mapping that has to
/// agree with it, and the cache that stands in for a terabyte of instrument files.
/// </summary>
public class IonAccountingTests
{
    // ---------------------------------------------------------------- window mapping

    /// <summary>
    /// The claims and the scans have to land in the SAME window index or nothing matches and the
    /// fraction reads as zero, so both sides resolve through one method. An overlapping scheme can
    /// cover one m/z with several windows; the narrowest wins, matching how the claims were placed.
    /// </summary>
    [Fact]
    public void AScanIsPlacedInTheNarrowestWindowCoveringIt()
    {
        var scheme = new IsolationScheme("overlapping", new[]
        {
            new IsolationWindow(400, 420),   // wide
            new IsolationWindow(405, 410),   // narrow, inside the first
        });
        var request = new IonAccountingRequest(
            new ClaimedSignalIndex(Array.Empty<ClaimedRegion>()), scheme, Array.Empty<string>());

        Assert.Equal(1, request.WindowIndexFor(407, rtMinutes: 5));   // both cover it; narrowest wins
        Assert.Equal(0, request.WindowIndexFor(402, rtMinutes: 5));   // only the wide one
    }

    /// <summary>
    /// A scan no window covers is not window 0. Returning 0 there would credit it with another
    /// window's claims, which is the kind of error that produces a plausible wrong number.
    /// </summary>
    [Fact]
    public void AScanOutsideEveryWindowGetsNoWindow()
    {
        var scheme = IsolationScheme.Cycle("c", start: 400, step: 4, width: 4, count: 2);
        var request = new IonAccountingRequest(
            new ClaimedSignalIndex(Array.Empty<ClaimedRegion>()), scheme, Array.Empty<string>());

        Assert.Equal(-1, request.WindowIndexFor(900, rtMinutes: 5));
        Assert.Equal(-1, request.WindowIndexFor(double.NaN, rtMinutes: 5));
    }

    // ---------------------------------------------------------------- the fraction

    [Fact]
    public void FractionsAreAssignedOverAcquiredAndNaNWithNoDenominator()
    {
        var row = Row(ms1Acquired: 200, ms2Acquired: 50, ms1Assigned: 50, ms2Assigned: 5);
        Assert.Equal(0.25, row.Ms1Fraction, 9);
        Assert.Equal(0.10, row.Ms2Fraction, 9);
        Assert.False(row.Exceeded);

        var noFile = Row(0, 0, 0, 0);
        Assert.True(double.IsNaN(noFile.Ms1Fraction));
        Assert.True(double.IsNaN(noFile.Ms2Fraction));
        Assert.False(noFile.IsUsable);
    }

    /// <summary>
    /// More assigned than acquired is impossible, so it is a defect rather than a number to clamp.
    /// Clamping to 100% would turn a visible bug into a plausible reading - which is exactly how the
    /// previous version of this feature shipped a fraction 7x too large.
    /// </summary>
    [Fact]
    public void AssigningMoreThanWasAcquiredIsReportedNotClamped()
    {
        var row = Row(ms1Acquired: 100, ms2Acquired: 100, ms1Assigned: 100, ms2Assigned: 101);
        Assert.True(row.Exceeded);
        Assert.True(row.Ms2Fraction > 1.0);   // NOT clamped; the caller refuses to draw it
    }

    /// <summary>
    /// Three representatives, not one: a single panel hides whether the cohort is uniform, and best
    /// against worst is what tells a user whether one injection misbehaved or the whole run did.
    /// </summary>
    [Fact]
    public void RepresentativesAreBestMedianAndWorstByMs2Fraction()
    {
        var rows = new[]
        {
            Row(100, 100, 10, 10, "worst"),    // 10%
            Row(100, 100, 20, 20, "low"),      // 20%
            Row(100, 100, 30, 30, "median"),   // 30%
            Row(100, 100, 40, 40, "high"),     // 40%
            Row(100, 100, 50, 50, "best"),     // 50%
        };
        var picks = Result(rows).Representatives().Select(r => r.Sample).ToArray();

        Assert.Equal(new[] { "best", "median", "worst" }, picks);
    }

    [Fact]
    public void RepresentativesExcludeRowsWhoseFractionIsImpossible()
    {
        var rows = new[]
        {
            Row(100, 100, 10, 10, "ok-low"),
            Row(100, 100, 30, 30, "ok-high"),
            Row(100, 100, 500, 500, "broken"),
        };
        var picks = Result(rows).Representatives().Select(r => r.Sample).ToArray();

        Assert.DoesNotContain("broken", picks);
        Assert.Equal(2, picks.Length);
    }

    /// <summary>
    /// The per-scan ion count is the ONLY check here that can catch a units error, and it is checked
    /// against the real numbers that motivated it.
    /// </summary>
    /// <remarks>
    /// Multiplying the intensity by the injection time in milliseconds rather than seconds makes
    /// every total 1000x too large while leaving the fraction bit-identical, because the error
    /// scales numerator and denominator alike and cancels. Nothing built on the ratio can see it.
    /// The values below are the real ones from a 4.44 GB Astral file, before and after the fix.
    /// </remarks>
    [Fact]
    public void TheMillisecondErrorIsCaughtByPerScanIonsAndNotByTheFraction()
    {
        // What the milliseconds bug produced: 1,006 MS1 scans totalling 3.745e11.
        var wrong = Scan(ms1Count: 1006, ms1Acquired: 3.745e11, ms2Count: 167914, ms2Acquired: 1.228e12);
        // And what seconds give, exactly 1000x smaller.
        var right = Scan(ms1Count: 1006, ms1Acquired: 3.745e8, ms2Count: 167914, ms2Acquired: 1.228e9);

        Assert.True(wrong.IonScaleImplausible, "3.7e8 ions in one MS1 scan should be rejected");
        Assert.False(right.IonScaleImplausible, "3.7e5 ions in one MS1 scan is an ordinary AGC target");

        // Stated as ranges, because what matters is the ORDER of magnitude against an AGC target.
        Assert.InRange(right.MeanMs1IonsPerScan, 3.0e5, 4.5e5);   // ~372,000 per survey scan
        Assert.InRange(right.MeanMs2IonsPerScan, 6.0e3, 9.0e3);   // ~7,300 per 3 Th MS2 scan
        Assert.InRange(wrong.MeanMs1IonsPerScan, 3.0e8, 4.5e8);   // the same, 1000x over

        // And the point of the whole test: the FRACTION cannot tell them apart.
        Assert.Equal(wrong.Ms1Fraction, right.Ms1Fraction, 12);
        Assert.Equal(wrong.Ms2Fraction, right.Ms2Fraction, 12);
    }

    [Fact]
    public void AnEmptyOrUnreadReplicateIsNotCalledImplausible()
    {
        // No scans at all: nothing to judge, and a NaN mean must not read as a defect.
        var none = Scan(ms1Count: 0, ms1Acquired: 0, ms2Count: 0, ms2Acquired: 0);
        Assert.False(none.IonScaleImplausible);
        Assert.True(double.IsNaN(none.MeanMs1IonsPerScan));
    }

    /// <summary>A row with the given scan counts and totals, at a fixed 40%/3.4% assigned share.</summary>
    private static IonAccountingRow Scan(
        int ms1Count, double ms1Acquired, int ms2Count, double ms2Acquired) =>
        new("s", "experimental", "s.raw", Ms2ReadStatus.Ok, "test", ms1Count, ms2Count,
            ms1Acquired, ms2Acquired, ms1Acquired * 0.405, ms2Acquired * 0.034,
            Ms2Explained: 0, HasExplained: false,
            0, 60, 1000, 0, 0, 1, Array.Empty<double>(), Array.Empty<double>());

    // ---------------------------------------------------------------- the cache

    /// <summary>
    /// The cache stands in for re-reading a terabyte, so what comes back has to be what went in -
    /// including the per-list totals, which live in their own file and are keyed by BIT ORDER rather
    /// than sorted, because the masks were built against that order.
    /// </summary>
    [Fact]
    public void TheCacheRoundTripsIncludingPerListTotalsAndCycles()
    {
        var dir = TempDir();
        try
        {
            var rows = new[]
            {
                Row(1000, 500, 400, 100, "s1", ms1ByList: new[] { 100.0, 25.0 },
                    ms2ByList: new[] { 40.0, 8.0 }),
                Row(2000, 800, 900, 200, "s2", ms1ByList: new[] { 300.0, 50.0 },
                    ms2ByList: new[] { 60.0, 9.0 }),
            };
            var cycles = new[]
            {
                new IonCycleRow("s1", 0, 0.0, 0.5, 1, 167, 500, 250, 200, 50),
                new IonCycleRow("s1", 1, 0.5, 1.0, 1, 167, 500, 250, 200, 50),
                new IonCycleRow("s2", 0, 0.0, 0.5, 1, 167, 2000, 800, 900, 200),
            };
            var written = new IonAccountingResult(
                "key-1", "+/-10 ppm (centroided)", "+/-10 ppm (centroided)", "167 windows",
                new[] { "Hemolysis", "Contaminants" }, 4321, true, rows, cycles);

            IonAccountingStore.Write(dir, written);
            var read = IonAccountingStore.Read(dir);

            Assert.NotNull(read);
            Assert.True(read!.MatchesSettings("key-1"));
            Assert.False(read.MatchesSettings("key-2"));
            Assert.Equal(4321, read.AssignedPeptides);
            Assert.Equal(new[] { "Hemolysis", "Contaminants" }, read.ListNames);
            Assert.Equal(2, read.Rows.Count);

            var s1 = read.Rows.Single(r => r.Sample == "s1");
            Assert.Equal(1000, s1.Ms1Acquired, 6);
            Assert.Equal(100, s1.Ms1Assigned / 4, 6);         // 400 assigned
            Assert.Equal(new[] { 100.0, 25.0 }, s1.Ms1ByList);
            Assert.Equal(new[] { 40.0, 8.0 }, s1.Ms2ByList);

            // Cycles are read on demand and per replicate: a cohort's cycles run to hundreds of
            // thousands of rows and a time plot shows one replicate.
            var s1Cycles = IonAccountingStore.ReadCycles(dir, "s1");
            Assert.Equal(2, s1Cycles.Count);
            Assert.Equal(250, s1Cycles[0].Ms2Acquired, 6);
            Assert.Equal(3, IonAccountingStore.ReadCycles(dir).Count);
            Assert.Equal(new[] { "s1", "s2" }, IonAccountingStore.SamplesWithCycles(dir));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// A re-run with the lists removed must not leave the previous run's per-list bars behind - the
    /// numbers would be real, and would belong to settings nobody asked for.
    /// </summary>
    [Fact]
    public void RemovingTheListsRemovesTheirCachedTotals()
    {
        var dir = TempDir();
        try
        {
            IonAccountingStore.Write(dir, new IonAccountingResult(
                "k", "t", "p", "s", new[] { "Hemolysis" }, 1, true,
                new[] { Row(10, 10, 1, 1, "s1", ms1ByList: new[] { 1.0 }, ms2ByList: new[] { 1.0 }) },
                Array.Empty<IonCycleRow>()));
            Assert.True(File.Exists(Path.Combine(dir, IonAccountingStore.ListsFile)));

            IonAccountingStore.Write(dir, new IonAccountingResult(
                "k", "t", "p", "s", Array.Empty<string>(), 1, true,
                new[] { Row(10, 10, 1, 1, "s1") }, Array.Empty<IonCycleRow>()));

            Assert.False(File.Exists(Path.Combine(dir, IonAccountingStore.ListsFile)));
            Assert.Empty(IonAccountingStore.Read(dir)!.ListNames);
        }
        finally
        {
            Cleanup(dir);
        }
    }

    [Fact]
    public void TheSettingsKeyChangesWithEverythingThatChangesTheNumbers()
    {
        var sources = new[] { "a.raw" };
        var baseline = IonAccountingStore.SettingsKeyFor("10 ppm", "10 ppm", "scheme", new[] { "L" }, sources);

        Assert.NotEqual(baseline,
            IonAccountingStore.SettingsKeyFor("20 ppm", "10 ppm", "scheme", new[] { "L" }, sources));
        Assert.NotEqual(baseline,
            IonAccountingStore.SettingsKeyFor("10 ppm", "20 ppm", "scheme", new[] { "L" }, sources));
        Assert.NotEqual(baseline,
            IonAccountingStore.SettingsKeyFor("10 ppm", "10 ppm", "other", new[] { "L" }, sources));
        Assert.NotEqual(baseline,
            IonAccountingStore.SettingsKeyFor("10 ppm", "10 ppm", "scheme", new[] { "L", "M" }, sources));
        Assert.Equal(baseline,
            IonAccountingStore.SettingsKeyFor("10 ppm", "10 ppm", "scheme", new[] { "L" }, sources));
    }

    /// <summary>
    /// A cache keyed for these settings is not necessarily COMPLETE for them, and the difference
    /// matters because nothing fails loudly.
    /// </summary>
    /// <remarks>
    /// The key covers the instrument files that could be measured, not the ones that were - so a
    /// <c>--max</c> spot check writes a whole-cohort key over a handful of rows. Trusting the key
    /// alone then makes a later full run return the short cache and never measure the rest, and the
    /// only symptom is a plot with too few bars. So the reader's job is to say which replicates are
    /// covered, and the run measures the remainder.
    /// </remarks>
    [Fact]
    public void APartialCacheIsDetectableByCoverageNotByItsKey()
    {
        var dir = TempDir();
        try
        {
            // What a "--max 2" run over a four-replicate cohort leaves behind.
            IonAccountingStore.Write(dir, new IonAccountingResult(
                "whole-cohort-key", "t", "p", "s", Array.Empty<string>(), 10, true,
                new[] { Row(100, 100, 10, 10, "s1"), Row(100, 100, 20, 20, "s2") },
                new[]
                {
                    new IonCycleRow("s1", 0, 0, 1, 1, 167, 100, 100, 10, 10),
                    new IonCycleRow("s2", 0, 0, 1, 1, 167, 100, 100, 20, 20),
                }));

            var cached = IonAccountingStore.Read(dir);
            Assert.NotNull(cached);

            // The key matches the cohort's settings, so a key check alone accepts it...
            Assert.True(cached!.MatchesSettings("whole-cohort-key"));

            // ...while the coverage check - which is what the run actually uses - does not.
            var wanted = new[] { "s1", "s2", "s3", "s4" };
            var covered = cached.Rows.Where(r => r.IsUsable).Select(r => r.Sample).ToHashSet();
            Assert.Equal(new[] { "s3", "s4" }, wanted.Where(s => !covered.Contains(s)));

            // And the reusable replicates' traces are readable, so they need not be re-measured.
            Assert.Single(IonAccountingStore.ReadCycles(dir, "s1"));
            Assert.Single(IonAccountingStore.ReadCycles(dir, "s2"));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// A row that failed to read is NOT coverage - it must be retried, not treated as done.
    /// </summary>
    [Fact]
    public void AFailedReplicateDoesNotCountAsCovered()
    {
        var failed = new IonAccountingRow(
            "s1", "experimental", "", Ms2ReadStatus.NotFound, "none", 0, 0,
            0, 0, 0, 0, 0, false, double.NaN, double.NaN, 0, 0, 0, 0,
            Array.Empty<double>(), Array.Empty<double>());

        Assert.False(failed.IsUsable);
        Assert.True(Row(100, 100, 10, 10, "s2").IsUsable);
    }

    // ---------------------------------------------------------------- the claim loader

    /// <summary>
    /// One pass over <c>merged_data/</c> has to produce BOTH levels, placed correctly: precursor rows
    /// into the single MS1 lane, fragment rows into their precursor's isolation window, each over the
    /// document's own extraction window. This exercises the SQL as well as the classification - the
    /// precursor flag is selected as a column rather than filtered on, which is what makes it one
    /// pass, and a wrong cast there would silently put every row at one level.
    /// </summary>
    [Fact]
    public void BothMsLevelsComeOffOnePassPlacedInTheRightLanes()
    {
        var dir = TempDir();
        try
        {
            var parquet = Path.Combine(dir, "merged.parquet");
            WriteMergedFixture(parquet, new[]
            {
                // sample, peptide, fragment, precursorMz, productMz, rt0, rt1
                ("r1", "PEPTIDEK", "precursor", 402.0, 402.0, 10.0, 11.0),
                ("r1", "PEPTIDEK", "precursor [M+1]", 402.0, 402.5, 10.0, 11.0),
                ("r1", "PEPTIDEK", "y5", 402.0, 600.0, 10.0, 11.0),
                ("r1", "PEPTIDEK", "y6", 402.0, 700.0, 10.0, 11.0),
                // A different precursor, in the second isolation window.
                ("r1", "OTHERPEPK", "y4", 406.0, 500.0, 20.0, 21.0),
                // A peptide the run dropped: claims nothing.
                ("r1", "DROPPEDK", "y3", 402.0, 650.0, 10.0, 11.0),
                // Another replicate, which must not leak into r1's claims.
                ("r2", "PEPTIDEK", "y5", 402.0, 600.0, 10.0, 11.0),
            });

            var dataset = MergedDataset.Open(parquet);
            var cols = SignalColumns.Resolve(
                ParquetTable.ReadColumnNames(parquet).ToList());
            Assert.NotNull(cols);

            var scheme = IsolationScheme.Cycle("c", start: 400, step: 4, width: 4, count: 2);
            var tolerance = ProductMassTolerance.ParseSetting("10 ppm");
            Assert.NotNull(tolerance);

            var classes = new Dictionary<string, PeptideClass>(StringComparer.Ordinal)
            {
                ["PEPTIDEK"] = new(true, 0b01),
                ["OTHERPEPK"] = new(true, 0b10),
                ["DROPPEDK"] = new(false, 0),
            };

            var loaded = ClaimedRegionLoader.ForReplicate(
                dataset, cols!, "r1", scheme, tolerance, tolerance, classes);

            // Six r1 rows: two precursor, four fragment. r2's row is filtered out by the sample.
            Assert.Equal(2, loaded.Ms1Rows);
            Assert.Equal(4, loaded.Ms2Rows);
            Assert.Equal(1, loaded.Unassigned);       // DROPPEDK's fragment
            Assert.Equal(0, loaded.OutsideScheme);
            Assert.Equal(0, loaded.NoGeometry);

            // Five claims: 2 MS1 + 3 assigned fragments.
            Assert.Equal(5, loaded.Regions.Count);

            var ms1 = loaded.Regions.Where(r => r.MsLevel == 1).ToArray();
            Assert.Equal(2, ms1.Length);
            // MS1 has no isolation window - a survey scan measures the whole range at once.
            Assert.All(ms1, r => Assert.Equal(ClaimedSignalIndex.AnyWindow, r.WindowIndex));

            var ms2 = loaded.Regions.Where(r => r.MsLevel == 2).ToArray();
            Assert.Equal(3, ms2.Length);
            // 402 is in window 0, 406 in window 1.
            Assert.Equal(2, ms2.Count(r => r.WindowIndex == 0));
            Assert.Equal(1, ms2.Count(r => r.WindowIndex == 1));
            // And the list bits came from the caller's identity map, not from the table.
            Assert.Equal(0b10u, ms2.Single(r => r.WindowIndex == 1).ListMask);

            // The extraction window is the document's: +/-10 ppm at m/z 600 is +/-0.006.
            var y5 = ms2.Single(r => Math.Abs((r.MzLow + r.MzHigh) / 2 - 600) < 0.01);
            Assert.Equal(600 - 0.006, y5.MzLow, 6);
            Assert.Equal(600 + 0.006, y5.MzHigh, 6);
            Assert.Equal(10.0, y5.RtStart, 6);
            Assert.Equal(11.0, y5.RtStop, 6);
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// With no precursor tolerance the MS1 half is dropped rather than guessed: a guessed window
    /// changes how much sharing is found, with nothing on the plot to say the number moved.
    /// </summary>
    [Fact]
    public void WithNoPrecursorToleranceOnlyTheMs2HalfIsClaimed()
    {
        var dir = TempDir();
        try
        {
            var parquet = Path.Combine(dir, "merged.parquet");
            WriteMergedFixture(parquet, new[]
            {
                ("r1", "PEPTIDEK", "precursor", 402.0, 402.0, 10.0, 11.0),
                ("r1", "PEPTIDEK", "y5", 402.0, 600.0, 10.0, 11.0),
            });

            var loaded = ClaimedRegionLoader.ForReplicate(
                MergedDataset.Open(parquet),
                SignalColumns.Resolve(ParquetTable.ReadColumnNames(parquet).ToList())!,
                "r1",
                IsolationScheme.Cycle("c", 400, 4, 4, 2),
                ProductMassTolerance.ParseSetting("10 ppm"),
                precursorTolerance: null,
                new Dictionary<string, PeptideClass>(StringComparer.Ordinal)
                {
                    ["PEPTIDEK"] = new(true, 0),
                });

            Assert.Single(loaded.Regions);
            Assert.Equal(2, loaded.Regions[0].MsLevel);
            Assert.Equal(1, loaded.NoGeometry);   // the precursor row, with no window to place it in
        }
        finally
        {
            Cleanup(dir);
        }
    }

    // ---------------------------------------------------------------- helpers

    private static IonAccountingRow Row(
        double ms1Acquired, double ms2Acquired, double ms1Assigned, double ms2Assigned,
        string sample = "s", double[]? ms1ByList = null, double[]? ms2ByList = null,
        double ms2Explained = 0) =>
        new(sample, "experimental", sample + ".raw", Ms2ReadStatus.Ok, "test",
            1, 167, ms1Acquired, ms2Acquired, ms1Assigned, ms2Assigned,
            ms2Explained, ms2Explained > 0,
            0, 60, 1000, 0, 0, 1,
            ms1ByList ?? Array.Empty<double>(), ms2ByList ?? Array.Empty<double>());

    private static IonAccountingResult Result(IReadOnlyList<IonAccountingRow> rows) =>
        new("k", "t", "p", "s", Array.Empty<string>(), 1, true, rows, Array.Empty<IonCycleRow>());

    /// <summary>
    /// A one-file merged table with the columns the accounting resolves.
    /// <see cref="MergedDataset.Open"/> accepts a single parquet as well as a partitioned directory,
    /// so a fixture does not need the hive layout.
    /// </summary>
    private static void WriteMergedFixture(
        string path,
        IReadOnlyList<(string Sample, string Peptide, string Fragment,
            double PrecursorMz, double ProductMz, double Start, double End)> rows)
    {
        var values = string.Join(
            " UNION ALL ",
            rows.Select(r => "SELECT "
                + $"'{r.Sample}' AS \"Sample ID\", "
                + $"'{r.Peptide}' AS \"Peptide Modified Sequence\", "
                + $"'{r.Fragment}' AS \"Fragment Ion\", "
                + $"1.0 AS \"Area\", "
                + $"{Num(r.PrecursorMz)} AS \"Precursor Mz\", "
                + $"{Num(r.ProductMz)} AS \"Product Mz\", "
                + $"{Num(r.Start)} AS \"Start Time\", "
                + $"{Num(r.End)} AS \"End Time\""));

        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"COPY ({values}) TO '{path.Replace('\\', '/')}' (FORMAT PARQUET)";
        cmd.ExecuteNonQuery();
    }

    private static string Num(double value) =>
        value.ToString("R", CultureInfo.InvariantCulture);

    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_ions_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void Cleanup(string dir)
    {
        try
        {
            Directory.Delete(dir, recursive: true);
        }
        catch (IOException)
        {
            // A temp directory that will not delete is not a test failure.
        }
    }
}
