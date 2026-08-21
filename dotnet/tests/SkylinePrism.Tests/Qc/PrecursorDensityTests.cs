using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The (isolation window x RT) precursor-density map behind the tool's Spectrum density tab: the
/// binning arithmetic, and the query that pulls one run's detected precursors out of a merged report.
/// </summary>
public class PrecursorDensityTests
{
    private static string MergedGolden => Fixtures.Path2("mini", "merge", "merged_data.parquet");

    [Fact]
    public void Bin_CountsEveryRtBinThePeakSpans()
    {
        // One precursor at m/z 500.4, eluting 10.0 -> 10.25 min. With 0.1 min bins the peak covers
        // bins 0, 1 and 2 (the last one partially - a spectrum acquired then still sees the peptide).
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(500.4, 10.0, 10.25) }, mzBinTh: 2.0, rtBinMin: 0.1);

        Assert.Equal(500.0, map.MzLow);   // floored to a whole Th
        Assert.Equal(10.0, map.RtLow);
        Assert.Equal(1, map.MzBins);
        Assert.Equal(3, map.RtBins);
        Assert.Equal(new[] { 1, 1, 1 }, Row(map, 0));
        Assert.Equal(1, map.MaxCount);
    }

    [Fact]
    public void Bin_SeparatesPrecursorsByIsolationWindow()
    {
        // Two precursors 6 Th apart, co-eluting. At a 2 Th window they land in different rows (different
        // DIA spectra); at a 10 Th window the same spectrum carries both.
        var precursors = new[]
        {
            new DetectedPrecursor(500.0, 5.0, 5.2),
            new DetectedPrecursor(506.0, 5.0, 5.2),
        };

        var narrow = PrecursorDensity.Bin(precursors, mzBinTh: 2.0, rtBinMin: 0.1);
        Assert.Equal(3, narrow.MzBins);  // 500 -> 506 spans 3 windows of 2 Th
        Assert.Equal(1, narrow.MaxCount); // one precursor each in the bottom and top window

        var wide = PrecursorDensity.Bin(precursors, mzBinTh: 10.0, rtBinMin: 0.1);
        Assert.Equal(1, wide.MzBins);
        Assert.Equal(2, wide.MaxCount); // both precursors in one window = one crowded spectrum
    }

    [Fact]
    public void Bin_CoElutingPrecursorsStackInTheSameCell()
    {
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(700.1, 20.0, 20.1),
                new DetectedPrecursor(700.9, 20.05, 20.3), // same 2 Th window, overlapping peak
                new DetectedPrecursor(700.5, 25.0, 25.1),  // same window, well after the others
            },
            mzBinTh: 2.0, rtBinMin: 0.1);

        Assert.Equal(1, map.MzBins);
        Assert.Equal(2, map.MaxCount);              // the two overlapping peaks, not the third
        Assert.Equal(2, Row(map, 0).First());       // first RT bin holds both
        Assert.Equal(0, Row(map, 0)[10]);           // the quiet stretch between them
    }

    [Fact]
    public void Bin_EmptyInputGivesEmptyMap()
    {
        var map = PrecursorDensity.Bin(Array.Empty<DetectedPrecursor>());
        Assert.True(map.IsEmpty);
        Assert.Equal(0, map.MaxCount);
    }

    [Fact]
    public void Bin_RejectsNonPositiveBins()
    {
        var one = new[] { new DetectedPrecursor(500, 1, 2) };
        Assert.Throws<ArgumentOutOfRangeException>(() => PrecursorDensity.Bin(one, mzBinTh: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PrecursorDensity.Bin(one, rtBinMin: -1));
    }

    [Fact]
    public void Bin_WidensBinsTooFineToRender()
    {
        // A 1e-6 Th bin over a 500 Th range would be half a billion rows. The grid stays bounded and the
        // map reports the bin size it actually used, so the plot never mislabels its own axes.
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(400, 0, 60), new DetectedPrecursor(900, 0, 60) },
            mzBinTh: 1e-6, rtBinMin: 1e-6);

        Assert.True(map.MzBins <= 4000, $"m/z bins: {map.MzBins}");
        Assert.True(map.RtBins <= 4000, $"RT bins: {map.RtBins}");
        Assert.True(map.Rows[0].Width > 1e-6);
        Assert.True(map.RtBinMin > 1e-6);
    }

    [Fact]
    public void Bin_OnRealWindows_UsesTheSchemesOwnEdges()
    {
        // A scheme starting at 400 with 25 Th windows. A precursor at 412 belongs to window 0 (400-425),
        // NOT to a bin whose edges happen to fall where a uniform grid over the observed data put them.
        var scheme = Scheme(("s", 400, 425), ("s", 425, 450), ("s", 450, 475));
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(412.0, 10.0, 10.2),
                new DetectedPrecursor(424.9, 10.0, 10.2), // same window as 412 despite being 13 Th away
                new DetectedPrecursor(425.1, 10.0, 10.2), // just over the edge -> next window
            },
            scheme, rtBinMin: 0.1);

        Assert.Equal(3, map.MzBins);            // the scheme's windows, not the data's range
        Assert.Equal(400, map.MzLow);
        Assert.Equal(475, map.MzHigh);
        Assert.Equal(2, map.Counts[0, 0]);      // 412 + 424.9
        Assert.Equal(1, map.Counts[1, 0]);      // 425.1
        Assert.Equal(0, map.Counts[2, 0]);      // an acquired-but-empty window stays in the map
        Assert.Equal(0, map.PrecursorsOutsideRows);
        Assert.Equal("s", map.RowSource);
    }

    [Fact]
    public void Bin_OnRealWindows_CountsPrecursorsOutsideEveryWindow()
    {
        // The wrong scheme for the data: precursors below its range are reported, never clamped into the
        // nearest window, because that count is how the user sees the scheme does not fit.
        var scheme = Scheme(("s", 600, 625));
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(610.0, 5.0, 5.1),
                new DetectedPrecursor(450.0, 5.0, 5.1),
                new DetectedPrecursor(900.0, 5.0, 5.1),
            },
            scheme, rtBinMin: 0.1);

        Assert.Equal(1, map.MaxCount);
        Assert.Equal(2, map.PrecursorsOutsideRows);
    }

    [Fact]
    public void Bin_OnRealWindows_CountsOverlappingWindowsSeparately()
    {
        // Staggered/overlapping DIA: a precursor in the overlap really was fragmented in both windows.
        var scheme = Scheme(("stagger", 500, 520), ("stagger", 510, 530));
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(515.0, 1.0, 1.05) }, scheme, rtBinMin: 0.1);

        Assert.Equal(1, map.Counts[0, 0]);
        Assert.Equal(1, map.Counts[1, 0]);
        Assert.Equal(0, map.PrecursorsOutsideRows);
    }

    [Fact]
    public void Bin_OnRealWindows_RejectsASchemeWithNoWindows()
    {
        // "Results only" is a named scheme with no windows - it cannot be binned on, and must not be
        // silently treated as an empty grid.
        var resultsOnly = new IsolationScheme(IsolationScheme.ResultsOnlyName, Array.Empty<IsolationWindow>());
        Assert.Throws<ArgumentException>(() => PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(500, 1, 2) }, resultsOnly));
    }

    [Fact]
    public void ToDisplayGrid_PreservesVariableWindowWidths()
    {
        // A variable-width scheme: narrow at low m/z, wide at high m/z. The uniform display grid must put
        // the counts at the right m/z, so the wide window occupies proportionally more rows.
        var scheme = Scheme(("vw", 400, 410), ("vw", 410, 500));
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(405.0, 1.0, 1.05),
                new DetectedPrecursor(450.0, 1.0, 1.05),
                new DetectedPrecursor(460.0, 1.0, 1.05),
            },
            scheme, rtBinMin: 0.1);

        var grid = map.ToDisplayGrid(100); // 100 rows over 400-500 -> 1 Th each
        // Row 0 is the TOP (high m/z) = the wide window, which holds 2 precursors.
        Assert.Equal(2, grid[0, 0]);
        // The bottom 10 rows are the narrow 400-410 window, which holds 1.
        Assert.Equal(1, grid[99, 0]);
        // The wide window covers 90% of the m/z range, so ~90 of the 100 display rows show its count.
        var wideRows = 0;
        for (var r = 0; r < 100; r++)
            if (grid[r, 0] == 2)
                wideRows++;
        Assert.InRange(wideRows, 88, 92);
    }

    [Fact]
    public void Coverage_DistinguishesTheRightSchemeFromTheWrongOne()
    {
        var mz = new[] { 405.0, 430.0, 455.0, 470.0 };
        Assert.Equal(1.0, Scheme(("right", 400, 425), ("right", 425, 450), ("right", 450, 475)).Coverage(mz));
        Assert.Equal(0.0, Scheme(("wrong", 700, 725)).Coverage(mz));
    }

    private static IsolationScheme Scheme(params (string Name, double Start, double End)[] windows) =>
        new(windows[0].Name, windows.Select(w => new IsolationWindow(w.Start, w.End)).ToList());

    [Fact]
    public void Resolve_FindsColumnsInBothExportSpellings()
    {
        var csvStyle = PrecursorDensity.Resolve(new[]
        {
            "Sample ID", "Replicate Name", "Peptide Modified Sequence Unimod Ids", "Precursor Charge",
            "Precursor Mz", "Start Time", "End Time", "Detection Q Value",
        });
        Assert.NotNull(csvStyle);
        Assert.Equal("Sample ID", csvStyle!.Sample); // the batch-disambiguated column wins
        Assert.Equal("Detection Q Value", csvStyle.DetectionQValue);

        // Parquet/invariant spelling: no spaces. FindColumn normalizes case, spaces and underscores.
        var parquetStyle = PrecursorDensity.Resolve(new[]
        {
            "ReplicateName", "PeptideModifiedSequenceUnimodIds", "PrecursorCharge", "PrecursorMz",
            "StartTime", "EndTime",
        });
        Assert.NotNull(parquetStyle);
        Assert.Equal("ReplicateName", parquetStyle!.Sample);
        Assert.Null(parquetStyle.DetectionQValue); // optional - the q-value filter is simply unavailable
    }

    [Fact]
    public void Resolve_ReturnsNullWhenBoundariesAreMissing()
    {
        // A report exported without the peak-boundary columns cannot answer this question at all.
        Assert.Null(PrecursorDensity.Resolve(new[]
        {
            "Replicate Name", "Peptide Modified Sequence", "Precursor Charge", "Precursor Mz", "Area",
        }));
    }

    [Fact]
    public void Load_ReadsOneRowPerPrecursorFromTheMergedReport()
    {
        Assert.True(File.Exists(MergedGolden), $"golden fixture missing: {MergedGolden}");
        var cols = PrecursorDensity.Resolve(ParquetTable.ReadColumnNames(MergedGolden).ToHashSet());
        Assert.NotNull(cols);

        var samples = MergedParquetReader.GetSortedSamples(MergedDataset.Open(MergedGolden), cols!.Sample);
        Assert.NotEmpty(samples);

        var precursors = PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols, samples[0], qValueCutoff: null);
        Assert.NotEmpty(precursors);
        Assert.All(precursors, p =>
        {
            Assert.True(p.Mz > 0, "precursor m/z should be positive");
            Assert.True(p.RtStop >= p.RtStart, "peak must not end before it starts");
            Assert.True(double.IsFinite(p.RtStart) && double.IsFinite(p.RtStop));
        });

        // The report is transition-level; the map is precursor-level, so it must be the smaller of the two.
        var transitionRows = CountRows(MergedGolden, cols.Sample, samples[0]);
        Assert.True(precursors.Count < transitionRows,
            $"{precursors.Count} precursors from {transitionRows} transition rows");

        // And it bins into a usable map.
        var map = PrecursorDensity.Bin(precursors);
        Assert.False(map.IsEmpty);
        Assert.True(map.MaxCount >= 1);
        Assert.True(map.MzHigh > map.MzLow && map.RtHigh > map.RtLow);
    }

    [Fact]
    public void Load_QValueCutoffOnlyKeepsConfidentDetections()
    {
        var cols = PrecursorDensity.Resolve(ParquetTable.ReadColumnNames(MergedGolden).ToHashSet());
        Assert.NotNull(cols);
        Assert.NotNull(cols!.DetectionQValue);
        var sample = MergedParquetReader.GetSortedSamples(MergedDataset.Open(MergedGolden), cols.Sample)[0];

        var all = PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols, sample, qValueCutoff: null);
        var confident = PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols, sample, qValueCutoff: 0.01);
        var none = PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols, sample, qValueCutoff: -1);

        Assert.True(confident.Count <= all.Count);
        Assert.Empty(none); // no q-value is below -1, so nothing counts as detected
    }

    [Fact]
    public void Load_UnknownSampleGivesNoPrecursors()
    {
        var cols = PrecursorDensity.Resolve(ParquetTable.ReadColumnNames(MergedGolden).ToHashSet());
        Assert.Empty(PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols!, "no such replicate'; --", null));
    }

    private static int[] Row(PrecursorDensityMap map, int row)
    {
        var result = new int[map.RtBins];
        for (var j = 0; j < map.RtBins; j++)
            result[j] = map.Counts[row, j];
        return result;
    }

    private static long CountRows(string parquet, string sampleCol, string sample)
    {
        using var conn = new DuckDB.NET.Data.DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT COUNT(*) FROM read_parquet('{parquet.Replace("'", "''")}') " +
            $"WHERE \"{sampleCol}\" = '{sample.Replace("'", "''")}'";
        return Convert.ToInt64(cmd.ExecuteScalar());
    }
}
