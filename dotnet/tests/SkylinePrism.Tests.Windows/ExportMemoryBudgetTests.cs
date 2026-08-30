using System;
using System.Collections.Generic;
using System.IO;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// How many headless Skyline exports PRISM runs at once. Each one loads a whole document into a separate
/// Skyline process, so the budget has to track DOCUMENT SIZE - and for a long time it did not.
///
/// <para>The constant it used, 2.5 GB, came from a benchmark that varied concurrency while holding the
/// document fixed: a 5 MB .sky with a 116 MB .skyd peaked at 1.26 GB. At that size the figure is almost
/// entirely Skyline's fixed baseline, so the benchmark could not see the term that dominates on a large
/// document. Measured on a 2-plate cohort of 11.3 GB documents: 22.4 GB and 17.2 GB resident. PRISM chose
/// two concurrent exports on a 64 GB machine, ran it down to 4.5 GB free, and one Skyline stopped
/// accumulating CPU entirely - and did NOT resume when the other finished and returned 22 GB. Starving a
/// Skyline is permanent, so under-budgeting does not cost time, it costs the export.</para>
/// </summary>
public class ExportMemoryBudgetTests
{
    private const long Gb = 1024L * 1024 * 1024;

    /// <summary>
    /// The many-small-documents case - the one the 100-document parallelization was built for - must be
    /// completely unchanged by making the budget size-aware. A 100 MB document is all baseline, so the
    /// floor still governs and the chosen concurrency is what it always was.
    /// </summary>
    [Fact]
    public void ASmallDocumentStillGetsTheFlatFloor()
    {
        Assert.Equal(MainWindow.GbPerConcurrentExport, MainWindow.PerExportGbForBytes(100L * 1024 * 1024));
        Assert.Equal(MainWindow.GbPerConcurrentExport, MainWindow.PerExportGbForBytes(5L * 1024 * 1024));

        // 32 GB and 100 small plates: 7 fit, so the cap of 4 governs - exactly as before this change.
        Assert.Equal(7, MainWindow.ExportsThatFit(32, MainWindow.GbPerConcurrentExport));
    }

    /// <summary>
    /// The regime the constant could not see. An 11.3 GB document is budgeted from its own size, and the
    /// answer on a 64 GB machine is ONE - which is what actually completes.
    /// </summary>
    [Fact]
    public void ALargeDocumentIsBudgetedFromItsSize()
    {
        var budget = MainWindow.PerExportGbForBytes((long)(11.34 * Gb));

        Assert.Equal(22.68, budget, 1);                      // ~2x the .sky, matching the 17-22 GB measured
        Assert.Equal(1, MainWindow.ExportsThatFit(64, budget));
    }

    /// <summary>
    /// Zero is the answer that has to survive to the caller. On 32 GB this cohort does not fit even once;
    /// the old code computed 7 from the flat constant, clamped to 2, and starved a Skyline. Returning 0
    /// is what lets ExportParallelism warn instead of proceeding silently.
    /// </summary>
    [Fact]
    public void ZeroIsReportedWhenNotEvenOneExportFits()
    {
        var budget = MainWindow.PerExportGbForBytes((long)(11.34 * Gb));

        Assert.Equal(0, MainWindow.ExportsThatFit(32, budget));
        Assert.Equal(0, MainWindow.ExportsThatFit(16, budget));

        // What the OLD flat constant would have said on the same machine, and why it went wrong.
        Assert.Equal(7, MainWindow.ExportsThatFit(32, MainWindow.GbPerConcurrentExport));
    }

    /// <summary>
    /// Sized off the LARGEST input, never the mean: one big document among many small ones still needs
    /// its own headroom, and an average would hide it completely.
    /// </summary>
    [Fact]
    public void TheBudgetFollowsTheLargestDocumentNotTheAverage()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_budget_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        var small = Path.Combine(dir, "small.sky");
        var big = Path.Combine(dir, "big.sky");
        File.WriteAllBytes(small, new byte[1024]);
        File.WriteAllBytes(big, new byte[64 * 1024]);

        var inputs = new List<PrismInput>
        {
            PrismInput.FromClosedDocument(small),
            PrismInput.FromClosedDocument(big),
        };

        MainWindow.PerExportGb(inputs, out var largest, out var largestGb);

        Assert.Equal("big", largest);
        Assert.Equal(64 * 1024 / (double)Gb, largestGb, 9);
    }

    /// <summary>
    /// A pre-exported report is not opened in Skyline at all, so it must not drive the budget - otherwise
    /// a large report file would shrink concurrency for documents that cost nothing to load.
    /// </summary>
    [Fact]
    public void APreExportedReportDoesNotCountTowardTheBudget()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_budget_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        var report = Path.Combine(dir, "already-exported.parquet");
        File.WriteAllBytes(report, new byte[256 * 1024]);

        var inputs = new List<PrismInput> { PrismInput.FromReportFile(report) };

        var budget = MainWindow.PerExportGb(inputs, out var largest, out var largestGb);

        Assert.Equal(0, largestGb);
        Assert.Equal("", largest);
        Assert.Equal(MainWindow.GbPerConcurrentExport, budget);
    }
}
