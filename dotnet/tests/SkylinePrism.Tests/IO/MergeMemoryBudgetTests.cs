using System;
using System.Runtime.InteropServices;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// The Stage 1 memory budget. DuckDB runs in-process and its buffer pool is native memory the GC
/// cannot see, so a budget written against TOTAL RAM on a machine that is already busy does not
/// spill - it pages, and the run looks like a hang with the system at 100% memory. These pin that
/// the budget is bounded by what is actually free, and that it can never come out unusably small.
/// </summary>
public class MergeMemoryBudgetTests
{
    private const long Mb = 1024L * 1024L;

    [Fact]
    public void Budget_NeverExceedsThreeQuartersOfTotal()
    {
        var totalMb = SystemMemory.TotalPhysicalBytes / Mb;

        var budget = DuckDbMerge.AutoMemoryBudgetMb();

        // The floor wins on a machine with very little RAM; above it, the total bound must hold.
        Assert.True(
            budget <= totalMb * 3 / 4 || budget == DuckDbMerge.MinMemoryBudgetMb,
            $"budget {budget} MB exceeds 75% of {totalMb} MB total");
    }

    [Fact]
    public void Budget_IsBoundedByFreeMemoryWhenThePlatformReportsIt()
    {
        var available = SystemMemory.AvailablePhysicalBytes();
        if (available is not > 0)
            return; // macOS and anything without a cheap probe: the total bound is all we have.

        var budget = DuckDbMerge.AutoMemoryBudgetMb();

        Assert.True(
            budget <= available.Value / Mb * 4 / 5 || budget == DuckDbMerge.MinMemoryBudgetMb,
            $"budget {budget} MB exceeds 80% of the {available.Value / Mb} MB free");
    }

    /// <summary>
    /// Below the floor DuckDB cannot even hold its reader buffers and fails outright rather than
    /// spilling, so a momentarily busy machine must still get a working merge.
    /// </summary>
    [Fact]
    public void Budget_NeverFallsBelowTheFloor()
        => Assert.True(DuckDbMerge.AutoMemoryBudgetMb() >= DuckDbMerge.MinMemoryBudgetMb);

    /// <summary>
    /// The probe has to return something real where we claim to support it. A wrong-but-plausible
    /// number here would silently mis-size every merge, so check it against the total rather than
    /// only for being positive.
    /// </summary>
    [Fact]
    public void AvailableMemory_IsReportedOnWindowsAndLinux()
    {
        var available = SystemMemory.AvailablePhysicalBytes();

        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            Assert.Null(available); // no probe implemented; callers must fall back, not guess
            return;
        }

        Assert.NotNull(available);
        Assert.True(available > 0, "free memory reported as zero");
        Assert.True(
            available <= SystemMemory.TotalPhysicalBytes,
            $"free ({available}) exceeds total ({SystemMemory.TotalPhysicalBytes})");
    }
}
