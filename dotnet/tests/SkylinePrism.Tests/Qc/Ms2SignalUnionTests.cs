using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;
using Region = SkylinePrism.Core.Qc.Ms2SignalUnion.Region;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Counting MS2 signal ONCE when several peptides extract it.
///
/// <para>The whole point of the type: summing transition areas double-counts, because a DIA window
/// co-isolates tens of peptides and two fragments within one extraction tolerance are the same
/// detector counts. Every test here is a statement about when two transitions are, or are not, the
/// same signal.</para>
///
/// <para>Tolerance throughout is centroided 10 ppm — the setting the real documents use — so at m/z
/// 500 Skyline extracts ±0.005, and two targets 0.004 apart have overlapping extraction windows
/// while 0.05 apart do not.</para>
/// </summary>
public class Ms2SignalUnionTests
{
    private static readonly ProductMassTolerance Ppm10 =
        ProductMassTolerance.Parse("centroided", "10")!;

    private static Region At(
        double mz, double rtStart, double rtStop, double area,
        int window = 0, bool assigned = true, uint lists = 0) =>
        new(window, mz, rtStart, rtStop, area, assigned, lists);

    private static Ms2SignalUnion.Result Compute(IEnumerable<Region> regions, int listCount = 0) =>
        Ms2SignalUnion.Compute(regions.ToList(), Ppm10, listCount);

    /// <summary>The correction itself: co-eluting fragments at the same mass count once.</summary>
    [Fact]
    public void SharedSignalIsCountedOnce()
    {
        var r = Compute(new[]
        {
            At(500.000, 10.0, 10.4, 100),
            At(500.004, 10.1, 10.5, 120),   // within 10 ppm, overlapping in RT: the same counts
        });

        Assert.Equal(120, r.AssignedArea, 9);    // max, not 220
        Assert.Equal(220, r.SummedArea, 9);      // what a naive sum would have claimed
        Assert.Equal(1, r.MergedGroups);
        Assert.Equal(2, r.LargestGroup);
    }

    /// <summary>
    /// Separated in retention time, the same fragment mass is different signal: both peptides extract
    /// the same chromatogram but integrate different parts of it. This is why RT is in the test.
    /// </summary>
    [Fact]
    public void SameMassAtDifferentTimesIsNotShared()
    {
        var r = Compute(new[]
        {
            At(500.000, 10.0, 10.4, 100),
            At(500.004, 20.0, 20.4, 120),
        });

        Assert.Equal(220, r.AssignedArea, 9);
        Assert.Equal(2, r.MergedGroups);
        Assert.Equal(1, r.LargestGroup);
    }

    /// <summary>Different isolation windows are different spectra; they never share.</summary>
    [Fact]
    public void DifferentIsolationWindowsNeverShare()
    {
        var r = Compute(new[]
        {
            At(500.000, 10.0, 10.4, 100, window: 0),
            At(500.004, 10.1, 10.5, 120, window: 1),
        });

        Assert.Equal(220, r.AssignedArea, 9);
        Assert.Equal(2, r.MergedGroups);
    }

    /// <summary>Outside the extraction tolerance, two masses are resolved and both count.</summary>
    [Fact]
    public void MassesOutsideTheToleranceBothCount()
    {
        var r = Compute(new[]
        {
            At(500.00, 10.0, 10.4, 100),
            At(500.05, 10.1, 10.5, 120),   // 100 ppm apart, far outside a 10 ppm window
        });

        Assert.Equal(220, r.AssignedArea, 9);
        Assert.Equal(2, r.MergedGroups);
    }

    /// <summary>
    /// Spans that merely touch do not overlap - a peak ending exactly where the next begins shares no
    /// scan. Without the strict inequality, adjacent elutions would silently collapse into one.
    /// </summary>
    [Fact]
    public void TouchingRtSpansAreNotShared()
    {
        var r = Compute(new[]
        {
            At(500.000, 10.0, 10.4, 100),
            At(500.004, 10.4, 10.8, 120),
        });

        Assert.Equal(220, r.AssignedArea, 9);
        Assert.Equal(2, r.MergedGroups);
    }

    /// <summary>
    /// A list is credited with a shared region at its full magnitude, and two lists claiming the same
    /// region BOTH get it. The totals nest rather than partition: the question is what portion of the
    /// signal a panel accounts for, not who owns the peptide.
    /// </summary>
    [Fact]
    public void ListsNestInsideAssignedAndMayOverlapEachOther()
    {
        var r = Compute(new[]
        {
            At(500.000, 10.0, 10.4, 100, lists: 0b01),   // list 0 claims it
            At(500.004, 10.1, 10.5, 120, lists: 0b10),   // list 1 claims the same signal
            At(600.000, 30.0, 30.4, 50),                 // assigned, in neither list
        }, listCount: 2);

        Assert.Equal(170, r.AssignedArea, 9);            // 120 (shared, once) + 50
        Assert.Equal(120, r.ListArea[0], 9);             // the whole shared region
        Assert.Equal(120, r.ListArea[1], 9);             // ...and the same region again, for list 1
        Assert.All(r.ListArea, a => Assert.True(a <= r.AssignedArea));
    }

    /// <summary>Signal from peptides that never reached the peptide matrix is not assigned signal.</summary>
    [Fact]
    public void UnassignedRegionsCountTowardsNothing()
    {
        var r = Compute(new[]
        {
            At(500.0, 10.0, 10.4, 100, assigned: false, lists: 0b1),
            At(600.0, 30.0, 30.4, 50),
        }, listCount: 1);

        Assert.Equal(50, r.AssignedArea, 9);
        Assert.Equal(0, r.ListArea[0], 9);
        Assert.Equal(50, r.SummedArea, 9);
    }

    /// <summary>
    /// A whole chain of near-identical masses collapses to one region, and the group size is reported
    /// so heavy sharing is visible instead of just showing up as a small total.
    /// </summary>
    [Fact]
    public void ManyCoElutingFragmentsCollapseToOneRegion()
    {
        var regions = Enumerable.Range(0, 10)
            .Select(k => At(500.000 + k * 0.002, 10.0, 10.6, 100 + k))
            .ToList();

        var r = Compute(regions);

        Assert.Equal(1, r.MergedGroups);
        Assert.Equal(10, r.LargestGroup);
        Assert.Equal(109, r.AssignedArea, 9);      // the largest of them, once
        Assert.Equal(1045, r.SummedArea, 9);       // what summing would have claimed
    }

    /// <summary>
    /// Unintegrated peaks arrive as #N/A and become non-finite. They are counted, never silently
    /// dropped - a run full of them should be visible in the diagnostics.
    /// </summary>
    [Fact]
    public void NonFiniteGeometryIsSkippedAndCounted()
    {
        var r = Compute(new[]
        {
            At(double.NaN, 10.0, 10.4, 100),
            At(500.0, double.NaN, 10.4, 100),
            At(500.0, 10.0, 10.4, double.NaN),
            At(500.0, 10.6, 10.2, 100),           // stop before start
            At(600.0, 30.0, 30.4, 50),
        });

        Assert.Equal(4, r.Skipped);
        Assert.Equal(50, r.AssignedArea, 9);
        Assert.Equal(1, r.Regions);
    }

    /// <summary>
    /// The answer must not depend on the order rows came back in. DuckDB runs with
    /// preserve_insertion_order off, so that order is arbitrary and would otherwise make the plot
    /// change between identical runs.
    /// </summary>
    [Fact]
    public void ResultDoesNotDependOnInputOrder()
    {
        var regions = new List<Region>
        {
            At(500.000, 10.0, 10.4, 100, lists: 0b1),
            At(500.004, 10.1, 10.5, 120),
            At(500.050, 10.0, 10.4, 70),
            At(600.000, 30.0, 30.4, 50, window: 1),
            At(600.003, 30.1, 30.9, 80, window: 1),
        };
        var forward = Ms2SignalUnion.Compute(regions, Ppm10, 1);

        var shuffled = new List<Region>(regions);
        shuffled.Reverse();
        var reversed = Ms2SignalUnion.Compute(shuffled, Ppm10, 1);

        Assert.Equal(forward.AssignedArea, reversed.AssignedArea, 12);
        Assert.Equal(forward.ListArea[0], reversed.ListArea[0], 12);
        Assert.Equal(forward.MergedGroups, reversed.MergedGroups);
        Assert.Equal(forward.LargestGroup, reversed.LargestGroup);
    }

    /// <summary>
    /// A wider extraction window can only merge more, never less. Masses 0.012 apart are separate at
    /// 10 ppm (±0.005, so the windows clear each other) and one chained region at 20 ppm (±0.01), which
    /// makes this discriminating rather than a tautology — the equivalent check on the committed
    /// cohort fixture cannot distinguish the two, because its collisions are exact-mass duplicates.
    /// </summary>
    [Fact]
    public void AWiderExtractionWindowMergesMoreNeverLess()
    {
        var regions = Enumerable.Range(0, 4)
            .Select(k => At(500.000 + k * 0.012, 10.0, 10.6, 100 + k))
            .ToList();

        var tight = Ms2SignalUnion.Compute(regions, ProductMassTolerance.Parse("centroided", "1")!, 0);
        var normal = Ms2SignalUnion.Compute(regions, Ppm10, 0);
        var wide = Ms2SignalUnion.Compute(regions, ProductMassTolerance.Parse("centroided", "20")!, 0);

        Assert.Equal(4, tight.MergedGroups);
        Assert.Equal(4, normal.MergedGroups);
        Assert.Equal(1, wide.MergedGroups);

        Assert.Equal(406, normal.AssignedArea, 9);   // 100+101+102+103, all separate
        Assert.Equal(103, wide.AssignedArea, 9);     // one region, the largest member

        Assert.True(normal.AssignedArea <= tight.AssignedArea + 1e-9);
        Assert.True(wide.AssignedArea < normal.AssignedArea);
    }

    /// <summary>
    /// The two area figures PARTITION what the union removed, exactly. Without this they are two
    /// plausible-looking numbers with no stated relationship to the totals beside them - and the whole
    /// reason they exist is that the ROW counts mislead about where the removed area came from.
    /// </summary>
    [Fact]
    public void RemovedAreaIsSplitBetweenTheTwoKindsOfCollisionAndNothingElse()
    {
        var regions = new List<Region>
        {
            // One peptide exported three times (a shared peptide, once per protein assignment).
            new(0, 500.000, 10.0, 10.4, 100, true, 0, PeptideId: 1),
            new(0, 500.000, 10.0, 10.4, 100, true, 0, PeptideId: 1),
            new(0, 500.000, 10.0, 10.4, 100, true, 0, PeptideId: 1),
            // Two different peptides sharing a fragment mass, co-eluting.
            new(0, 600.000, 20.0, 20.4, 80, true, 0, PeptideId: 2),
            new(0, 600.003, 20.1, 20.5, 30, true, 0, PeptideId: 3),
            // A region that shares with nothing, so it contributes to neither.
            new(0, 700.000, 30.0, 30.4, 55, true, 0, PeptideId: 4),
        };

        var r = Ms2SignalUnion.Compute(regions, Ppm10, 0);

        Assert.Equal(465, r.SummedArea, 9);        // 300 + 110 + 55
        Assert.Equal(235, r.AssignedArea, 9);      // 100 + 80 + 55
        Assert.Equal(200, r.DuplicateArea, 9);     // two of the three identical copies
        Assert.Equal(30, r.SharedArea, 9);         // the quieter of the co-isolated pair
        Assert.Equal(2, r.DuplicateRows);
        Assert.Equal(1, r.SharedAcrossPeptides);

        // The invariant: nothing removed is unaccounted for, and nothing is counted in both.
        Assert.Equal(r.SummedArea - r.AssignedArea, r.DuplicateArea + r.SharedArea, 9);
    }

    /// <summary>The union can never exceed the sum; that inequality is the whole correction.</summary>
    [Fact]
    public void UnionNeverExceedsTheNaiveSum()
    {
        var rng = new Random(20260901);
        var regions = Enumerable.Range(0, 500).Select(_ =>
        {
            var start = rng.NextDouble() * 40;
            return At(400 + rng.NextDouble() * 200, start, start + rng.NextDouble() * 0.5,
                rng.NextDouble() * 1000, window: rng.Next(0, 4));
        }).ToList();

        var r = Ms2SignalUnion.Compute(regions, Ppm10, 0);

        Assert.True(r.AssignedArea <= r.SummedArea + 1e-9,
            $"union {r.AssignedArea} exceeded the sum {r.SummedArea}");
        Assert.True(r.MergedGroups <= r.Regions);
    }

    /// <summary>Empty input is a zero, not a crash, and a list count beyond the bit budget is rejected.</summary>
    [Fact]
    public void EdgeCases()
    {
        var empty = Ms2SignalUnion.Compute(Array.Empty<Region>(), Ppm10, 2);
        Assert.Equal(0, empty.AssignedArea);
        Assert.Equal(2, empty.ListArea.Count);
        Assert.Equal(0, empty.MergedGroups);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => Ms2SignalUnion.Compute(Array.Empty<Region>(), Ppm10, Ms2SignalUnion.MaxLists + 1));
    }
}
