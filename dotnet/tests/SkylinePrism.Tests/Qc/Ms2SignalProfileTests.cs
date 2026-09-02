using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Core.Visualization;
using Xunit;
using MergedRegion = SkylinePrism.Core.Qc.Ms2SignalUnion.MergedRegion;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The retention-time profile behind Plot B. What matters is that the traces are COMPARABLE - the
/// same quantity on the same grid - because the whole point of the plot is reading one against
/// another, and that the assigned trace conserves the total it came from.
/// </summary>
public class Ms2SignalProfileTests
{
    private static readonly string[] NoLists = Array.Empty<string>();

    private static MergedRegion Region(
        double start, double stop, double area, bool assigned = true, uint lists = 0) =>
        new(start, stop, area, assigned, lists, 1);

    private static Ms2Cycle Cycle(int i, double start, double stop, double signal) =>
        new(i, start, stop, 10, signal);

    /// <summary>
    /// The assigned trace sums to the assigned total. Spreading a peak over its elution changes
    /// WHERE the signal is, never how much - so the plot cannot disagree with the bar in Plot A.
    /// </summary>
    [Fact]
    public void TheAssignedTraceConservesTheTotal()
    {
        var merged = new[]
        {
            Region(1.00, 1.40, 100),
            Region(2.10, 2.90, 250),   // spans several bins
            Region(5.00, 5.05, 70),    // narrower than a bin
        };

        var profile = Ms2SignalProfile.Build("S1", merged, null, NoLists, NoLists, binWidthMin: 0.25);

        Assert.Equal(420, profile.Assigned.Sum(), 6);
        Assert.False(profile.HasAcquired);
    }

    /// <summary>
    /// A peak is spread across the bins its span covers, in proportion to the overlap. Pinned with a
    /// span deliberately straddling a bin edge, because that is where an off-by-one would hide.
    /// </summary>
    [Fact]
    public void APeakIsSpreadOverItsElutionInProportionToOverlap()
    {
        // One region from 1.0 to 1.5 with 0.25 min bins starting at 1.0: two bins, half each.
        var profile = Ms2SignalProfile.Build(
            "S1", new[] { Region(1.0, 1.5, 100) }, null, NoLists, NoLists, binWidthMin: 0.25);

        Assert.Equal(2, profile.BinCount);
        Assert.Equal(50, profile.Assigned[0], 6);
        Assert.Equal(50, profile.Assigned[1], 6);

        // Three quarters in the first bin, one quarter in the second.
        var skewed = Ms2SignalProfile.Build(
            "S1", new[] { Region(1.0, 1.4, 100) }, null, NoLists, NoLists, binWidthMin: 0.30);
        Assert.Equal(100, skewed.Assigned.Sum(), 6);
        Assert.True(skewed.Assigned[0] > skewed.Assigned[1], "most of the peak is in the first bin");
    }

    /// <summary>
    /// A cycle is spread over its own span, by the same rule as a peak - and the acquired trace
    /// conserves its total.
    /// </summary>
    [Fact]
    public void AcquiredCyclesAreSpreadOverTheirSpan()
    {
        var cycles = new[]
        {
            Cycle(0, 1.00, 1.05, 1000),
            Cycle(1, 1.05, 1.10, 2000),
            Cycle(2, 1.30, 1.35, 4000),   // a later bin
        };

        var profile = Ms2SignalProfile.Build(
            "S1", Array.Empty<MergedRegion>(), cycles, NoLists, NoLists, binWidthMin: 0.25);

        Assert.True(profile.HasAcquired);
        Assert.Equal(7000, profile.Acquired.Sum(), 6);
        Assert.Equal(3000, profile.Acquired[0], 6);   // the first two cycles
        Assert.Equal(4000, profile.Acquired[1], 6);
    }

    /// <summary>
    /// Contiguous cycles of equal signal must give a FLAT acquired trace. Binning each cycle whole
    /// into the bin holding its midpoint did not: bins then held alternating numbers of cycles
    /// whenever the bin width was not a multiple of the cycle time, and the result was a sawtooth
    /// that looked like instrument behaviour. This is the regression guard for that.
    /// </summary>
    [Fact]
    public void AUniformAcquisitionGivesAFlatTrace()
    {
        // 0.1 min cycles in 0.25 min bins - deliberately NOT a whole multiple, which is what
        // produced the ripple.
        var cycles = Enumerable.Range(0, 200)
            .Select(i => Cycle(i, 1 + i * 0.1, 1 + (i + 1) * 0.1, 1000))
            .ToArray();

        var profile = Ms2SignalProfile.Build(
            "S1", Array.Empty<MergedRegion>(), cycles, NoLists, NoLists, binWidthMin: 0.25);

        // Interior bins only: the first and last are partly outside the acquisition.
        var interior = profile.Acquired.Skip(1).Take(profile.BinCount - 2).ToArray();
        var mean = interior.Average();
        var ripple = (interior.Max() - interior.Min()) / mean;
        Assert.True(ripple < 1e-9,
            $"a uniform acquisition rippled by {ripple:P2}; the acquired trace is aliasing");
        Assert.Equal(200 * 1000.0, profile.Acquired.Sum(), 3);
    }

    /// <summary>
    /// Lists nest inside assigned at every retention time, and two lists may both claim a region -
    /// the same rule as the per-replicate totals, now per bin.
    /// </summary>
    [Fact]
    public void ListTracesNestInsideAssignedAndMayOverlap()
    {
        var merged = new[]
        {
            Region(1.0, 1.5, 100, lists: 0b11),   // both lists claim it
            Region(2.0, 2.5, 60, lists: 0b01),
            Region(3.0, 3.5, 40),                 // assigned, in neither
        };

        var profile = Ms2SignalProfile.Build(
            "S1", merged, null, new[] { "A", "B" }, new[] { "#2ca02c", "#9467bd" }, 0.25);

        Assert.Equal(2, profile.PerList.Count);
        for (var bin = 0; bin < profile.BinCount; bin++)
        {
            Assert.True(profile.PerList[0][bin] <= profile.Assigned[bin] + 1e-9);
            Assert.True(profile.PerList[1][bin] <= profile.Assigned[bin] + 1e-9);
        }
        Assert.Equal(160, profile.PerList[0].Sum(), 6);   // 100 + 60
        Assert.Equal(100, profile.PerList[1].Sum(), 6);   // the shared region only
    }

    /// <summary>Unassigned regions contribute to nothing, exactly as in the per-replicate totals.</summary>
    [Fact]
    public void UnassignedRegionsAreNotInTheTrace()
    {
        var profile = Ms2SignalProfile.Build(
            "S1",
            new[] { Region(1.0, 1.5, 100, assigned: false, lists: 0b1), Region(2.0, 2.5, 40) },
            null, new[] { "A" }, new[] { "#2ca02c" }, 0.25);

        Assert.Equal(40, profile.Assigned.Sum(), 6);
        Assert.Equal(0, profile.PerList[0].Sum(), 6);
    }

    /// <summary>
    /// The grid spans BOTH sources, so neither trace is clipped by the other's range. A peptide
    /// eluting before the first cycle midpoint must not fall off the plot.
    /// </summary>
    [Fact]
    public void TheGridCoversBothSources()
    {
        var profile = Ms2SignalProfile.Build(
            "S1",
            new[] { Region(0.5, 0.8, 100) },              // earlier than any cycle
            new[] { Cycle(0, 5.0, 5.1, 1000) },           // later than any peptide
            NoLists, NoLists, binWidthMin: 0.5);

        Assert.Equal(100, profile.Assigned.Sum(), 6);
        Assert.Equal(1000, profile.Acquired.Sum(), 6);
        Assert.True(profile.BinStartMin[0] <= 0.5);
        Assert.True(profile.BinStartMin[^1] + profile.BinWidthMin >= 5.0);
    }

    /// <summary>
    /// With no instrument file there is no acquired trace, and that has to be distinguishable from a
    /// run that acquired nothing - the plot label depends on it.
    /// </summary>
    [Fact]
    public void WithoutAnInstrumentFileTheAcquiredTraceIsAbsentNotZero()
    {
        var profile = Ms2SignalProfile.Build(
            "S1", new[] { Region(1.0, 1.5, 100) }, null, NoLists, NoLists, 0.25);

        Assert.False(profile.HasAcquired);
        Assert.All(profile.AssignedFraction, f => Assert.True(double.IsNaN(f)));
    }

    /// <summary>The fraction per bin is the reading the plot exists to give.</summary>
    [Fact]
    public void AssignedFractionIsPerBin()
    {
        var profile = Ms2SignalProfile.Build(
            "S1",
            new[] { Region(1.0, 1.25, 250) },
            new[] { Cycle(0, 1.0, 1.1, 1000), Cycle(1, 1.30, 1.35, 500) },
            NoLists, NoLists, binWidthMin: 0.25);

        // First bin: 250 assigned of 1000 acquired.
        Assert.Equal(0.25, profile.AssignedFraction[0], 6);
        // Second bin: acquired but nothing assigned.
        Assert.Equal(0, profile.AssignedFraction[1], 6);
    }

    /// <summary>Degenerate inputs are answered, not crashed on.</summary>
    [Fact]
    public void EdgeCases()
    {
        var empty = Ms2SignalProfile.Build(
            "S1", Array.Empty<MergedRegion>(), null, NoLists, NoLists, 0.25);
        Assert.True(empty.IsEmpty);
        Assert.Equal(0, empty.BinCount);

        // A zero-width peak still carries area and has to land somewhere.
        var instant = Ms2SignalProfile.Build(
            "S1", new[] { Region(1.0, 1.0, 100) }, null, NoLists, NoLists, 0.25);
        Assert.Equal(100, instant.Assigned.Sum(), 6);

        Assert.Throws<ArgumentOutOfRangeException>(() => Ms2SignalProfile.Build(
            "S1", new[] { Region(1.0, 1.5, 100) }, null, NoLists, NoLists, binWidthMin: 0));
    }

    /// <summary>It renders, including the no-acquired-trace case the report will hit first.</summary>
    [Fact]
    public void RendersWithAndWithoutTheAcquiredTrace()
    {
        var merged = Enumerable.Range(0, 40)
            .Select(i => Region(1 + i * 0.5, 1.4 + i * 0.5, 100 + i * 7, lists: (uint)(i % 3 == 0 ? 1 : 0)))
            .ToArray();
        var cycles = Enumerable.Range(0, 200)
            .Select(i => Cycle(i, 1 + i * 0.1, 1.05 + i * 0.1, 900 + i * 3))
            .ToArray();

        foreach (var withAcquired in new[] { true, false })
        {
            var profile = Ms2SignalProfile.Build(
                "S1", merged, withAcquired ? cycles : null,
                new[] { "Panel A" }, new[] { "#2ca02c" }, 0.25);

            var png = PlotRenderer.Ms2RtProfilePng(profile, "MS2 Signal over Retention Time");
            Assert.True(png.Length > 1000, "the PNG should not be empty");
            Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47 }, png.Take(4).ToArray());

            if (Environment.GetEnvironmentVariable("PRISM_MS2_PLOT_OUT") is { Length: > 0 } dump)
            {
                System.IO.Directory.CreateDirectory(dump);
                System.IO.File.WriteAllBytes(
                    System.IO.Path.Combine(
                        dump, withAcquired ? "ms2_rt_profile.png" : "ms2_rt_profile_no_acquired.png"),
                    png);
            }
        }
    }
}
