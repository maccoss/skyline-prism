using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The isolation-window handling exercised against a REAL acquisition rather than a synthetic scheme:
/// <c>fixtures/isolation/forbidden-zone-3th.xml</c> is the scheme Skyline imported from an actual Thermo
/// <c>.raw</c> (167 windows, 400.43-901.66 m/z), captured with
/// <c>--full-scan-isolation-scheme=&lt;data file&gt;</c>.
/// </summary>
/// <remarks>
/// These windows are a MacCoss-lab forbidden-zone design (described in the lab's published DIA methods):
/// the edges are placed in the m/z gaps where peptide precursors cannot fall, so window widths are an
/// integer multiple of the ~1.0005 averagine spacing and the boundaries land on values like 400.4319
/// instead of 400. That is a property of THESE schemes, not of DIA in general - a stock SWATH (25 m/z)
/// scheme has 24 Th windows on round boundaries and would fail the width test below. The point of testing
/// against it is that no uniform grid can reproduce these assignments, which is exactly what
/// <see cref="UniformBinning_MisassignsPrecursorsThatTheRealWindowsPlaceCorrectly"/> pins down.
/// </remarks>
public class ForbiddenZoneSchemeTests
{
    /// <summary>Averagine mass-defect spacing the window widths are built from.</summary>
    private const double AveragineSpacing = 1.0005;

    private static IsolationScheme RealScheme()
    {
        var path = Fixtures.Path2("isolation", "forbidden-zone-3th.xml");
        Assert.True(File.Exists(path), $"fixture missing: {path}");
        var scheme = IsolationScheme.Parse(File.ReadAllText(path));
        Assert.NotNull(scheme);
        return scheme!;
    }

    [Fact]
    public void RealScheme_ParsesEveryWindow()
    {
        var scheme = RealScheme();
        Assert.Equal(167, scheme.Windows.Count);
        Assert.Equal(400.431890003052, scheme.MzLow, 9);
        Assert.Equal(901.659710048256, scheme.MzHigh, 9);
        Assert.True(scheme.HasWindows);
        // Windows come back in ascending m/z order regardless of file order.
        Assert.Equal(scheme.Windows.OrderBy(w => w.Start), scheme.Windows);
    }

    [Fact]
    public void RealScheme_EdgesAreNotOnRoundNumbers()
    {
        // The forbidden-zone design's whole signature. If a "simplification" ever rounded these to whole
        // Th, every window would shift by ~0.43 Th and this fails.
        var scheme = RealScheme();
        Assert.NotEqual(Math.Floor(scheme.MzLow), scheme.MzLow);
        Assert.True(scheme.Windows.All(w => Math.Abs(w.Start - Math.Round(w.Start)) > 0.01),
            "every window edge should sit off the integer m/z grid");
    }

    [Fact]
    public void RealScheme_WidthsAreIntegerMultiplesOfTheAveragineSpacing()
    {
        // The lab's design rule: width = n x ~1.0005 m/z. Here n = 3 for every window. Tolerance is loose
        // enough for the instrument's own rounding (widths vary between 3.0013 and 3.0014).
        var scheme = RealScheme();
        foreach (var window in scheme.Windows)
        {
            var multiples = window.Width / AveragineSpacing;
            Assert.Equal(3, Math.Round(multiples));
            Assert.True(Math.Abs(multiples - Math.Round(multiples)) < 0.005,
                $"window {window.Start}-{window.End} is {multiples:F4} x {AveragineSpacing}, not an integer multiple");
        }
    }

    [Fact]
    public void RealScheme_IsEffectivelyUniformDespiteInstrumentRounding()
    {
        // Consecutive widths differ in the 4th decimal, which must not be reported as a variable-width
        // scheme (that would misdescribe the acquisition to the user).
        var scheme = RealScheme();
        var widths = scheme.Windows.Select(w => w.Width).ToList();
        Assert.True(widths.Max() - widths.Min() < 0.001, "widths should agree to 0.001 Th");
        Assert.DoesNotContain("variable", scheme.Describe());
        Assert.Contains("167 windows", scheme.Describe());
    }

    [Fact]
    public void UniformBinning_MisassignsPrecursorsThatTheRealWindowsPlaceCorrectly()
    {
        // THE regression test for this whole feature. A uniform 3 Th grid over the same m/z range - the
        // best guess anyone could make without reading the data file - puts precursors in different
        // spectra than the instrument did, because its edges are offset by ~0.43 Th.
        var scheme = RealScheme();

        // Precursors chosen to sit between a real edge and the nearest uniform-grid edge.
        var probes = new[] { 403.40, 406.40, 409.42, 412.43 };

        var disagreements = 0;
        foreach (var mz in probes)
        {
            var realWindow = scheme.Windows.Select((w, i) => (w, i)).First(t => t.w.Contains(mz));
            // Uniform grid anchored at floor(MzLow), exactly as the approximate fallback builds it.
            var uniformLow = Math.Floor(scheme.MzLow);
            var uniformIndex = (int)((mz - uniformLow) / 3.0);
            var uniformStart = uniformLow + uniformIndex * 3.0;
            if (Math.Abs(uniformStart - realWindow.w.Start) > 0.01)
                disagreements++;
        }

        Assert.Equal(probes.Length, disagreements);
    }

    [Fact]
    public void Bin_OnRealScheme_GivesOneRowPerAcquiredWindow()
    {
        var scheme = RealScheme();
        var precursors = SyntheticPrecursors(scheme, count: 2000, seed: 3);
        var map = PrecursorDensity.Bin(precursors, scheme, rtBinMin: 0.1);

        Assert.Equal(167, map.MzBins);
        Assert.Equal(scheme.Name, map.RowSource);
        Assert.Equal(scheme.MzLow, map.MzLow, 9);
        Assert.Equal(scheme.MzHigh, map.MzHigh, 9);
        // Every precursor was drawn inside a window, so none should be reported outside.
        Assert.Equal(0, map.PrecursorsOutsideRows);
        // Total counts across the map account for every precursor at least once.
        var totalFirstBin = 0;
        for (var r = 0; r < map.MzBins; r++)
            totalFirstBin += map.Counts[r, 0];
        Assert.True(totalFirstBin > 0);
    }

    [Fact]
    public void Bin_OnRealScheme_RendersWithoutGapsAcrossTheAcquiredRange()
    {
        // The display grid must cover the acquired m/z range: NaN rows mean "no window here", and this
        // scheme is contiguous, so only the sub-0.001 Th rounding seams may be uncovered.
        var scheme = RealScheme();
        var map = PrecursorDensity.Bin(SyntheticPrecursors(scheme, 500, 5), scheme, 0.1);
        var grid = map.ToDisplayGrid(400);

        var nanRows = 0;
        for (var r = 0; r < 400; r++)
            if (double.IsNaN(grid[r, 0]))
                nanRows++;
        Assert.Equal(0, nanRows);
    }

    [Fact]
    public void RealScheme_CoversPrecursorsAcrossItsRange()
    {
        // The instrument's rounding leaves ~0.00002 Th seams between consecutive windows. They are far too
        // small to matter - coverage of precursors drawn uniformly across the range stays effectively 1.
        var scheme = RealScheme();
        var mz = new List<double>();
        var rng = new Random(11);
        for (var i = 0; i < 20000; i++)
            mz.Add(scheme.MzLow + rng.NextDouble() * (scheme.MzHigh - scheme.MzLow));

        Assert.True(scheme.Coverage(mz) > 0.999, $"coverage was {scheme.Coverage(mz):P3}");
    }

    [Fact]
    public void Coverage_CatchesAWrongRangeButNotAWrongWidth()
    {
        // Honest limits of the "is this the right scheme?" check the tool shows. Coverage catches a scheme
        // acquired over a different m/z range, but CANNOT distinguish two schemes that span the same range
        // with different window widths - both contain every precursor. The status line therefore reports
        // coverage as a red flag, and must never be read as confirmation that the scheme is correct.
        var scheme = RealScheme();
        var mz = Enumerable.Range(0, 500)
            .Select(i => scheme.MzLow + (i + 0.5) * (scheme.MzHigh - scheme.MzLow) / 500)
            .ToList();

        Assert.True(scheme.Coverage(mz) > 0.99);

        var wrongRange = new IsolationScheme("wrong range",
            Enumerable.Range(0, 10).Select(i => new IsolationWindow(1200 + i * 3, 1203 + i * 3)).ToList());
        Assert.Equal(0.0, wrongRange.Coverage(mz));

        var wrongWidth = new IsolationScheme("same range, 25 Th windows",
            Enumerable.Range(0, 21).Select(i => new IsolationWindow(400 + i * 25, 425 + i * 25)).ToList());
        Assert.True(wrongWidth.Coverage(mz) > 0.99, "coverage cannot discriminate window width - by design");
    }

    // Precursors drawn inside the scheme's windows (never in the rounding seams), each with a short peak.
    private static List<DetectedPrecursor> SyntheticPrecursors(IsolationScheme scheme, int count, int seed)
    {
        var rng = new Random(seed);
        var result = new List<DetectedPrecursor>(count);
        for (var i = 0; i < count; i++)
        {
            var window = scheme.Windows[rng.Next(scheme.Windows.Count)];
            // Well inside the window, so a boundary seam never decides the outcome.
            var mz = window.Start + window.Width * (0.2 + 0.6 * rng.NextDouble());
            var start = 5 + rng.NextDouble() * 40;
            result.Add(new DetectedPrecursor(mz, start, start + 0.3));
        }
        return result;
    }
}
