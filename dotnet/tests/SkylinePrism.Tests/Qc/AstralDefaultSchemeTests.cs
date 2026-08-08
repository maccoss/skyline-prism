using System;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The built-in starting scheme for the Spectrum density map.
///
/// <para>It replaced Skyline's saved isolation schemes, which were being offered as fallbacks: generic
/// SWATH templates (25 m/z, VW 64) from an era before narrow-window DIA. Binning a 3 Th acquisition on
/// a 25 Th grid gives a map that looks plausible and is wrong, so the fallback had to be something
/// representative instead.</para>
///
/// <para>The values are generated from (start, step, width) rather than stored as 167 literals. These
/// pin that the generation still reproduces the real forbidden-zone acquisition it was taken from -
/// otherwise "3 Th grid" could drift into a round 400.0/3.0000 grid, which is exactly the approximation
/// the whole feature exists to avoid.</para>
/// </summary>
public class AstralDefaultSchemeTests
{
    [Fact]
    public void CoversTheAdvertisedRangeWithTheAdvertisedWindows()
    {
        var scheme = IsolationScheme.AstralDefault();

        Assert.Equal(IsolationScheme.AstralDefaultName, scheme.Name);
        Assert.True(scheme.HasWindows);
        Assert.Equal(167, scheme.Windows.Count);

        Assert.Equal(400.4319, scheme.Windows[0].Start, 4);
        Assert.Equal(901.66, scheme.Windows[^1].End, 2);
    }

    /// <summary>
    /// The edges are deliberately NOT round. A forbidden-zone scheme places its boundaries where peptide
    /// isotope clusters do not fall, which is the entire reason a uniform grid starting at 400.0 is
    /// wrong - it sits ~14% of a window off and cuts through the clusters the scheme protects.
    /// </summary>
    [Fact]
    public void StartsOffARoundNumber()
    {
        var first = IsolationScheme.AstralDefault().Windows[0];

        Assert.NotEqual(400.0, first.Start, 2);
        Assert.InRange(first.Start - 400.0, 0.4, 0.5);
    }

    /// <summary>Windows tile the range: each starts where the previous ended, within instrument rounding.</summary>
    [Fact]
    public void WindowsTileWithoutGapsOrOverlaps()
    {
        var w = IsolationScheme.AstralDefault().Windows;

        for (var i = 1; i < w.Count; i++)
        {
            var seam = w[i].Start - w[i - 1].End;
            Assert.True(Math.Abs(seam) < 0.001,
                $"window {i} leaves a {seam * 1000:F2} mDa {(seam > 0 ? "gap" : "overlap")}");
        }
    }

    /// <summary>
    /// ~3.0014 Th, not 3.0000: the width is an integer multiple of the ~1.00045 Th spacing between
    /// isotope peaks, which is what keeps the boundaries inside the forbidden zones.
    /// </summary>
    [Fact]
    public void WindowsAreThreeOhOhOneFourWideNotThree()
    {
        var widths = IsolationScheme.AstralDefault().Windows
            .Select(x => x.End - x.Start).ToList();

        Assert.All(widths, x => Assert.InRange(x, 3.0013, 3.0015));
        Assert.DoesNotContain(widths, x => Math.Abs(x - 3.0) < 1e-6);
    }

    /// <summary>
    /// Reproduces the acquisition it was derived from. The measured windows of that run started at
    /// 400.431890 and ended at 901.6597; the generated template must land on them to well under a
    /// milli-Dalton, or it is no longer representative of the instrument it is named for.
    /// </summary>
    [Fact]
    public void MatchesTheAcquisitionItWasDerivedFrom()
    {
        const double measuredFirstStart = 400.431890003052;
        const double measuredLastEnd = 901.6597;

        var w = IsolationScheme.AstralDefault().Windows;

        Assert.True(Math.Abs(w[0].Start - measuredFirstStart) < 0.001,
            $"first window off by {(w[0].Start - measuredFirstStart) * 1000:F3} mDa");
        Assert.True(Math.Abs(w[^1].End - measuredLastEnd) < 0.001,
            $"last window off by {(w[^1].End - measuredLastEnd) * 1000:F3} mDa");
    }

    /// <summary>Every precursor in the range lands in a window - the map has no dead columns.</summary>
    [Fact]
    public void EveryPrecursorInRangeFallsInAWindow()
    {
        var scheme = IsolationScheme.AstralDefault();
        var probes = Enumerable.Range(0, 500)
            .Select(i => 401.0 + i * (900.0 - 401.0) / 499.0)
            .ToList();

        Assert.Equal(1.0, scheme.Coverage(probes), 3);
    }
}
