using System;
using System.Linq;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// The RT-bin axes say which minutes each bin covers, not which bin it is.
/// </summary>
/// <remarks>
/// The bars sit at 0..n-1, so ScottPlot's automatic ticks labelled the midpoints too and eight bins
/// came out as "-0.5, 0, 0.5, 1 ... 7.5" - neither eight of anything nor a retention time. The index
/// was the deeper problem: the reason to plot CV against RT is to find WHEN a run is noisy, and
/// "bin 6" cannot be held against a chromatogram or an acquisition method, both of which are in
/// minutes.
/// </remarks>
public class RtBinLabelTests
{
    /// <summary>A real gradient: whole minutes, one label per bin, spanning the real range.</summary>
    [Fact]
    public void AGradientIsLabelledInWholeMinutes()
    {
        var labels = PlotRenderer.RtBinLabels(2.3, 64.8, 8);

        Assert.Equal(8, labels.Length);
        Assert.Equal("2-10", labels[0]);

        // The last bin ends at the last peptide, not at the open-ended edge the binning uses.
        Assert.Equal("57-65", labels[^1]);
        Assert.DoesNotContain(labels, l => l.Contains("Infinity", StringComparison.Ordinal));
        Assert.DoesNotContain(labels, l => l.Contains('.'));
    }

    /// <summary>Each bin starts where the last one ended, so the axis reads continuously.</summary>
    [Fact]
    public void TheBinsJoinUp()
    {
        var labels = PlotRenderer.RtBinLabels(0, 80, 8);

        Assert.Equal(
            new[] { "0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80" }, labels);
    }

    /// <summary>
    /// A gradient short enough for sub-minute bins keeps the decimals it needs.
    /// </summary>
    /// <remarks>
    /// Rounding to whole minutes was the request, and on a four-minute run it would print "1-1" for
    /// three bins running - a label that is not wrong so much as empty. The precision follows the bin
    /// width rather than being fixed.
    /// </remarks>
    [Fact]
    public void AShortGradientDoesNotCollapseToRepeatedLabels()
    {
        var labels = PlotRenderer.RtBinLabels(0, 4, 8);

        Assert.Equal(8, labels.Length);
        Assert.Equal(labels.Length, labels.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal("0-0.5", labels[0]);
    }

    /// <summary>And a very short one goes further rather than repeating itself.</summary>
    [Fact]
    public void AVeryShortGradientGoesToTwoDecimals()
    {
        var labels = PlotRenderer.RtBinLabels(10, 10.8, 8);

        Assert.Equal(labels.Length, labels.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal("10-10.1", labels[0]);
    }
}
