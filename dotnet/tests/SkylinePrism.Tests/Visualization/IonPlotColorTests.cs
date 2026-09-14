using System;
using System.Linq;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// Hue says whose replicate it is; shade says which of the two nested quantities.
/// </summary>
/// <remarks>
/// Every ion plot draws acquired, then explained, then quantified, nested - and each replicate also
/// belongs to a sample type. Two things to encode, so they take the two axes a color has. Before
/// this, explained was one fixed muted blue for every replicate: it disappeared into the acquired
/// background, and it threw away the type of the replicate it was drawn on.
/// </remarks>
public class IonPlotColorTests
{
    /// <summary>
    /// Quantified is DARKER than explained, for every type - which is the direction that matters.
    /// </summary>
    /// <remarks>
    /// Lightening would put the inner quantity closer to the light gray acquired bar it is drawn on,
    /// which is the defect this whole change started as.
    /// </remarks>
    [Theory]
    [InlineData("experimental")]
    [InlineData("qc")]
    [InlineData("reference")]
    [InlineData("")]
    [InlineData("some-cohort-specific-type")]
    public void QuantifiedIsTheDarkerShade(string type)
    {
        var explained = PlotRenderer.ExplainedBarColor(type, 0);
        var quantified = PlotRenderer.QuantifiedBarColor(type, 0);

        Assert.True(
            Luminance(quantified) < Luminance(explained),
            $"{type}: quantified {Hex(quantified)} is not darker than explained {Hex(explained)}");
    }

    /// <summary>
    /// And it stays the same hue, so the pair still reads as one quantity inside another rather than
    /// as two unrelated series.
    /// </summary>
    [Theory]
    [InlineData("experimental")]
    [InlineData("qc")]
    [InlineData("reference")]
    public void TheDarkerShadeKeepsTheHue(string type)
    {
        var explained = PlotRenderer.ExplainedBarColor(type, 0);
        var quantified = PlotRenderer.QuantifiedBarColor(type, 0);

        Assert.True(
            Math.Abs(Hue(explained) - Hue(quantified)) < 25,
            $"{type}: {Hex(explained)} and {Hex(quantified)} are not the same hue");
    }

    /// <summary>
    /// The controls keep their own colors. That is what coloring these bars is for - a control
    /// replicate has to stay findable along a row of two hundred.
    /// </summary>
    [Fact]
    public void ControlsAreStillTellableFromExperimentals()
    {
        var hues = new[] { "experimental", "qc", "reference" }
            .Select(t => Hue(PlotRenderer.ExplainedBarColor(t, 0)))
            .ToArray();

        Assert.True(Math.Abs(hues[0] - hues[1]) > 40, "experimental and qc look alike");
        Assert.True(Math.Abs(hues[0] - hues[2]) > 40, "experimental and reference look alike");
    }

    /// <summary>An untyped replicate is drawn as experimental rather than as its own category.</summary>
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void AnUntypedReplicateIsExperimental(string? type)
    {
        Assert.Equal(
            Hex(PlotRenderer.ExplainedBarColor("experimental", 0)),
            Hex(PlotRenderer.ExplainedBarColor(type, 0)));
    }

    private static string Hex(ScottPlot.Color c) => $"#{c.R:X2}{c.G:X2}{c.B:X2}";

    private static double Luminance(ScottPlot.Color c) =>
        0.2126 * c.R + 0.7152 * c.G + 0.0722 * c.B;

    /// <summary>Hue in degrees, enough to say "still orange" without a color library.</summary>
    private static double Hue(ScottPlot.Color c)
    {
        double r = c.R / 255.0, g = c.G / 255.0, b = c.B / 255.0;
        var max = Math.Max(r, Math.Max(g, b));
        var min = Math.Min(r, Math.Min(g, b));
        var d = max - min;
        if (d < 1e-9)
            return 0;
        var h = max == r ? (g - b) / d % 6 : max == g ? (b - r) / d + 2 : (r - g) / d + 4;
        h *= 60;
        return h < 0 ? h + 360 : h;
    }
}
