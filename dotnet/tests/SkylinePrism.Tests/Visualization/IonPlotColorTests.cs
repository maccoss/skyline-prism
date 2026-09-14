using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// One color scheme across the ion plots, without giving up the sample types.
/// </summary>
/// <remarks>
/// Explained is the strong blue on every plot in this family now - it was a muted blue that
/// disappeared into the acquired band, which is backwards for the headline number. Experimental's own
/// color IS that strong blue, so an experimental quantified bar and the explained bar nested above it
/// would have been one shape. Only experimental moves to the navy; qc and reference keep orange and
/// red, because a control replicate has to stay findable along a row of two hundred.
/// </remarks>
public class IonPlotColorTests
{
    [Fact]
    public void ExperimentalTakesTheQuantifiedNavySoItCannotMergeWithExplained()
    {
        var experimental = PlotRenderer.QuantifiedBarColor("experimental", 0);

        Assert.NotEqual(PlotRenderer.ExplainedColorHex, ToHex(experimental));
        Assert.Equal("#08306B", ToHex(experimental));
    }

    /// <summary>Blank and "unknown" are the same case - no type stated, so not a control.</summary>
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    [InlineData("Experimental")]
    [InlineData("unknown")]
    public void AnUntypedReplicateIsDrawnAsExperimental(string? type)
    {
        Assert.Equal("#08306B", ToHex(PlotRenderer.QuantifiedBarColor(type, 0)));
    }

    /// <summary>The whole reason these bars are colored at all.</summary>
    [Theory]
    [InlineData("qc", "#FF7F0E")]
    [InlineData("QC", "#FF7F0E")]
    [InlineData("reference", "#D62728")]
    [InlineData("standard", "#D62728")]
    public void ControlsKeepTheirColors(string type, string expected)
    {
        Assert.Equal(expected, ToHex(PlotRenderer.QuantifiedBarColor(type, 0)));
    }

    private static string ToHex(ScottPlot.Color c) =>
        $"#{c.R:X2}{c.G:X2}{c.B:X2}";
}
