using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// The intensity-distribution plot must look the same in the HTML QC report and in the tool's
/// interactive QC tab. Both render through <see cref="PlotRenderer.DrawIntensityDensity"/>, so the only
/// way they can diverge is the caller dropping the group labels - which is exactly what the report used
/// to do, giving it a per-sample rainbow while the tool colored by sample type.
/// </summary>
public class IntensityDistributionColorTests
{
    private static double[,] Matrix(int nFeatures, int nSamples)
    {
        var rng = new Random(7);
        var m = new double[nFeatures, nSamples];
        for (var f = 0; f < nFeatures; f++)
            for (var s = 0; s < nSamples; s++)
                m[f, s] = 14.0 + (s % 3) * 0.4 + rng.NextDouble() * 2 - 1;
        return m;
    }

    private static List<string> Types(int nSamples) =>
        Enumerable.Range(0, nSamples)
            .Select(i => i % 3 == 0 ? "reference" : i % 3 == 1 ? "qc" : "experimental")
            .ToList();

    /// <summary>Line colors of the density curves, in draw order.</summary>
    private static List<Color> CurveColors(Plot plt) =>
        plt.GetPlottables().OfType<ScottPlot.Plottables.Scatter>().Select(s => s.Color).ToList();

    [Fact]
    public void GroupLabels_ColorEverySampleOfATypeIdentically()
    {
        var plt = new Plot();
        PlotRenderer.DrawIntensityDensity(plt, Matrix(200, 9), Types(9));

        var colors = CurveColors(plt);
        Assert.Equal(9, colors.Count);
        // 9 samples, 3 types -> exactly 3 distinct colors, not 9.
        Assert.Equal(3, colors.Distinct().Count());
        // Samples of the same type (indices 0,3,6) share a color.
        Assert.Equal(colors[0], colors[3]);
        Assert.Equal(colors[0], colors[6]);
        Assert.NotEqual(colors[0], colors[1]);
    }

    [Fact]
    public void WithoutGroupLabels_EachSampleGetsItsOwnColor()
    {
        // The old report behavior, retained for the genuinely ungrouped case.
        var plt = new Plot();
        PlotRenderer.DrawIntensityDensity(plt, Matrix(200, 9));

        Assert.Equal(9, CurveColors(plt).Distinct().Count());
    }

    [Fact]
    public void TheHtmlReportPathColorsByType_NotPerSample()
    {
        // IntensityDistribution used to accept sampleTypes and drop them. Rendering the same matrix with
        // uniform vs varied types must therefore differ - if it did not, the labels are being ignored again.
        var m = Matrix(200, 9);
        var varied = PlotRenderer.IntensityDistribution(m, Types(9), "t");
        var uniform = PlotRenderer.IntensityDistribution(
            m, Enumerable.Repeat("reference", 9).ToList(), "t");

        Assert.NotEqual(varied, uniform);
    }

    [Fact]
    public void SkylineAndPrismSpellingsOfATypeShareAColor()
    {
        // The tool groups by the Replicates report's "Sample Type" (Skyline spellings); the HTML report
        // uses PRISM's mapped names. They must still agree on color, or the two views disagree.
        Assert.Equal(PlotRenderer.GroupColor("Standard"), PlotRenderer.GroupColor("reference"));
        Assert.Equal(PlotRenderer.GroupColor("Quality Control"), PlotRenderer.GroupColor("qc"));
        Assert.Equal(PlotRenderer.GroupColor("Unknown"), PlotRenderer.GroupColor("experimental"));
    }

    [Fact]
    public void ATypeKeepsItsColorRegardlessOfHowManyGroupsArePresent()
    {
        // Colors must not shift with the cycle index when a group set changes between plots.
        var two = new Plot();
        PlotRenderer.DrawIntensityDensity(two, Matrix(100, 2), new List<string> { "reference", "qc" });
        var three = new Plot();
        PlotRenderer.DrawIntensityDensity(
            three, Matrix(100, 3), new List<string> { "reference", "qc", "experimental" });

        Assert.Equal(CurveColors(two)[0], CurveColors(three)[0]); // reference
        Assert.Equal(CurveColors(two)[1], CurveColors(three)[1]); // qc
    }

    [Fact]
    public void GroupedPlotsGetALegend_UngroupedDoNot()
    {
        var grouped = new Plot();
        PlotRenderer.DrawIntensityDensity(grouped, Matrix(100, 6), Types(6));
        var ungrouped = new Plot();
        PlotRenderer.DrawIntensityDensity(ungrouped, Matrix(100, 6));

        // One legend entry per distinct type, and none when there is no grouping to explain.
        Assert.Equal(3, grouped.GetPlottables().OfType<ScottPlot.Plottables.Scatter>()
            .Count(s => !string.IsNullOrEmpty(s.LegendText)));
        Assert.Equal(0, ungrouped.GetPlottables().OfType<ScottPlot.Plottables.Scatter>()
            .Count(s => !string.IsNullOrEmpty(s.LegendText)));
    }
}
