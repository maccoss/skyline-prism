using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// Every piece of text PRISM draws must be in the same family, across the QC report PNGs, the
/// interactive tool's plots, and the HTML that embeds them.
/// <para>
/// This is a regression test with a specific history: the correlation heatmap set its own font
/// SIZES and never called <see cref="PlotRenderer.StyleQcPlot"/>, so it silently rendered in the
/// backend's default typeface while the plots either side of it were in Segoe UI. The color bars
/// and text annotations had the same hole - ScottPlot styles those per item, so a call site that
/// forgets is invisible until someone looks at the report.
/// </para>
/// </summary>
public class PlotFontConsistencyTests
{
    /// <summary>
    /// Walks a plot and reports every text style whose family is not the pinned one. Checks the
    /// items that are actually configurable per plot; anything a future plot adds and forgets to
    /// style shows up here as soon as it is included in <see cref="AllPlots"/>.
    /// </summary>
    private static List<string> UnpinnedText(Plot plt)
    {
        var wrong = new List<string>();

        void Check(string what, string? fontName)
        {
            if (!string.Equals(fontName, PlotRenderer.PlotFontName, StringComparison.Ordinal))
                wrong.Add($"{what} = '{fontName}' (expected '{PlotRenderer.PlotFontName}')");
        }

        Check("title", plt.Axes.Title.Label.FontName);
        Check("left axis label", plt.Axes.Left.Label.FontName);
        Check("bottom axis label", plt.Axes.Bottom.Label.FontName);
        Check("left tick labels", plt.Axes.Left.TickLabelStyle.FontName);
        Check("bottom tick labels", plt.Axes.Bottom.TickLabelStyle.FontName);
        Check("legend", plt.Legend.FontName);

        foreach (var text in plt.GetPlottables<ScottPlot.Plottables.Text>())
            Check($"text annotation '{text.LabelText}'", text.LabelFontName);

        foreach (var bar in ColorBars(plt))
        {
            Check("color bar label", bar.LabelStyle.FontName);
            Check("color bar tick labels", bar.Axis.TickLabelStyle.FontName);
        }

        return wrong;
    }

    public static IEnumerable<object[]> AllPlots() => new[]
    {
        new object[] { "correlation heatmap" },
        new object[] { "precursor density" },
        new object[] { "precursor load histogram" },
        new object[] { "precursor load over time" },
        new object[] { "dynamic range" },
        new object[] { "intensity density" },
    };

    [Theory]
    [MemberData(nameof(AllPlots))]
    public void EveryPlot_UsesThePinnedFont(string which)
    {
        var plt = new Plot();
        switch (which)
        {
            case "correlation heatmap":
                // 6 samples, enough to be under the n<=15 threshold that draws the per-cell text.
                PlotRenderer.DrawCorrelationHeatmap(
                    plt, SyntheticMatrix(40, 6), Enumerable.Range(0, 6).ToArray(),
                    "Control correlation",
                    new[] { "reference", "reference", "qc", "qc", "experimental", "experimental" });
                break;

            case "precursor density":
                PlotRenderer.DrawPrecursorDensity(plt, DensityMap(), title: "A run");
                break;

            case "precursor load histogram":
                PlotRenderer.DrawPrecursorLoadHistogram(plt, DensityMap(), title: "A run");
                break;

            case "precursor load over time":
                PlotRenderer.DrawPrecursorLoadOverTime(plt, DensityMap(), title: "A run");
                break;

            case "dynamic range":
                PlotRenderer.DrawDynamicRange(
                    plt, RangeEntries(), Array.Empty<(string, string, IReadOnlyList<AbundanceEntry>)>(),
                    yLabel: "Log10 abundance", xLabel: "Protein rank");
                break;

            case "intensity density":
                PlotRenderer.DrawIntensityDensity(plt, SyntheticMatrix(60, 6));
                break;
        }

        var wrong = UnpinnedText(plt);
        Assert.True(wrong.Count == 0,
            $"{which} draws text in a different family from every other plot:\n  "
            + string.Join("\n  ", wrong));
    }

    /// <summary>
    /// The heatmap is the one that regressed, so it gets a direct check that it really did draw the
    /// per-cell annotations and a color bar - otherwise the theory above would pass by having
    /// nothing to inspect.
    /// </summary>
    [Fact]
    public void CorrelationHeatmap_ReallyDrawsTheItemsBeingChecked()
    {
        var plt = new Plot();
        PlotRenderer.DrawCorrelationHeatmap(
            plt, SyntheticMatrix(40, 6), Enumerable.Range(0, 6).ToArray(), "Control correlation");

        Assert.Equal(36, plt.GetPlottables<ScottPlot.Plottables.Text>().Count()); // 6x6 cells
        Assert.Single(ColorBars(plt));
    }

    /// <summary>
    /// The HTML report embeds the plot images, so its own text has to be in the family the plots
    /// resolved to - not a fixed stack that happens to start with a different name.
    /// </summary>
    [Fact]
    public void HtmlFontStack_LeadsWithTheResolvedPlotFont()
    {
        var stack = PlotRenderer.HtmlFontStack;
        Assert.StartsWith($"'{PlotRenderer.PlotFontName}'", stack, StringComparison.Ordinal);
        Assert.EndsWith("sans-serif", stack, StringComparison.Ordinal); // always a generic fallback
    }

    // ---------------------------------------------------------------- fixtures

    private static double[,] SyntheticMatrix(int features, int samples)
    {
        var m = new double[features, samples];
        for (var f = 0; f < features; f++)
            for (var s = 0; s < samples; s++)
                m[f, s] = 18.0 + ((f * 7 + s * 13) % 11) * 0.31 + s * 0.05;
        return m;
    }

    private static IEnumerable<ScottPlot.Panels.ColorBar> ColorBars(Plot plt) =>
        plt.Axes.GetPanels().OfType<ScottPlot.Panels.ColorBar>();

    private static PrecursorDensityMap DensityMap() => PrecursorDensity.Bin(
        new[]
        {
            new DetectedPrecursor(500.4, 10.0, 10.25),
            new DetectedPrecursor(506.0, 10.1, 10.4),
        },
        mzBinTh: 2.0, rtBinMin: 0.1);

    private static List<AbundanceEntry> RangeEntries()
    {
        var entries = new List<AbundanceEntry>();
        for (var i = 0; i < 25; i++)
            entries.Add(new AbundanceEntry(
                Key: $"PG{i:D3}", Label: $"Gene{i}", Accession: $"P{i:D3}", Gene: $"Gene{i}",
                ProteinName: $"Protein {i}", MeanAbundance: Math.Pow(10, 8.0 - i * 0.12),
                Log10Abundance: 8.0 - i * 0.12, Rank: i, SamplesUsed: 6));
        return entries;
    }
}
