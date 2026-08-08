using System;
using System.Linq;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// Render smoke tests for the newer QC plots: they must produce a valid PNG (SkiaSharp path).
/// (CI installs libfontconfig1/libfreetype6 for the headless SkiaSharp backend.)
/// </summary>
public class PlotRendererSmokeTests
{
    private static void AssertPng(byte[] png)
    {
        Assert.True(png.Length > 100, "expected non-trivial PNG output");
        // PNG magic bytes.
        Assert.Equal(0x89, png[0]);
        Assert.Equal(0x50, png[1]);
        Assert.Equal(0x4E, png[2]);
        Assert.Equal(0x47, png[3]);
    }

    [Fact]
    public void PrecursorDensity_Renders()
    {
        // A tiny cohort of co-eluting precursors across a 400-460 m/z span.
        var precursors = new System.Collections.Generic.List<DetectedPrecursor>();
        for (var i = 0; i < 200; i++)
            precursors.Add(new DetectedPrecursor(400 + i % 60, 5 + i % 20 * 0.5, 5.4 + i % 20 * 0.5));
        var map = PrecursorDensity.Bin(precursors, mzBinTh: 4.0, rtBinMin: 0.1);

        AssertPng(PlotRenderer.PrecursorDensityPng(map, title: "Precursors per DIA spectrum"));
        // Every colormap offered in the tool's dropdown must render.
        foreach (var cmap in new ScottPlot.IColormap[]
                 {
                     new ScottPlot.Colormaps.Viridis(), new ScottPlot.Colormaps.Turbo(),
                     new ScottPlot.Colormaps.Magma(), new ScottPlot.Colormaps.Inferno(),
                     new ScottPlot.Colormaps.Plasma(), new ScottPlot.Colormaps.Thermal(),
                     new ScottPlot.Colormaps.GrayscaleReversed(),
                 })
        {
            AssertPng(PlotRenderer.PrecursorDensityPng(map, cmap));
        }
    }

    [Fact]
    public void DynamicRange_Renders()
    {
        var entries = new System.Collections.Generic.List<AbundanceEntry>();
        for (var i = 0; i < 500; i++)
        {
            var log10 = 10 - 5.0 * i / 500;
            entries.Add(new AbundanceEntry(
                $"sp|P{i:D5}|G{i}_HUMAN", $"GEN{i}", $"P{i:D5}", $"GEN{i}", $"sp|P{i:D5}|G{i}_HUMAN",
                System.Math.Pow(10, log10), log10, i + 1, 3));
        }
        var highlight = entries.Take(10).ToList();

        AssertPng(PlotRenderer.DynamicRangePng(
            entries.Skip(10).ToList(),
            new[] { ("EV markers", "#1f77b4", (System.Collections.Generic.IReadOnlyList<AbundanceEntry>)highlight) }));

        // No lists defined: just the background curve, and no legend to draw.
        AssertPng(PlotRenderer.DynamicRangePng(
            entries,
            System.Array.Empty<(string, string, System.Collections.Generic.IReadOnlyList<AbundanceEntry>)>()));
    }

    [Fact]
    public void PrecursorDensity_EmptyMapRendersAMessage()
    {
        // The tool draws this when a run has no detections at the chosen q-value; it must not throw.
        AssertPng(PlotRenderer.PrecursorDensityPng(
            PrecursorDensity.Bin(Array.Empty<DetectedPrecursor>()), title: "No detected precursors"));
    }

    /// <summary>
    /// The other two views of the same map. They are drawn into the tool's live plot rather than saved,
    /// so there is no ...Png helper for them - but they still have to survive the SkiaSharp render path,
    /// which is where a bad fill or an empty series shows up.
    /// </summary>
    [Fact]
    public void PrecursorLoadSummaries_Render()
    {
        var precursors = new System.Collections.Generic.List<DetectedPrecursor>();
        for (var i = 0; i < 200; i++)
            precursors.Add(new DetectedPrecursor(400 + i % 60, 5 + i % 20 * 0.5, 5.4 + i % 20 * 0.5));
        var map = PrecursorDensity.Bin(precursors, mzBinTh: 4.0, rtBinMin: 0.1);
        var empty = PrecursorDensity.Bin(Array.Empty<DetectedPrecursor>());

        foreach (var (m, title) in new[] { (map, "Precursor load"), (empty, "No detected precursors") })
        {
            var histogram = new ScottPlot.Plot();
            PlotRenderer.DrawPrecursorLoadHistogram(histogram, m, title);
            AssertPng(histogram.GetImageBytes(1400, 900, ScottPlot.ImageFormat.Png));

            var overTime = new ScottPlot.Plot();
            PlotRenderer.DrawPrecursorLoadOverTime(overTime, m, title);
            AssertPng(overTime.GetImageBytes(1400, 900, ScottPlot.ImageFormat.Png));
        }
    }

    [Fact]
    public void CorrelationHeatmap_Renders()
    {
        var m = new double[20, 4];
        for (var f = 0; f < 20; f++)
            for (var s = 0; s < 4; s++)
                m[f, s] = f + 0.3 * s + (f % 3 == 0 ? 1 : 0);
        AssertPng(PlotRenderer.CorrelationHeatmap(m, new[] { 0, 1, 2, 3 }, "Control Correlation"));
    }

    [Fact]
    public void RtLowessCurves_Renders()
    {
        var m = new double[40, 3];
        var rt = new double[40];
        for (var f = 0; f < 40; f++)
        {
            rt[f] = f;
            for (var s = 0; s < 3; s++)
                m[f, s] = 12 + 0.02 * s * f + (f % 5 - 2) * 0.1;
        }
        AssertPng(PlotRenderer.RtLowessCurves(m, rt, new[] { "reference", "qc", "experimental" }, "RT curves"));
    }

    [Fact]
    public void RtBinCv_Renders()
    {
        var raw = new double[40, 4];
        var corr = new double[40, 4];
        var rt = new double[40];
        for (var f = 0; f < 40; f++)
        {
            rt[f] = f;
            for (var s = 0; s < 4; s++)
            {
                raw[f, s] = 12 + 0.05 * s * (f % 8);
                corr[f, s] = 12 + 0.01 * s;
            }
        }
        AssertPng(PlotRenderer.RtBinCv(raw, corr, rt, new[] { 0, 1, 2, 3 }, "RT-binned CV", "#d62728"));
    }

    [Fact]
    public void RtBinBoxplot_Renders()
    {
        var m = new double[60, 4];
        var rt = new double[60];
        for (var f = 0; f < 60; f++)
        {
            rt[f] = f;
            for (var s = 0; s < 4; s++)
                m[f, s] = 12 + 0.03 * f + 0.1 * s + (f % 7 - 3) * 0.2;
        }
        AssertPng(PlotRenderer.RtBinBoxplot(m, rt, "Abundance by RT bin", "#1f77b4"));
    }
}
