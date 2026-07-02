using System;
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
}
