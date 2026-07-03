using System;
using System.Diagnostics;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Visualization;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// The RT-LOWESS QC plot re-fits a curve per sample; done over every peptide it is O(n * frac*n * iters)
/// and dominated the whole QC report. PlotRenderer.RtLowessCurves now strides the fit input down, so a
/// large matrix must still render quickly. The Lowess.Fit sizing test documents why (3k vs 40k points).
/// </summary>
public class RtLowessPerfTests
{
    private readonly ITestOutputHelper _out;
    public RtLowessPerfTests(ITestOutputHelper o) => _out = o;

    private static double Frac(long seed) => ((seed & 0x7FFFFFFFL) / (double)0x7FFFFFFF);

    [Fact]
    public void RtLowessCurves_LargeInput_RendersQuickly()
    {
        const int nF = 40000, nS = 4;
        var m = new double[nF, nS];
        var rt = new double[nF];
        long seed = 12345;
        for (var f = 0; f < nF; f++)
        {
            seed = seed * 6364136223846793005L + 1442695040888963407L;
            rt[f] = Frac(seed) * 120.0;
            for (var s = 0; s < nS; s++)
            {
                seed = seed * 6364136223846793005L + 1442695040888963407L;
                m[f, s] = 20.0 + Math.Sin(rt[f] / 10.0) + Frac(seed);
            }
        }
        var types = new[] { "reference", "reference", "qc", "qc" };

        var sw = Stopwatch.StartNew();
        var png = PlotRenderer.RtLowessCurves(m, rt, types, "perf");
        sw.Stop();

        _out.WriteLine($"RtLowessCurves 40k x 4: {sw.ElapsedMilliseconds} ms, {png.Length} bytes");
        Assert.True(png.Length > 0);
        // Un-subsampled this is tens of seconds; the strided fit keeps it well under this generous bound.
        Assert.True(sw.ElapsedMilliseconds < 8000, $"RtLowessCurves too slow: {sw.ElapsedMilliseconds} ms");
    }

    [Fact]
    public void LowessFit_CostGrowsWithInputSize()
    {
        foreach (var n in new[] { 3000, 40000 })
        {
            var x = new double[n];
            var y = new double[n];
            for (var i = 0; i < n; i++)
            {
                x[i] = i * (120.0 / n);
                y[i] = Math.Sin(i * (12.0 / n));
            }
            var sw = Stopwatch.StartNew();
            Lowess.Fit(x, y, 0.3);
            sw.Stop();
            _out.WriteLine($"Lowess.Fit n={n}: {sw.ElapsedMilliseconds} ms");
        }
    }
}
