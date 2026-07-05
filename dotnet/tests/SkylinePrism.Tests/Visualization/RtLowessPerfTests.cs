using System;
using System.Diagnostics;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Visualization;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// RT-LOWESS (normalization and the QC plot) fits a curve per sample. Un-delta'd this is
/// O(n * frac*n * iters) and dominated the whole QC report. Lowess.Fit now supports the statsmodels
/// delta speedup (delta = 1% of range), as the Python pipeline uses: fit anchors, interpolate between.
/// These tests document the speedup and check the delta curve stays close to the full fit.
/// </summary>
public class RtLowessPerfTests
{
    private readonly ITestOutputHelper _out;
    public RtLowessPerfTests(ITestOutputHelper o) => _out = o;

    private static double Frac(long seed) => (seed & 0x7FFFFFFFL) / (double)0x7FFFFFFF;

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
        // Un-delta'd this is tens of seconds; the delta fit keeps it well under this generous bound.
        Assert.True(sw.ElapsedMilliseconds < 8000, $"RtLowessCurves too slow: {sw.ElapsedMilliseconds} ms");
    }

    [Fact]
    public void LowessFit_DeltaMatchesFullFit_AndIsFaster()
    {
        const int n = 40000;
        var x = new double[n];
        var y = new double[n];
        long seed = 7;
        for (var i = 0; i < n; i++)
        {
            x[i] = i * (120.0 / n);
            seed = seed * 6364136223846793005L + 1442695040888963407L;
            y[i] = Math.Sin(x[i] / 10.0) + (Frac(seed) - 0.5) * 0.2;
        }

        var swFull = Stopwatch.StartNew();
        var full = Lowess.Fit(x, y, 0.3);
        swFull.Stop();

        var delta = (x[^1] - x[0]) * 0.01;
        var swDelta = Stopwatch.StartNew();
        var fast = Lowess.Fit(x, y, 0.3, delta: delta);
        swDelta.Stop();

        double maxDiff = 0;
        for (var i = 0; i < n; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(full[i] - fast[i]));

        _out.WriteLine($"Lowess.Fit 40k: full={swFull.ElapsedMilliseconds} ms, delta={swDelta.ElapsedMilliseconds} ms, max|full-delta|={maxDiff:F4}");

        // The delta curve interpolates within 1% of the RT range, so it tracks the full fit closely...
        Assert.True(maxDiff < 0.05, $"delta curve diverges from full fit: {maxDiff}");
        // ...and is dramatically cheaper (guard against the delta path regressing to the full cost).
        Assert.True(swDelta.ElapsedMilliseconds < 2000, $"delta fit too slow: {swDelta.ElapsedMilliseconds} ms");
    }
}
