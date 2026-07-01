using System;
using System.Linq;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Core.Numerics;
using Xunit;

namespace SkylinePrism.Tests.Normalization;

/// <summary>
/// RT-lowess normalization (functional-parity). LOWESS recovers a linear trend exactly, and
/// RT-lowess removes per-sample RT-dependent offsets, aligning samples to the global median
/// RT profile.
/// </summary>
public class RtLowessTests
{
    [Fact]
    public void Lowess_RecoversLinearTrend()
    {
        var x = Enumerable.Range(0, 30).Select(i => (double)i).ToArray();
        var y = x.Select(v => 3.0 * v + 5.0).ToArray();
        var fit = Lowess.Fit(x, y, frac: 0.3);
        for (var i = 0; i < x.Length; i++)
            Assert.Equal(y[i], fit[i], 6); // local linear fit is exact on linear data
    }

    [Fact]
    public void RtLowess_RemovesPerSampleRtTrend()
    {
        // 30 peptides at RT 0..29; a non-RT peptide pattern shared by all samples, plus a
        // sample-specific RT slope (0, +0.05*rt, -0.05*rt). Median slope is 0, so after
        // normalization every sample should collapse to the shared pattern.
        const int nPep = 30, nSamp = 3;
        var rt = new double[nPep];
        var m = new double[nPep, nSamp];
        double[] slope = { 0.0, 0.05, -0.05 };
        for (var i = 0; i < nPep; i++)
        {
            rt[i] = i;
            var basePattern = 12.0 + (i % 5 - 2) * 0.4; // small, sub-smoothing-scale signal
            for (var s = 0; s < nSamp; s++)
                m[i, s] = basePattern + slope[s] * rt[i];
        }

        var spreadBefore = MaxCrossSampleSpread(m, nPep, nSamp);
        var norm = Normalizer.RtLowessNormalize(m, rt);
        var spreadAfter = MaxCrossSampleSpread(norm, nPep, nSamp);

        Assert.True(spreadBefore > 1.0, $"expected a strong RT trend before, got {spreadBefore}");
        Assert.True(spreadAfter < 0.35 * spreadBefore,
            $"RT-lowess should flatten the trend: before={spreadBefore}, after={spreadAfter}");
    }

    private static double MaxCrossSampleSpread(double[,] m, int nPep, int nSamp)
    {
        double max = 0;
        for (var i = 0; i < nPep; i++)
        {
            double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
            for (var s = 0; s < nSamp; s++)
            {
                var v = m[i, s];
                if (double.IsNaN(v)) continue;
                lo = Math.Min(lo, v);
                hi = Math.Max(hi, v);
            }
            if (hi >= lo)
                max = Math.Max(max, hi - lo);
        }
        return max;
    }
}
