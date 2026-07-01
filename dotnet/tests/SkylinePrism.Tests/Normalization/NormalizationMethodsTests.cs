using System;
using System.Linq;
using SkylinePrism.Core.Normalization;
using Xunit;

namespace SkylinePrism.Tests.Normalization;

/// <summary>Quantile and VSN peptide normalization.</summary>
public class NormalizationMethodsTests
{
    [Fact]
    public void Quantile_MakesEveryColumnShareTheReferenceDistribution()
    {
        // No within-column ties -> each normalized column's sorted values equal the reference
        // (per-rank mean of the sorted columns).
        var m = new double[,]
        {
            { 5, 3, 7 },
            { 2, 1, 8 },
            { 3, 6, 5 },
            { 4, 2, 9 },
        };
        var reference = new[] { 8.0 / 3, 4.0, 5.0, 20.0 / 3 };

        var norm = Normalizer.QuantileNormalize(m);

        for (var j = 0; j < 3; j++)
        {
            var col = Enumerable.Range(0, 4).Select(i => norm[i, j]).OrderBy(x => x).ToArray();
            for (var k = 0; k < 4; k++)
                Assert.Equal(reference[k], col[k], 9);
        }
        // g1 is the max in A1 (-> 20/3), the mid-high in A2, the mid in A3.
        Assert.Equal(20.0 / 3, norm[0, 0], 9);
    }

    [Fact]
    public void Vsn_AppliesArcsinhOverMedian()
    {
        // log2 input [0,1,2,3] -> linear [1,2,4,8], median 3, a=1/3, arcsinh(a*linear).
        var m = new double[,] { { 0 }, { 1 }, { 2 }, { 3 } };
        var norm = Normalizer.VsnNormalize(m);
        double[] linear = { 1, 2, 4, 8 };
        for (var i = 0; i < 4; i++)
            Assert.Equal(Math.Asinh(linear[i] / 3.0), norm[i, 0], 9);
    }

    [Fact]
    public void Quantile_PreservesNaN()
    {
        var m = new double[,] { { 5, double.NaN }, { 2, 1 }, { 3, 4 } };
        var norm = Normalizer.QuantileNormalize(m);
        Assert.True(double.IsNaN(norm[0, 1]));
        Assert.False(double.IsNaN(norm[0, 0]));
    }
}
