using SkylinePrism.Core.Normalization;
using SkylinePrism.Core.Numerics;
using Xunit;

namespace SkylinePrism.Tests.Normalization;

/// <summary>
/// Layer 4 unit tests: median normalization equalizes every sample median to the global
/// median (mirrors tests/test_scale_handling.py), and one-sided outlier detection flags
/// low-signal samples on the linear scale.
/// </summary>
public class NormalizationTests
{
    [Fact]
    public void MedianNormalize_EqualizesSampleMedians()
    {
        // Sample medians 2, 6, 11 -> global median 6; every column median becomes 6.
        var m = new double[,]
        {
            { 1.0, 5.0, 10.0 },
            { 2.0, 6.0, 11.0 },
            { 3.0, 7.0, 12.0 },
        };

        var result = Normalizer.MedianNormalize(m);

        for (var j = 0; j < 3; j++)
        {
            var col = new[] { result[0, j], result[1, j], result[2, j] };
            Assert.Equal(6.0, Stats.NanMedian(col), 9);
        }
        // Column 1 (median already 6) is unchanged.
        Assert.Equal(5.0, result[0, 1], 9);
        Assert.Equal(6.0, result[1, 1], 9);
    }

    [Fact]
    public void MedianNormalize_PreservesNaN()
    {
        var m = new double[,]
        {
            { 1.0, double.NaN },
            { 2.0, 6.0 },
            { 3.0, 7.0 },
        };
        var result = Normalizer.MedianNormalize(m);
        Assert.True(double.IsNaN(result[0, 1]));
    }

    [Fact]
    public void Outlier_Iqr_FlagsLowSignalSample()
    {
        // Samples A,B,C at log2=10 (linear 1024); D at log2=2 (linear 4).
        var m = new double[,]
        {
            { 10.0, 10.0, 10.0, 2.0 },
            { 10.0, 10.0, 10.0, 2.0 },
            { 10.0, 10.0, 10.0, 2.0 },
        };
        var samples = new[] { "A", "B", "C", "D" };

        var result = OutlierDetector.Detect(m, samples, OutlierDetector.Method.Iqr);
        Assert.Equal(new[] { "D" }, result.Outliers);
    }

    [Fact]
    public void Outlier_FoldMedian_FlagsLowSignalSample()
    {
        var m = new double[,]
        {
            { 10.0, 10.0, 10.0, 2.0 },
            { 10.0, 10.0, 10.0, 2.0 },
        };
        var samples = new[] { "A", "B", "C", "D" };

        var result = OutlierDetector.Detect(m, samples, OutlierDetector.Method.FoldMedian, foldThreshold: 0.1);
        Assert.Equal(new[] { "D" }, result.Outliers);
        Assert.Equal(1024.0, result.OverallMedian, 9);
    }

    [Fact]
    public void Outlier_NoneFlaggedWhenUniform()
    {
        var m = new double[,]
        {
            { 10.0, 10.1, 9.9, 10.05 },
            { 10.0, 10.1, 9.9, 10.05 },
        };
        var samples = new[] { "A", "B", "C", "D" };
        var result = OutlierDetector.Detect(m, samples, OutlierDetector.Method.Iqr);
        Assert.Empty(result.Outliers);
    }
}
