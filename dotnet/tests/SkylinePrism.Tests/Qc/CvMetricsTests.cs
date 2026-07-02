using System;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>CV metrics on the linear scale: NaN-feature skipping, all-NaN, and the before/after edges.</summary>
public class CvMetricsTests
{
    // feature 0 = linear [10, 12] (CV = sqrt(2)/11*100); feature 1 has a NaN -> skipped.
    private static readonly double[,] Matrix =
    {
        { 3.321928094887362, 3.584962500721156 }, // log2(10), log2(12)
        { double.NaN, 4.321928094887363 },        // NaN, log2(20)
    };

    private static readonly double ExpectedCv = Math.Sqrt(2.0) / 11.0 * 100.0;

    [Fact]
    public void MedianCv_SkipsFeaturesWithNaN()
    {
        var cv = CvMetrics.MedianCv(Matrix, new[] { 0, 1 });
        Assert.Equal(ExpectedCv, cv, 6); // only the NaN-free feature contributes
    }

    [Fact]
    public void MedianCv_AllFeaturesHaveNaN_ReturnsNaN()
    {
        var m = new double[,] { { double.NaN, 1.0 }, { 2.0, double.NaN } };
        Assert.True(double.IsNaN(CvMetrics.MedianCv(m, new[] { 0, 1 })));
    }

    [Fact]
    public void PerFeatureCvs_DropsNaNFeatures()
    {
        var cvs = CvMetrics.PerFeatureCvs(Matrix, new[] { 0, 1 });
        Assert.Single(cvs);
        Assert.Equal(ExpectedCv, cvs[0], 6);
    }

    [Fact]
    public void Compute_ReturnsNull_WhenFewerThanTwoSamples()
    {
        var m = new double[,] { { 1.0 } };
        Assert.Null(CvMetrics.Compute(m, m, new[] { 0 }));
    }

    [Fact]
    public void Compute_ReturnsBeforeAfter_WhenEnoughSamples()
    {
        var ba = CvMetrics.Compute(Matrix, Matrix, new[] { 0, 1 });
        Assert.NotNull(ba);
        Assert.Equal(ba!.Value.Before, ba.Value.After, 9); // same matrix -> identical
    }

    [Fact]
    public void ImprovementPercent_HandlesZeroBefore()
    {
        Assert.Equal(50.0, new CvMetrics.BeforeAfter(20, 10).ImprovementPercent, 9);
        Assert.Equal(0.0, new CvMetrics.BeforeAfter(0, 10).ImprovementPercent, 9); // Before <= 0 -> 0
    }
}
