using System;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.BatchCorrection;

/// <summary>
/// The ComBat auto-revert decision (ports evaluate_batch_correction): revert when the primary control
/// CV (QC preferred, else reference) worsens by >10%; not evaluable without a control.
/// </summary>
public class BatchCorrectionEvaluatorTests
{
    // MedianCv converts log2 -> linear internally, so set feature values from linear numbers.
    private static void SetLinear(double[,] m, int feature, int[] cols, double[] linear)
    {
        for (var k = 0; k < cols.Length; k++)
            m[feature, cols[k]] = Math.Log2(linear[k]);
    }

    private static (double[,] tight, double[,] spread) Fixtures()
    {
        var tight = new double[3, 4];   // near-constant per feature -> low CV
        var spread = new double[3, 4];  // wide spread -> high CV
        for (var f = 0; f < 3; f++)
        {
            SetLinear(tight, f, new[] { 0, 1, 2, 3 }, new[] { 100.0, 101, 99, 100 });
            SetLinear(spread, f, new[] { 0, 1, 2, 3 }, new[] { 60.0, 140, 80, 120 });
        }
        return (tight, spread);
    }

    [Fact]
    public void Revert_WhenControlCvWorsens()
    {
        var (tight, spread) = Fixtures();
        var d = BatchCorrectionEvaluator.Evaluate(tight, spread, new[] { 0, 1, 2, 3 }, Array.Empty<int>());
        Assert.True(d.Evaluable);
        Assert.Equal("QC", d.ControlName);
        Assert.True(d.Revert);
        Assert.True(d.ControlCvAfter > d.ControlCvBefore);
    }

    [Fact]
    public void NoRevert_WhenControlCvImproves()
    {
        var (tight, spread) = Fixtures();
        var d = BatchCorrectionEvaluator.Evaluate(spread, tight, new[] { 0, 1, 2, 3 }, Array.Empty<int>());
        Assert.False(d.Revert);
    }

    [Fact]
    public void FallsBackToReference_WhenNoQc()
    {
        var (tight, spread) = Fixtures();
        var d = BatchCorrectionEvaluator.Evaluate(tight, spread, Array.Empty<int>(), new[] { 0, 1, 2, 3 });
        Assert.Equal("reference", d.ControlName);
        Assert.True(d.Revert);
    }

    [Fact]
    public void NotEvaluable_WithoutTwoControls()
    {
        var (tight, spread) = Fixtures();
        var d = BatchCorrectionEvaluator.Evaluate(tight, spread, Array.Empty<int>(), new[] { 0 });
        Assert.False(d.Evaluable);
        Assert.False(d.Revert);
    }
}
