using System;
using SkylinePrism.Core.Rollup;
using Xunit;

namespace SkylinePrism.Tests.Rollup;

/// <summary>
/// Mirrors tests/test_transition_rollup.py: exact sum==log2(sum(2^m)) parity, the
/// preprocessing impute value, and the peptide-stage +log2(n) offset.
/// </summary>
public class TransitionRollupTests
{
    [Fact]
    public void Preprocess_ImputeValue_MatchesPython()
    {
        // clip -5 -> 0; positives {100,200,50}; percentile(_,1)=51.0; impute=max(25.5,1)=25.5.
        var linear = new double[,]
        {
            { 100.0, -5.0, double.NaN },
            { 0.0, 200.0, 50.0 },
        };
        var result = RollupPreprocess.ImputeAndLog2(linear);

        Assert.Equal(25.5, result.ImputeValue, 12);
        // NaN and 0 cells imputed to log2(25.5); real cells log2 of themselves.
        Assert.Equal(Math.Log2(100.0), result.Log2Matrix[0, 0], 12);
        Assert.Equal(Math.Log2(25.5), result.Log2Matrix[0, 1], 12); // was 0 -> impute -- wait clip
        Assert.Equal(Math.Log2(25.5), result.Log2Matrix[0, 2], 12); // was NaN -> impute
        Assert.Equal(Math.Log2(25.5), result.Log2Matrix[1, 0], 12); // was 0 -> impute
        Assert.Equal(Math.Log2(200.0), result.Log2Matrix[1, 1], 12);
        Assert.Equal(Math.Log2(50.0), result.Log2Matrix[1, 2], 12);
    }

    [Fact]
    public void Sum_EqualsLog2OfLinearSum_Exact()
    {
        // Python asserts sum == np.log2(linear.sum(axis=0)) to decimal=10.
        var log2 = new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
        };
        var result = new SumRollup().Aggregate(log2);

        var expected0 = Math.Log2(Math.Pow(2.0, 1.0) + Math.Pow(2.0, 3.0));
        var expected1 = Math.Log2(Math.Pow(2.0, 2.0) + Math.Pow(2.0, 4.0));
        Assert.Equal(expected0, result[0], 10);
        Assert.Equal(expected1, result[1], 10);
    }

    [Fact]
    public void Sum_ClipsLowerBoundToOne()
    {
        // Very small linear sum (< 1) is clipped so log2 stays >= 0.
        var log2 = new double[,] { { -50.0 } }; // 2^-50 ~ 8.9e-16, sum < 1
        var result = new SumRollup().Aggregate(log2);
        Assert.Equal(0.0, result[0], 12); // log2(clip(<1, min=1)) = log2(1) = 0
    }

    [Fact]
    public void MedianPolish_PeptideStage_AddsLog2NOffset()
    {
        var matrix = new double[,]
        {
            { 15.77, 16.84, 17.16 },
            { 13.46, 14.92, 15.36 },
            { 15.20, 15.43, 17.09 },
            { 14.63, 16.36, 16.41 },
            { 13.74, 14.62, 14.98 },
        };

        var withOffset = new MedianPolishRollup(addLog2NOffset: true).Aggregate(matrix);
        var withoutOffset = new MedianPolishRollup(addLog2NOffset: false).Aggregate(matrix);

        var expectedOffset = Math.Log2(5.0);
        for (var j = 0; j < 3; j++)
            Assert.Equal(expectedOffset, withOffset[j] - withoutOffset[j], 12);
    }

    [Fact]
    public void MedianPolish_NoOffset_EqualsTukeyColEffects()
    {
        var matrix = new double[,]
        {
            { 15.77, 16.84, 17.16 },
            { 13.46, 14.92, 15.36 },
            { 15.20, 15.43, 17.09 },
        };
        var polish = TukeyMedianPolish.Run(matrix);
        var rollup = new MedianPolishRollup(addLog2NOffset: false).Aggregate(matrix);
        for (var j = 0; j < 3; j++)
            Assert.Equal(polish.ColEffects[j], rollup[j], 12);
    }

    [Fact]
    public void MedianPolish_BelowMinTransitions_ReturnsNaN()
    {
        var matrix = new double[,] { { 10.0, 11.0, 12.0 } }; // 1 transition
        var rollup = new MedianPolishRollup(addLog2NOffset: true, minTransitions: 2).Aggregate(matrix);
        Assert.All(rollup, v => Assert.True(double.IsNaN(v)));
    }
}
