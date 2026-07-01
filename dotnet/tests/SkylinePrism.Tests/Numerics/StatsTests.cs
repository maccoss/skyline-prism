using System;
using SkylinePrism.Core.Numerics;
using Xunit;

namespace SkylinePrism.Tests.Numerics;

/// <summary>
/// Parity micro-tests for <see cref="Stats"/>. Expected values are hand-computed and
/// cross-checked against numpy / scipy so these lock the primitives that everything
/// downstream (rollup, ComBat) depends on.
/// </summary>
public class StatsTests
{
    [Fact]
    public void NanMedian_OddCount_ReturnsMiddle()
    {
        Assert.Equal(2.0, Stats.NanMedian(new[] { 1.0, 2.0, 3.0 }));
    }

    [Fact]
    public void NanMedian_EvenCount_AveragesTwoMiddle()
    {
        // numpy.median([1,2,3,4]) == 2.5
        Assert.Equal(2.5, Stats.NanMedian(new[] { 1.0, 2.0, 3.0, 4.0 }));
    }

    [Fact]
    public void NanMedian_SkipsNaN()
    {
        // Non-NaN values {1,2,4} -> median 2.0
        Assert.Equal(2.0, Stats.NanMedian(new[] { 1.0, 2.0, double.NaN, 4.0 }));
    }

    [Fact]
    public void NanMedian_AllNaN_ReturnsNaN()
    {
        Assert.True(double.IsNaN(Stats.NanMedian(new[] { double.NaN, double.NaN })));
    }

    [Fact]
    public void Median_PropagatesNaN()
    {
        Assert.True(double.IsNaN(Stats.Median(new[] { 1.0, double.NaN, 3.0 })));
    }

    [Fact]
    public void PercentileLinear_MatchesNumpy_Percentile1()
    {
        // numpy.percentile([1,2,3,4], 1) == 1.03 (index 0.01*(4-1)=0.03)
        Assert.Equal(1.03, Stats.PercentileLinear(new[] { 1.0, 2.0, 3.0, 4.0 }, 1), 12);
    }

    [Fact]
    public void PercentileLinear_MatchesNumpy_Quartile()
    {
        // numpy.percentile([1,2,3,4,5], 25) == 2.0 (index 0.25*4 = 1.0)
        Assert.Equal(2.0, Stats.PercentileLinear(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }, 25), 12);
    }

    [Fact]
    public void PercentileLinear_Median50Matches()
    {
        Assert.Equal(2.5, Stats.PercentileLinear(new[] { 1.0, 2.0, 3.0, 4.0 }, 50), 12);
    }

    [Fact]
    public void Var_Ddof1_MatchesNumpy()
    {
        // numpy.var([1,2,3,4], ddof=1) == 1.6666666666666667
        Assert.Equal(5.0 / 3.0, Stats.Var(new[] { 1.0, 2.0, 3.0, 4.0 }, ddof: 1), 12);
    }

    [Fact]
    public void Var_Ddof0_MatchesNumpy()
    {
        // numpy.var([1,2,3,4]) == 1.25
        Assert.Equal(1.25, Stats.Var(new[] { 1.0, 2.0, 3.0, 4.0 }), 12);
    }

    [Fact]
    public void NanVar_Ddof1_SkipsNaN()
    {
        // Non-NaN {1,2,3,4} -> var ddof=1 = 5/3
        Assert.Equal(5.0 / 3.0, Stats.NanVar(new[] { 1.0, 2.0, double.NaN, 3.0, 4.0 }, ddof: 1), 12);
    }

    [Fact]
    public void RankAverage_TiesGetAverageRank()
    {
        // scipy.stats.rankdata([10,20,20,30]) == [1, 2.5, 2.5, 4]
        var ranks = Stats.RankAverage(new[] { 10.0, 20.0, 20.0, 30.0 });
        Assert.Equal(new[] { 1.0, 2.5, 2.5, 4.0 }, ranks);
    }

    [Fact]
    public void RankAverage_NoTies()
    {
        // scipy.stats.rankdata([3,1,2]) == [3,1,2]
        var ranks = Stats.RankAverage(new[] { 3.0, 1.0, 2.0 });
        Assert.Equal(new[] { 3.0, 1.0, 2.0 }, ranks);
    }

    [Fact]
    public void Interp_MidPoint()
    {
        Assert.Equal(25.0, Stats.Interp(2.5, new[] { 1.0, 2.0, 3.0 }, new[] { 10.0, 20.0, 30.0 }), 12);
    }

    [Fact]
    public void Interp_ClampsBelow()
    {
        Assert.Equal(10.0, Stats.Interp(0.0, new[] { 1.0, 2.0, 3.0 }, new[] { 10.0, 20.0, 30.0 }));
    }

    [Fact]
    public void Interp_ClampsAbove()
    {
        Assert.Equal(30.0, Stats.Interp(5.0, new[] { 1.0, 2.0, 3.0 }, new[] { 10.0, 20.0, 30.0 }));
    }

    [Fact]
    public void NLargestIndices_KeepFirstOnTies()
    {
        // pandas Series([5,3,5,1]).nlargest(2) keeps first occurrences -> indices [0,2]
        var idx = Stats.NLargestIndices(new[] { 5.0, 3.0, 5.0, 1.0 }, 2);
        Assert.Equal(new[] { 0, 2 }, idx);
    }

    [Fact]
    public void ArgSort_Stable()
    {
        // numpy.argsort([3,1,2,1], kind="stable") == [1,3,2,0]
        var idx = Stats.ArgSort(new[] { 3.0, 1.0, 2.0, 1.0 });
        Assert.Equal(new[] { 1, 3, 2, 0 }, idx);
    }
}
