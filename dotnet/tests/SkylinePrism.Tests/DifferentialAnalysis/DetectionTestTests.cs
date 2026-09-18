using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Parity tests for <see cref="FisherExact"/> (vs scipy.stats.fisher_exact) and
/// <see cref="DetectionTest"/> (vs the explorer's detection_test).
/// </summary>
public class DetectionTestTests
{
    [Theory]
    [InlineData(3, 0, 0, 3, 0.10000000000000002)]
    [InlineData(2, 1, 1, 2, 1.0)]
    [InlineData(5, 0, 2, 3, 0.16666666666666666)]
    [InlineData(4, 1, 0, 4, 0.04761904761904762)]
    [InlineData(3, 3, 3, 3, 1.0)]
    [InlineData(6, 0, 0, 2, 0.03571428571428571)]
    public void FisherExact_MatchesScipy(int a, int b, int c, int d, double expected)
    {
        Assert.Equal(expected, FisherExact.TwoSidedP(a, b, c, d), 12);
    }

    [Fact]
    public void FisherExact_LargeCounts_MatchScipy()
    {
        // Large tables (log-gamma path); compared relatively since the p-values are tiny.
        static void Rel(double expected, double actual) =>
            Assert.True(System.Math.Abs(actual - expected) / expected <= 1e-9,
                $"expected {expected:R}, actual {actual:R}");
        Rel(1.1885236381381294e-19, FisherExact.TwoSidedP(150, 50, 60, 140));
        Rel(1.1422937075022777e-192, FisherExact.TwoSidedP(1000, 200, 300, 900));
    }

    [Fact]
    public void DetectionTest_CountsAndAdjustsAndSorts()
    {
        // pep0 detected in all of A, none of B -> [[3,0],[0,3]] -> p 0.1.
        // pep1 detected 2/3 A, 1/3 B        -> [[2,1],[1,2]] -> p 1.0.
        var det = new[,]
        {
            { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 },
            { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 },
        };
        var rows = DetectionTest.Run(det, new[] { "pep0", "pep1" }, new[] { 0, 1, 2 }, new[] { 3, 4, 5 });

        Assert.Equal("pep0", rows[0].PeptideId); // smaller p sorts first
        Assert.Equal(3, rows[0].DetA);
        Assert.Equal(0, rows[0].DetB);
        Assert.Equal(1.0, rows[0].RateA, 12);
        Assert.Equal(0.0, rows[0].RateB, 12);
        Assert.Equal(0.10000000000000002, rows[0].P, 12);
        Assert.Equal(0.20000000000000004, rows[0].Q, 12); // BH of [0.1, 1.0] -> [0.2, 1.0]

        var pep1 = rows.Single(r => r.PeptideId == "pep1");
        Assert.Equal(1.0, pep1.P, 12);
        Assert.Equal(1.0, pep1.Q, 12);
    }
}
