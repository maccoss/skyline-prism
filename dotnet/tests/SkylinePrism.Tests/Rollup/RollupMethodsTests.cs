using System;
using SkylinePrism.Core.Rollup;
using Xunit;

namespace SkylinePrism.Tests.Rollup;

/// <summary>topN (transition + protein) and maxLFQ rollup methods.</summary>
public class RollupMethodsTests
{
    [Fact]
    public void ProteinTopN_SelectsHighestMedianPeptides_ThenMeans()
    {
        // Peptide medians: p0=10, p1=8, p2=12, p3=6. Top-2 => p2, p0; per-sample mean = 11.
        var m = new double[,]
        {
            { 10, 10 },
            { 8, 8 },
            { 12, 12 },
            { 6, 6 },
        };
        var abund = ProteinMatrixRollup.Aggregate(m, ProteinRollupMethod.TopN, minPeptides: 1, topN: 2);
        Assert.Equal(11.0, abund[0], 9);
        Assert.Equal(11.0, abund[1], 9);
    }

    [Fact]
    public void MaxLfq_RecoversRelativeProfileAnchoredToOverall()
    {
        // Both peptides are +2 in sample 1; overall level (median of per-sample medians) = 10.
        var m = new double[,]
        {
            { 10, 12 },
            { 8, 10 },
        };
        var abund = ProteinMatrixRollup.Aggregate(m, ProteinRollupMethod.MaxLfq, minPeptides: 1);
        Assert.Equal(9.0, abund[0], 9);
        Assert.Equal(11.0, abund[1], 9);
    }

    [Fact]
    public void Ibaq_DividesSummedIntensityByTheoreticalCount()
    {
        // 2 peptides at log2=3 -> linear 8 each, sum 16; iBAQ = log2(16 / 10).
        var m = new double[,] { { 3 }, { 3 } };
        var abund = ProteinMatrixRollup.Aggregate(
            m, ProteinRollupMethod.Ibaq, minPeptides: 1, topN: 3, nTheoretical: 10);
        Assert.Equal(Math.Log2(16.0 / 10.0), abund[0], 9);
    }

    [Fact]
    public void TransitionTopN_SumsTopIntensityTransitions()
    {
        // Mean linear intensities: t0=1, t1=4, t2=2. Top-2 => t1,t2; sum linear = 6 -> log2(6).
        var m = new double[,]
        {
            { 0, 0 },
            { 2, 2 },
            { 1, 1 },
        };
        var abund = new TopNRollup(n: 2, minTransitions: 1).Aggregate(m);
        Assert.Equal(Math.Log2(6.0), abund[0], 9);
        Assert.Equal(Math.Log2(6.0), abund[1], 9);
    }
}
