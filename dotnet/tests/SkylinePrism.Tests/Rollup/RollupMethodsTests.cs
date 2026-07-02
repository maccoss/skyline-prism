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
    public void TopN_CorrelationVsIntensitySelection_PickDifferentTransitions()
    {
        // t0,t1: low intensity (2^3) but high shape corr; t2,t3: high intensity (2^5) but low corr.
        var log2 = new double[,] { { 3, 3 }, { 3, 3 }, { 5, 5 }, { 5, 5 } };
        var shape = new double[,] { { 0.99, 0.99 }, { 0.99, 0.99 }, { 0.1, 0.1 }, { 0.1, 0.1 } };

        // Correlation selection -> t0,t1 -> sum 8+8=16 -> log2 16 = 4.
        var byCorr = TopNRollup.Compute(log2, shape, n: 2, minTransitions: 1, selection: "correlation", weighting: "sum");
        Assert.Equal(4.0, byCorr[0], 9);

        // Intensity selection -> t2,t3 -> sum 32+32=64 -> log2 64 = 6.
        var byInt = TopNRollup.Compute(log2, null, n: 2, minTransitions: 1, selection: "intensity", weighting: "sum");
        Assert.Equal(6.0, byInt[0], 9);
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
    public void Consensus_AgreeingTransitions_MatchSum()
    {
        // When all transitions share the same pattern, inverse-variance weights are equal, so
        // consensus == the sum-method result.
        var m = new double[,] { { 10, 11 }, { 10, 11 }, { 10, 11 } };
        var consensus = new ConsensusRollup(minTransitions: 1).Aggregate(m);
        var sum = new SumRollup().Aggregate(m);
        Assert.Equal(sum[0], consensus[0], 9);
        Assert.Equal(sum[1], consensus[1], 9);
    }

    [Fact]
    public void Consensus_DownweightsDeviantTransition()
    {
        // t2 is wildly high in sample 1 only -> consensus down-weights it, so its sample-1 value is
        // far below the plain sum (which the outlier dominates).
        var m = new double[,] { { 10, 11 }, { 10, 11 }, { 10, 20 } };
        var consensus = new ConsensusRollup(minTransitions: 1).Aggregate(m);
        var sum = new SumRollup().Aggregate(m);
        Assert.True(consensus[1] < sum[1] - 1.0, $"consensus {consensus[1]} should be well below sum {sum[1]}");
        Assert.Equal(sum[0], consensus[0], 6); // sample 0 unaffected (all agree there)
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
