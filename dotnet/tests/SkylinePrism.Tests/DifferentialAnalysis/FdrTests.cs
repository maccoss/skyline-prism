using System;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Parity micro-tests for <see cref="Fdr"/>. Expected values are the golden numbers encoded in
/// the Python reference's tests/test_significance.py (statsmodels multipletests fdr_bh), so these
/// lock BH parity with the Streamlit explorer.
/// </summary>
public class FdrTests
{
    [Fact]
    public void BenjaminiHochberg_EqualSteps_AllTakeMaxAdjusted()
    {
        // multipletests([0.01,0.02,0.03,0.04], method="fdr_bh")[1] == [0.04,0.04,0.04,0.04]
        var adj = Fdr.BenjaminiHochberg(new[] { 0.01, 0.02, 0.03, 0.04 });
        foreach (var v in adj)
            Assert.Equal(0.04, v, 12);
    }

    [Fact]
    public void BenjaminiHochberg_OneStrongHit_MatchesReference()
    {
        // multipletests([0.001,0.5,0.5,0.5], method="fdr_bh")[1] == [0.004,0.5,0.5,0.5]
        var adj = Fdr.BenjaminiHochberg(new[] { 0.001, 0.5, 0.5, 0.5 });
        Assert.Equal(0.004, adj[0], 12);
        Assert.Equal(0.5, adj[1], 12);
        Assert.Equal(0.5, adj[2], 12);
        Assert.Equal(0.5, adj[3], 12);
    }

    [Fact]
    public void BenjaminiHochberg_PreservesNaN_AndExcludesFromCount()
    {
        // NaN is passed through and does not count toward m: the three finite values are corrected
        // as m=3. multipletests-style: [0.01,0.02,0.03] with m=3 -> all 0.03.
        var adj = Fdr.BenjaminiHochberg(new[] { 0.01, double.NaN, 0.02, 0.03 });
        Assert.Equal(0.03, adj[0], 12);
        Assert.True(double.IsNaN(adj[1]));
        Assert.Equal(0.03, adj[2], 12);
        Assert.Equal(0.03, adj[3], 12);
    }

    [Fact]
    public void BenjaminiHochberg_AllNaN_ReturnsAllNaN()
    {
        var adj = Fdr.BenjaminiHochberg(new[] { double.NaN, double.NaN });
        Assert.True(double.IsNaN(adj[0]));
        Assert.True(double.IsNaN(adj[1]));
    }

    [Fact]
    public void BenjaminiHochberg_ClipsAboveOne()
    {
        // [0.9,0.95]: ranked [1.8, 0.95] -> reverse-cummin [0.95, 0.95] -> clipped [0.95, 0.95].
        var adj = Fdr.BenjaminiHochberg(new[] { 0.9, 0.95 });
        Assert.Equal(0.95, adj[0], 12);
        Assert.Equal(0.95, adj[1], 12);
    }

    [Fact]
    public void BenjaminiHochberg_Ties_MatchReference()
    {
        // Bit-identical to the Python _bh on repeated p-values (note the 0.8999999999999999).
        var adj = Fdr.BenjaminiHochberg(new[]
            { 0.013, 0.0007, 0.21, 0.013, 0.5, 0.0007, 0.9, 0.31, 0.013 });
        var expected = new[]
            { 0.0234, 0.00315, 0.315, 0.0234, 0.5625, 0.00315, 0.8999999999999999, 0.3985714285714286, 0.0234 };
        for (var i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], adj[i], 12);
    }

    [Fact]
    public void BenjaminiHochberg_InfiniteAndAboveOne_ClampToOne()
    {
        // Infinity is not NaN, so it stays in the correction; it and any >1 value clamp to 1.
        var adj = Fdr.BenjaminiHochberg(new[] { 0.01, double.PositiveInfinity, 0.02, 1.5 });
        Assert.Equal(0.04, adj[0], 12);
        Assert.Equal(1.0, adj[1], 12);
        Assert.Equal(0.04, adj[2], 12);
        Assert.Equal(1.0, adj[3], 12);
    }

    [Fact]
    public void BenjaminiHochberg_Empty_ReturnsEmpty()
    {
        Assert.Empty(Fdr.BenjaminiHochberg(Array.Empty<double>()));
    }
}
