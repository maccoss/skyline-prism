using System;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Parity tests for <see cref="SignificanceScan"/> against the Python explorer's
/// <c>significance_scan</c> / <c>_enumerate_splits</c> / <c>scan_summary</c>.
/// </summary>
public class SignificanceScanTests
{
    private const int Fn = 8;
    private const int Sn = 12;

    // s0..s3 = g1, s4..s7 = g2, s8..s11 = g3; f0,f1 are +4 in g1 so g1-vs-rest has hits.
    private static readonly string?[] Labels =
    {
        "g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2", "g3", "g3", "g3", "g3",
    };

    private static double[,] BuildMatrix()
    {
        var m = new double[Fn, Sn];
        for (var f = 0; f < Fn; f++)
        for (var s = 0; s < Sn; s++)
        {
            var v = 10.0 + ((f * 2 + s) % 5) / 2.0;
            if (Labels[s] == "g1" && (f == 0 || f == 1))
                v += 4.0;
            m[f, s] = v;
        }

        return m;
    }

    private static void AssertRel(double expected, double actual, double rtol)
    {
        var err = Math.Abs(actual - expected) / Math.Abs(expected);
        Assert.True(err <= rtol, $"expected {expected:R}, actual {actual:R}, rel err {err:R}");
    }

    private static readonly string[] FeatureIds = Enumerable.Range(0, Fn).Select(f => $"f{f}").ToArray();

    [Fact]
    public void Run_OneVsRest_MatchesExplorer()
    {
        var res = SignificanceScan.Run(BuildMatrix(), FeatureIds, Labels, null, ScanScope.OneVsRest);

        Assert.False(res.Truncated);
        Assert.Equal(3, res.Rows.Count);

        Assert.Equal("g1", res.Rows[0].A);
        Assert.Equal("g2 + g3", res.Rows[0].B);
        Assert.Equal(4, res.Rows[0].NA);
        Assert.Equal(8, res.Rows[0].NB);
        Assert.Equal(2, res.Rows[0].NSig);
        AssertRel(1.038708214235576e-17, res.Rows[0].MinQ, 1e-9);

        Assert.Equal("g3", res.Rows[1].A);
        Assert.Equal(0, res.Rows[1].NSig);
        AssertRel(0.2050333588347331, res.Rows[1].MinQ, 1e-9);

        Assert.Equal("g2", res.Rows[2].A);
        Assert.Equal(0, res.Rows[2].NSig);

        var summary = SignificanceScan.Summarize(res);
        Assert.Equal(new ScanSummary(3, 2, 1, "g1", "g2 + g3"), summary);
    }

    [Fact]
    public void Run_Pairwise_MatchesExplorer()
    {
        var res = SignificanceScan.Run(BuildMatrix(), FeatureIds, Labels, null, ScanScope.Pairwise);

        Assert.Equal(3, res.Rows.Count);
        Assert.Equal("g1", res.Rows[0].A);
        Assert.Equal("g3", res.Rows[0].B);
        Assert.Equal(2, res.Rows[0].NSig);
        AssertRel(5.542228508823184e-14, res.Rows[0].MinQ, 1e-9);

        Assert.Equal("g1", res.Rows[1].A);
        Assert.Equal("g2", res.Rows[1].B);
        Assert.Equal(2, res.Rows[1].NSig);

        var summary = SignificanceScan.Summarize(res);
        Assert.Equal(new ScanSummary(3, 2, 2, "g1", "g3"), summary);
    }

    [Fact]
    public void EnumerateSplits_SubsetPairs_MatchesItertoolsOrder()
    {
        var splits = SignificanceScan
            .EnumerateSplits(new[] { "a", "b", "c" }, ScanScope.SubsetPairs, 400)
            .Select(s => (string.Join(",", s.A), string.Join(",", s.B)))
            .ToArray();

        var expected = new[]
        {
            ("a,b", "c"), ("a,c", "b"), ("a", "b,c"), ("a", "b"), ("a", "c"), ("b", "c"),
        };
        Assert.Equal(expected, splits);
    }

    [Fact]
    public void PermuteCalibrate_IsReproducibleAndBelowObserved()
    {
        var m = BuildMatrix();
        var a = new[] { 0, 1, 2, 3 };            // g1
        var b = new[] { 4, 5, 6, 7, 8, 9, 10, 11 }; // g2 + g3
        const int observedNSig = 2;              // g1-vs-rest recovers f0, f1

        var null1 = SignificanceScan.PermuteCalibrate(m, FeatureIds, a, b, nPerm: 40);
        var null2 = SignificanceScan.PermuteCalibrate(m, FeatureIds, a, b, nPerm: 40);

        Assert.Equal(null1, null2); // seeded -> reproducible
        Assert.NotEmpty(null1);
        // Shuffling the labels breaks the planted signal, so the null sits below the observed hit.
        var median = null1.OrderBy(x => x).ElementAt(null1.Length / 2);
        Assert.True(median < observedNSig, $"null median {median} not below observed {observedNSig}");
    }

    [Fact]
    public void ScanPermutationNull_IsReproducibleWithExpectedLength()
    {
        var m = BuildMatrix();
        var null1 = SignificanceScan.ScanPermutationNull(m, FeatureIds, Labels, null, ScanScope.OneVsRest, nPerm: 8);
        var null2 = SignificanceScan.ScanPermutationNull(m, FeatureIds, Labels, null, ScanScope.OneVsRest, nPerm: 8);

        Assert.Equal(8, null1.Length);
        Assert.Equal(null1, null2);
    }

    [Fact]
    public void EnumerateSplits_OneVsRest_And_Pairwise()
    {
        var ovr = SignificanceScan.EnumerateSplits(new[] { "a", "b", "c" }, ScanScope.OneVsRest, 400)
            .Select(s => (string.Join(",", s.A), string.Join(",", s.B))).ToArray();
        Assert.Equal(new[] { ("a", "b,c"), ("b", "a,c"), ("c", "a,b") }, ovr);

        var pw = SignificanceScan.EnumerateSplits(new[] { "a", "b", "c" }, ScanScope.Pairwise, 400)
            .Select(s => (string.Join(",", s.A), string.Join(",", s.B))).ToArray();
        Assert.Equal(new[] { ("a", "b"), ("a", "c"), ("b", "c") }, pw);
    }
}
