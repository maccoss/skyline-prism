using System;
using System.Linq;
using SkylinePrism.Core.Normalization;
using Xunit;

namespace SkylinePrism.Tests.Normalization;

/// <summary>
/// Marker-based normalization: PC1 of a marker block as a per-sample score, then every feature
/// residualized on it. Ported from the EV-content analysis it was built for, and checked against that
/// notebook's numpy on the real cohort: same variance explained, same correlation with the mean marker
/// profile, the same four opposing loadings (ANXA2, ANXA6, CD81, SDCBP), and per-sample scores agreeing
/// to 5e-07.
/// <para>
/// The parity run was against the notebook AS IT STOOD AT PORT TIME (variance explained 0.645136,
/// correlation with the mean profile 0.927921). The notebook has since been re-run on newer PRISM
/// output and now reports 0.704 and 0.951 - which is what the docs quote. Both are that cohort at that
/// moment; neither is a property of this code, so nothing here asserts either number. What IS pinned
/// is the behavior, on synthetic blocks, below.
/// </para>
/// </summary>
public class MarkerNormalizationTests
{
    /// <summary>A block where every marker moves with one latent factor, plus one that opposes it.</summary>
    private static double[,] Block(int markers, int samples, double[] factor, bool opposeLast = false)
    {
        var m = new double[markers, samples];
        var rng = new Random(7);
        for (var i = 0; i < markers; i++)
        {
            var sign = opposeLast && i == markers - 1 ? -1.0 : 1.0;
            var level = 15 + i * 0.3;
            for (var j = 0; j < samples; j++)
                m[i, j] = level + sign * factor[j] + (rng.NextDouble() - 0.5) * 0.02;
        }
        return m;
    }

    private static double[] Factor(int n)
        => Enumerable.Range(0, n).Select(j => Math.Sin(j * 0.7) * 1.5).ToArray();

    [Fact]
    public void TheScoreTracksTheLatentFactorTheMarkersShare()
    {
        var factor = Factor(20);
        var score = MarkerNormalization.ComputeScore(
            Block(8, 20, factor), Enumerable.Range(0, 8).Select(i => $"M{i}").ToArray());

        Assert.True(score.VarianceExplained > 0.95, $"expected one dominant axis, got {score.VarianceExplained}");
        var r = Corr(score.Score, factor);
        Assert.True(r > 0.99, $"score should track the shared factor, r={r}");
    }

    [Fact]
    public void TheSignIsOriented_SoHigherScoreAlwaysMeansMoreMarkedMaterial()
    {
        // A principal component's sign is arbitrary: without orientation, "higher score" would mean
        // more marked material or less, at random, between runs of the same data.
        var factor = Factor(16);
        var names = Enumerable.Range(0, 6).Select(i => $"M{i}").ToArray();

        var up = MarkerNormalization.ComputeScore(Block(6, 16, factor), names);
        var down = MarkerNormalization.ComputeScore(
            Block(6, 16, factor.Select(v => -v).ToArray()), names);

        Assert.True(Corr(up.Score, factor) > 0);
        Assert.True(Corr(down.Score, factor.Select(v => -v).ToArray()) > 0);
        Assert.True(up.CorrelationWithMean > 0 && down.CorrelationWithMean > 0);
    }

    [Fact]
    public void AnOpposingMarkerGetsANegativeLoading_WhichIsWhyPc1BeatsTheMean()
    {
        // The reason for PC1: on the EV panel, four of eighteen markers load opposite to the rest, so a
        // plain mean partially cancels. PC1 weights them by contribution and keeps the sign structure.
        var factor = Factor(24);
        var names = Enumerable.Range(0, 7).Select(i => $"M{i}").ToArray();
        var pc1 = MarkerNormalization.ComputeScore(Block(7, 24, factor, opposeLast: true), names);

        Assert.Equal(1, pc1.Loadings.Count(l => l < 0));
        Assert.True(Math.Abs(Corr(pc1.Score, factor)) > 0.99);

        // The mean over the same block is blunted by the opposing member.
        var mean = MarkerNormalization.ComputeScore(
            Block(7, 24, factor, opposeLast: true), names, MarkerScoreMethod.Mean);
        Assert.True(Math.Abs(Corr(pc1.Score, factor)) > Math.Abs(Corr(mean.Score, factor)));
    }

    [Fact]
    public void ResidualizingRemovesTheScoreAxis_AndKeepsEachFeaturesOwnLevel()
    {
        var factor = Factor(18);
        var score = MarkerNormalization.ComputeScore(
            Block(6, 18, factor), Enumerable.Range(0, 6).Select(i => $"M{i}").ToArray());

        // Two features: one that tracks the factor, one flat at a distinct level.
        var m = new double[2, 18];
        for (var j = 0; j < 18; j++)
        {
            m[0, j] = 20 + 2.0 * factor[j];
            m[1, j] = 12.0;
        }
        var meanBefore = Enumerable.Range(0, 18).Select(j => m[0, j]).Average();

        MarkerNormalization.Residualize(m, score.Score);

        // The tracking feature is flattened...
        var after = Enumerable.Range(0, 18).Select(j => m[0, j]).ToArray();
        Assert.True(after.Max() - after.Min() < 0.05, $"expected the score axis removed, span {after.Max() - after.Min()}");
        // ...at its own abundance level, not at zero. A bare residual would make every feature 2^0 = 1.
        Assert.Equal(meanBefore, after.Average(), 6);
        // The flat feature is untouched.
        Assert.Equal(12.0, m[1, 5], 9);
    }

    [Fact]
    public void AFeatureWithTooFewObservationsIsLeftAlone()
    {
        // Two points through a two-parameter model has no residual to speak of; zeroing it would
        // fabricate a result.
        var score = new[] { -2.0, -1.0, 0.0, 1.0, 2.0 };
        var m = new double[1, 5];
        for (var j = 0; j < 5; j++)
            m[0, j] = double.NaN;
        m[0, 1] = 10.0;
        m[0, 3] = 14.0;

        MarkerNormalization.Residualize(m, score);

        Assert.Equal(10.0, m[0, 1], 9);
        Assert.Equal(14.0, m[0, 3], 9);
    }

    [Fact]
    public void TooFewMarkersIsRefused_NotGuessed()
    {
        var ex = Assert.Throws<ArgumentException>(() => MarkerNormalization.ComputeScore(
            new double[2, 10], new[] { "A", "B" }));
        Assert.Contains("at least", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static double Corr(double[] a, double[] b)
    {
        var ma = a.Average();
        var mb = b.Average();
        var sab = a.Zip(b, (x, y) => (x - ma) * (y - mb)).Sum();
        var saa = a.Sum(x => (x - ma) * (x - ma));
        var sbb = b.Sum(y => (y - mb) * (y - mb));
        return sab / Math.Sqrt(saa * sbb);
    }
}
