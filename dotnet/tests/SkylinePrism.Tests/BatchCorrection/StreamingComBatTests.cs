using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.BatchCorrection;
using SkylinePrism.Core.Numerics;
using Xunit;

namespace SkylinePrism.Tests.BatchCorrection;

/// <summary>
/// The empirical-Bayes step is the one part of ComBat that looks unstreamable, because <c>ItSol</c>
/// re-reads the standardized matrix on every iteration. <see cref="StreamingComBat"/> replaces that
/// with two numbers per (batch, feature). These tests pin that the substitution is exact enough to
/// be invisible, including on the input designed to break the obvious alternative formulation.
/// </summary>
public class StreamingComBatTests
{
    /// <summary>
    /// <paramref name="maxRel"/> is predicted from the input's conditioning, not fitted to what
    /// passes: reconstructing the sum of squares costs about the digits lost in
    /// <c>value - batchMean</c>, i.e. roughly <c>center/spread</c> squared. Standardized data
    /// (center 0, spread 1) loses none of them and lands at machine noise; the 50 / 1e-3 row is a
    /// deliberately ill-conditioned input standardization does not actually produce, included to
    /// show the degradation is bounded and gradual rather than a cliff.
    /// </summary>
    [Theory]
    [InlineData(0.0, 1.0, 1e-14)]     // standardized data as ComBat actually produces it
    [InlineData(-8.0, 4.0, 1e-14)]    // off-center but well-scaled
    [InlineData(50.0, 1e-3, 1e-10)]   // ~5e4 center-to-spread ratio -> ~9.4 digits lost of 16
    public void Estimate_ReproducesInMemoryItSol(double center, double spread, double maxRel)
    {
        const int nf = 60;
        var batches = new List<List<int>>
        {
            new() { 0, 1, 2, 3, 4 },
            new() { 5, 6, 7 },
            new() { 8, 9, 10, 11 },
        };
        var nSamples = batches.Sum(b => b.Count);

        // A stand-in for ComBat's standardized data. The EB step does not care where it came from,
        // so driving it directly isolates the sufficient-statistics substitution from everything else.
        var rng = new Random(4242);
        var sData = new double[nf, nSamples];
        for (var f = 0; f < nf; f++)
        for (var s = 0; s < nSamples; s++)
            sData[f, s] = center + spread * (rng.NextDouble() - 0.5);

        var (gammaHat, deltaHat, sumSq) = Summarize(sData, batches, nf);

        // Reference: the real in-memory iteration, reading the whole matrix every pass.
        var expectedGamma = new double[batches.Count, nf];
        var expectedDelta = new double[batches.Count, nf];
        var gammaHat2d = To2d(gammaHat, nf);
        var deltaHat2d = To2d(deltaHat, nf);
        for (var i = 0; i < batches.Count; i++)
        {
            var (gBar, t2, a, b) = Priors(gammaHat[i], deltaHat[i]);
            ComBat.ItSol(sData, batches[i], i, gammaHat2d, deltaHat2d, gBar, t2, a, b,
                expectedGamma, expectedDelta, nf);
        }

        var (actualGamma, actualDelta, unestimable) = StreamingComBat.Estimate(new ComBatSufficientStats
        {
            Plan = ComBatPlan.Standard(batches, BatchOfSample(batches, nSamples)),
            NFeatures = nf,
            GrandMean = new double[nf],
            StdPooled = Enumerable.Repeat(1.0, nf).ToArray(),
            Counts = batches.Select(b => Enumerable.Repeat(b.Count, nf).ToArray()).ToArray(),
            GammaHat = gammaHat,
            SumSq = sumSq,
        });
        Assert.Equal(0, unestimable); // this fixture has spread in every batch

        double worstGamma = 0, worstDelta = 0;
        for (var i = 0; i < batches.Count; i++)
        for (var f = 0; f < nf; f++)
        {
            worstGamma = Math.Max(worstGamma, RelativeDifference(expectedGamma[i, f], actualGamma[i][f]));
            worstDelta = Math.Max(worstDelta, RelativeDifference(expectedDelta[i, f], actualDelta[i][f]));
        }

        // On real (standardized) input this is machine noise, orders below the 1e-12 the end-to-end
        // harness allows. If it ever loosens, the headroom protecting the corrected outputs has
        // gone - better to learn that here than as a mysterious end-to-end failure.
        Assert.True(worstGamma < maxRel, $"gamma* worst relative difference {worstGamma:E3}");
        Assert.True(worstDelta < maxRel, $"delta* worst relative difference {worstDelta:E3}");
    }

    /// <summary>
    /// Why the shifted sum-of-squares identity is used rather than the algebraically equivalent
    /// <c>S2 - 2*g*S1 + n*g^2</c> the plan originally proposed: on data far from zero the expanded
    /// form subtracts two large nearly-equal quantities and cancels away most of its significant
    /// digits, and the corrected abundances inherit the error.
    /// <para>
    /// Neither form is exact here - the input is 1000 +/- 5e-4, so even forming <c>value - mean</c>
    /// costs ~6 digits - which is the point: the claim being pinned is the gap between the two, not
    /// that the shifted form is free.
    /// </para>
    /// </summary>
    [Fact]
    public void ShiftedSumOfSquares_BeatsTheExpandedForm()
    {
        var values = new[] { 1000.0, 1000.001, 999.9995, 1000.0002, 999.9998 };
        var n = values.Length;
        var mean = NumpyMath.PairwiseSum(values) / n;
        var center = mean + 1e-6; // where the EB iteration evaluates the sum

        var exact = values.Sum(v => ((decimal)v - (decimal)center) * ((decimal)v - (decimal)center));

        var sumSqAboutMean = values.Sum(v => (v - mean) * (v - mean));
        var shifted = sumSqAboutMean + n * (mean - center) * (mean - center);

        var s1 = values.Sum();
        var s2 = values.Sum(v => v * v);
        var expanded = s2 - 2 * center * s1 + n * center * center;

        var shiftedError = Math.Abs(shifted - (double)exact) / (double)exact;
        var expandedError = Math.Abs(expanded - (double)exact) / (double)exact;

        // ~6.3 digits lost forming the deviations (2e6 value-to-deviation ratio) leaves ~1e-10.
        Assert.True(shiftedError < 1e-9, $"shifted form error {shiftedError:E3}");
        Assert.True(expandedError > shiftedError * 100,
            $"expected the expanded form to be far worse; shifted {shiftedError:E3} vs expanded {expandedError:E3}");
    }

    // gammaHat / deltaHat / sumSq exactly as ComBat._fit_batch_effects computes them.
    /// <summary>Batch index per sample, from the per-batch index lists.</summary>
    private static int[] BatchOfSample(IReadOnlyList<List<int>> batches, int nSamples)
    {
        var of = new int[nSamples];
        for (var i = 0; i < batches.Count; i++)
            foreach (var s in batches[i])
                of[s] = i;
        return of;
    }

    private static (double[][] GammaHat, double[][] DeltaHat, double[][] SumSq) Summarize(
        double[,] sData, List<List<int>> batches, int nf)
    {
        var gammaHat = new double[batches.Count][];
        var deltaHat = new double[batches.Count][];
        var sumSq = new double[batches.Count][];
        for (var i = 0; i < batches.Count; i++)
        {
            var idx = batches[i];
            gammaHat[i] = new double[nf];
            deltaHat[i] = new double[nf];
            sumSq[i] = new double[nf];
            var buf = new double[idx.Count];
            for (var f = 0; f < nf; f++)
            {
                for (var k = 0; k < idx.Count; k++)
                    buf[k] = sData[f, idx[k]];
                gammaHat[i][f] = NumpyMath.PairwiseSum(buf) / idx.Count;

                var v = Stats.Var(buf, ddof: 1);
                deltaHat[i][f] = v == 0.0 ? 1.0 : v;

                var sq = new double[idx.Count];
                for (var k = 0; k < idx.Count; k++)
                {
                    var d = buf[k] - gammaHat[i][f];
                    sq[k] = d * d;
                }
                sumSq[i][f] = NumpyMath.PairwiseSum(sq);
            }
        }
        return (gammaHat, deltaHat, sumSq);
    }

    private static (double GBar, double T2, double APrior, double BPrior) Priors(
        double[] gammaHat, double[] deltaHat)
    {
        var gBar = Stats.Mean(gammaHat);
        var t2 = Stats.Var(gammaHat, ddof: 1);
        var m = Stats.Mean(deltaHat);
        var v = Stats.Var(deltaHat, ddof: 1);
        return v > 0 && m > 0
            ? (gBar, t2, m * m / v + 2, m * (m * m / v + 1))
            : (gBar, t2, 1.0, 1.0);
    }

    private static double[,] To2d(double[][] rows, int nf)
    {
        var result = new double[rows.Length, nf];
        for (var i = 0; i < rows.Length; i++)
            for (var f = 0; f < nf; f++)
                result[i, f] = rows[i][f];
        return result;
    }

    private static double RelativeDifference(double expected, double actual)
    {
        if (expected == actual)
            return 0;
        var scale = Math.Max(Math.Abs(expected), double.Epsilon);
        return Math.Abs(expected - actual) / scale;
    }
}
