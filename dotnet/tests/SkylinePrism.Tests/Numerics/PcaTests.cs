using System;
using System.Linq;
using SkylinePrism.Core.Numerics;
using Xunit;

namespace SkylinePrism.Tests.Numerics;

/// <summary>
/// 2-D PCA for QC scatter. Separates structured groups, and - critically - stays cheap when there
/// are far more features than samples (proteomics), which the naive full SVD did not.
/// </summary>
public class PcaTests
{
    [Fact]
    public void Fit2D_SeparatesTwoGroupsOnPc1()
    {
        const int nSamples = 4, nFeatures = 20;
        var m = new double[nSamples, nFeatures];
        for (var i = 0; i < nSamples; i++)
        {
            var group = i < 2 ? 1.0 : -1.0;
            for (var j = 0; j < nFeatures; j++)
                m[i, j] = group + ((i * nFeatures + j) % 3 - 1) * 0.01; // tiny deterministic noise
        }

        var scores = Pca.Fit2D(m);

        // Group members share a PC1 sign; the two groups sit on opposite sides.
        Assert.True(Math.Sign(scores[0, 0]) == Math.Sign(scores[1, 0]));
        Assert.True(Math.Sign(scores[2, 0]) == Math.Sign(scores[3, 0]));
        Assert.NotEqual(Math.Sign(scores[0, 0]), Math.Sign(scores[2, 0]));
    }

    [Fact]
    public void Fit2D_WideMatrix_IsCheap_NoFullFeatureSvd()
    {
        // Regression guard: 20k features would make a full feature-space SVD allocate a
        // 20000 x 20000 (~3 GB) V matrix. The Gram-based PCA must handle this in a blink.
        const int nSamples = 6, nFeatures = 20_000;
        var m = new double[nSamples, nFeatures];
        for (var i = 0; i < nSamples; i++)
            for (var j = 0; j < nFeatures; j++)
                m[i, j] = (i % 2 == 0 ? 1.0 : -1.0) + ((i * nFeatures + j) % 7) * 0.001;

        var scores = Pca.Fit2D(m);

        var anyNonZero = false;
        for (var i = 0; i < nSamples; i++)
            if (scores[i, 0] != 0.0)
                anyNonZero = true;
        Assert.True(anyNonZero, "expected non-degenerate PCA scores");
    }

    /// <summary>
    /// The transpose-free entry point must be bit-identical to transposing first. It exists purely to
    /// avoid a second full copy of the biggest matrix in the pipeline, so "same answer" is the whole
    /// contract - and NaN handling, dropped constant features and the sign of each component all have
    /// to come out the same, not merely close.
    /// </summary>
    [Fact]
    public void Fit2DOfFeaturesBySamples_MatchesTransposingFirst()
    {
        const int nSamples = 6, nFeatures = 25;
        var rand = new Random(20260820);
        var featuresBySamples = new double[nFeatures, nSamples];
        var samplesByFeatures = new double[nSamples, nFeatures];
        for (var f = 0; f < nFeatures; f++)
            for (var s = 0; s < nSamples; s++)
            {
                // Include a constant feature, an all-NaN feature and scattered NaNs - the branches
                // where a transposed read could diverge.
                var v = f == 3 ? 1.0
                    : f == 7 ? double.NaN
                    : (s % 5 == 4 && f % 6 == 0) ? double.NaN
                    : (s < nSamples / 2 ? 0.0 : 4.0) + rand.NextDouble();
                featuresBySamples[f, s] = v;
                samplesByFeatures[s, f] = v;
            }

        var viaTranspose = Pca.Fit2D(samplesByFeatures);
        var direct = Pca.Fit2DOfFeaturesBySamples(featuresBySamples);

        Assert.Equal(viaTranspose.GetLength(0), direct.GetLength(0));
        Assert.Equal(viaTranspose.GetLength(1), direct.GetLength(1));
        for (var i = 0; i < viaTranspose.GetLength(0); i++)
            for (var j = 0; j < viaTranspose.GetLength(1); j++)
                Assert.Equal(viaTranspose[i, j], direct[i, j]);
    }
}
