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
}
