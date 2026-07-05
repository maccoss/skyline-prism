using System;
using System.Linq;
using SkylinePrism.Core.BatchCorrection;
using Xunit;

namespace SkylinePrism.Tests.BatchCorrection;

/// <summary>
/// Reference-anchored ComBat: removes a purely-technical batch offset seen in the reference
/// samples (identical material) while preserving experimental biology.
/// </summary>
public class ReferenceAnchoredComBatTests
{
    [Fact]
    public void AlignsReferenceMeans_AndPreservesBiology()
    {
        const int nF = 8;
        // samples: 0,1 = refA; 2 = expA; 3,4 = refB; 5 = expB. Batch B has a +3 technical offset.
        var batch = new[] { "A", "A", "A", "B", "B", "B" };
        var refMask = new[] { true, true, false, true, true, false };
        const double offset = 3.0;

        var data = new double[nF, 6];
        for (var f = 0; f < nF; f++)
        {
            var base_ = 10 + 0.5 * f;
            var bio = f % 2 == 0 ? 1.0 : -1.0; // biological deviation of experimental vs reference
            data[f, 0] = base_;
            data[f, 1] = base_ + 0.1;          // 2 refs per batch -> scale estimable
            data[f, 2] = base_ + bio;
            data[f, 3] = base_ + offset;
            data[f, 4] = base_ + offset + 0.1;
            data[f, 5] = base_ + offset + bio;
        }

        var c = ReferenceAnchoredComBat.Run(data, batch, refMask);

        for (var f = 0; f < nF; f++)
        {
            var refA = (c[f, 0] + c[f, 1]) / 2;
            var refB = (c[f, 3] + c[f, 4]) / 2;
            // The 3.0 technical offset in the references is removed (means align).
            Assert.True(Math.Abs(refA - refB) < 0.5, $"ref means not aligned at f={f}: {refA} vs {refB}");
            // Experimental biology (exp - its batch reference) is preserved on both sides.
            Assert.True(Math.Abs((c[f, 2] - refA) - (c[f, 5] - refB)) < 0.3, $"biology not preserved at f={f}");
        }
    }

    [Fact]
    public void SingleBatch_ReturnsUnchanged()
    {
        var data = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var c = ReferenceAnchoredComBat.Run(data, new[] { "A", "A", "A" }, new[] { true, false, false });
        for (var f = 0; f < 2; f++)
            for (var s = 0; s < 3; s++)
                Assert.Equal(data[f, s], c[f, s], 12);
    }

    [Fact]
    public void NoReferenceSamples_Throws()
    {
        var data = new double[,] { { 1, 2 }, { 3, 4 } };
        Assert.Throws<ArgumentException>(() =>
            ReferenceAnchoredComBat.Run(data, new[] { "A", "B" }, new[] { false, false }));
    }
}
