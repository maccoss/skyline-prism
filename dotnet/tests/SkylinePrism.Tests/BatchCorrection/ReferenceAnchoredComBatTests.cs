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

    // ---------------------------------------------------------------- shared-core behavior
    //
    // Reference-anchored runs on the same ComBatCore as standard ComBat, so the guarantees that were
    // fixed for standard have to hold here too. Before unification this path had its own estimator
    // and none of them: it substituted a placeholder scale of 1.0 and fed it to the priors, and it
    // reported nothing about what it could not estimate.

    /// <summary>
    /// A feature its own references never measured in some batch has no anchor there. That is not a
    /// zero offset - it is an unknown one, so the feature is passed through untouched rather than
    /// silently corrected as though the batch had no effect on it.
    /// </summary>
    [Fact]
    public void FeatureMissingFromABatchsReferences_IsHeldOutUnchanged()
    {
        var (data, batch, refMask) = Cohort();
        // Feature 1 is NaN in every reference of batch B, but present in its experimental samples.
        data[1, 4] = double.NaN;
        data[1, 5] = double.NaN;

        var diagnostics = new ComBatDiagnostics();
        var c = ReferenceAnchoredComBat.Run(data, batch, refMask, diagnostics: diagnostics);

        for (var s = 0; s < batch.Length; s++)
            Assert.Equal(data[1, s], c[1, s], 12);
        Assert.True(diagnostics.HeldOutFeatures >= 1);
    }

    /// <summary>
    /// References that agree exactly give no spread to estimate a scale from. The location
    /// correction they DO support is still applied; only the rescaling is skipped, and the
    /// non-estimate is reported instead of being passed off as a scale of 1.0.
    /// </summary>
    [Fact]
    public void ReferencesWithNoSpread_KeepLocationButAreNotRescaled()
    {
        var (data, batch, refMask) = Cohort();
        // Batch B's references are identical for feature 2 - no spread, but a real offset.
        data[2, 4] = 14.0;
        data[2, 5] = 14.0;

        var diagnostics = new ComBatDiagnostics();
        var c = ReferenceAnchoredComBat.Run(data, batch, refMask, diagnostics: diagnostics);

        Assert.True(diagnostics.UnestimableScales >= 1, "unestimable scale not reported");

        // Still corrected: the two batches' reference levels are brought together.
        var refA = (c[2, 0] + c[2, 1]) / 2;
        var refB = (c[2, 4] + c[2, 5]) / 2;
        Assert.True(Math.Abs(refA - refB) < Math.Abs((data[2, 0] + data[2, 1]) / 2 - 14.0),
            "location correction was dropped along with the scale");
    }

    /// <summary>
    /// <c>no_reference_batch: "skip"</c> means that batch is left exactly as it came in - and, just
    /// as importantly, that it does not drag other batches' features into the hold-out. A batch we
    /// are not fitting cannot veto a feature it never anchored.
    /// </summary>
    [Fact]
    public void SkipPolicy_LeavesTheUnreferencedBatchAloneWithoutHoldingFeaturesOut()
    {
        // Three batches, so there is still a real offset to remove once the middle one is skipped.
        // (With only two batches and one skipped, the single remaining anchor IS the center, and
        // correctly nothing moves at all.)
        const int nF = 6;
        var batch = new[] { "A", "A", "A", "B", "B", "B", "C", "C", "C" };
        var refMask = new[] { true, true, false, false, false, false, true, true, false };
        var data = new double[nF, 9];
        for (var f = 0; f < nF; f++)
        {
            var b = 12 + 0.7 * f;
            for (var s = 0; s < 9; s++)
                data[f, s] = b + 1.0 * (s / 3) + 0.1 * (s % 3); // +0 / +1 / +2 per batch
        }

        var c = ReferenceAnchoredComBat.Run(data, batch, refMask, noReferenceBatch: "skip");

        // The skipped batch is returned exactly as it came in.
        for (var s = 3; s < 6; s++)
            for (var f = 0; f < nF; f++)
                Assert.Equal(data[f, s], c[f, s], 12);

        // The anchored batches are still corrected - a skipped batch must not hold their features out.
        var movedInAnchored = false;
        foreach (var s in new[] { 0, 1, 2, 6, 7, 8 })
            for (var f = 0; f < nF; f++)
                movedInAnchored |= Math.Abs(data[f, s] - c[f, s]) > 1e-9;
        Assert.True(movedInAnchored, "anchored batches were not corrected");
    }

    /// <summary>NaN in must be NaN out, and nothing else may become NaN.</summary>
    [Fact]
    public void MissingValuesStayLocal()
    {
        var (data, batch, refMask) = Cohort();
        data[3, 2] = double.NaN; // one experimental cell

        var c = ReferenceAnchoredComBat.Run(data, batch, refMask);

        for (var f = 0; f < data.GetLength(0); f++)
            for (var s = 0; s < batch.Length; s++)
                Assert.Equal(double.IsNaN(data[f, s]), double.IsNaN(c[f, s]));
    }

    /// <summary>
    /// 6 features x 8 samples: batches A and B of 4, the first two of each a reference, with a
    /// technical offset of +2 on B.
    /// </summary>
    private static (double[,] Data, string[] Batch, bool[] RefMask) Cohort()
    {
        const int nF = 6;
        var batch = new[] { "A", "A", "A", "A", "B", "B", "B", "B" };
        var refMask = new[] { true, true, false, false, true, true, false, false };
        var data = new double[nF, 8];
        for (var f = 0; f < nF; f++)
        {
            var b = 12 + 0.7 * f;
            var bio = f % 2 == 0 ? 0.8 : -0.6;
            data[f, 0] = b;
            data[f, 1] = b + 0.15;
            data[f, 2] = b + bio;
            data[f, 3] = b + bio + 0.2;
            data[f, 4] = b + 2.0;
            data[f, 5] = b + 2.0 + 0.15;
            data[f, 6] = b + 2.0 + bio;
            data[f, 7] = b + 2.0 + bio + 0.2;
        }
        return (data, batch, refMask);
    }
}
