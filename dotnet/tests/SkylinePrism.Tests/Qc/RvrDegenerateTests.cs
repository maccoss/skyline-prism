using System;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The relative variance reduction (RVR = QC improvement / reference improvement) is only meaningful
/// when the reference actually improved.
///
/// <para>When it did not - improvement zero or NEGATIVE - the ratio used to be forced to +infinity, which
/// tripped the "&gt; 2" branch and announced <i>"QC improved much more than reference - possible
/// overfitting to the reference"</i>. That is backwards: a reference that got WORSE is the opposite of
/// overfitting to it. It also failed the whole verdict on a degenerate number. RVR is now NaN
/// (undefined) in that case, the ratio warnings are skipped, and the situation is reported on its own
/// terms.</para>
/// </summary>
public class RvrDegenerateTests
{
    /// <summary>
    /// Builds a [feature, sample] log2 matrix whose per-type CV is controlled by a spread factor: index
    /// 0..1 are reference samples, 2..3 QC. A larger spread means a larger CV within that type.
    /// </summary>
    private static double[,] Matrix(double refSpread, double qcSpread)
    {
        const int nF = 400;
        var m = new double[nF, 4];
        var rng = new Random(11);
        for (var f = 0; f < nF; f++)
        {
            var baseLog2 = 14.0 + rng.NextDouble();
            m[f, 0] = baseLog2 - refSpread;
            m[f, 1] = baseLog2 + refSpread;
            m[f, 2] = baseLog2 - qcSpread;
            m[f, 3] = baseLog2 + qcSpread;
        }
        return m;
    }

    private static readonly int[] RefIdx = { 0, 1 };
    private static readonly int[] QcIdx = { 2, 3 };

    private static ValidationStatus Evaluate(
        double refBefore, double refAfter, double qcBefore, double qcAfter) =>
        ValidationStatus.Compute(
            Matrix(refBefore, qcBefore), Matrix(refAfter, qcAfter), RefIdx, QcIdx)!;

    [Fact]
    public void WhenTheReferenceGetsWorse_RvrIsUndefinedNotInfinite()
    {
        // Reference degrades (spread grows), QC improves slightly - the reported scenario.
        var status = Evaluate(refBefore: 0.30, refAfter: 0.34, qcBefore: 0.50, qcAfter: 0.48);

        Assert.True(status.ReferenceCvImprovement < 0, "fixture should degrade the reference");
        Assert.True(double.IsNaN(status.RelativeVarianceReduction));
        Assert.False(double.IsInfinity(status.RelativeVarianceReduction));
    }

    [Fact]
    public void WhenTheReferenceGetsWorse_TheOverfittingWarningIsNotClaimed()
    {
        var status = Evaluate(0.30, 0.34, 0.50, 0.48);

        // The old behavior CLAIMED overfitting to the reference while the reference had degraded. The
        // new message mentions the overfitting check by name, so match the claim, not the word.
        Assert.DoesNotContain(status.Warnings,
            w => w.Contains("possible overfitting", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(status.Warnings, w => w.Contains("RVR=", StringComparison.Ordinal));
        Assert.Contains(status.Warnings, w => w.Contains("could not be evaluated", StringComparison.OrdinalIgnoreCase));
        // The genuine problem is still reported.
        Assert.Contains(status.Warnings, w => w.Contains("Reference CV increased", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void AnUnevaluableRvrDoesNotByItselfFailTheVerdict()
    {
        // Previously rvr=+inf made "rvr < 2.0" false, so the run failed on a degenerate number rather
        // than on anything measured.
        var status = Evaluate(0.30, 0.34, 0.50, 0.48);

        Assert.True(status.QcCvImprovement > 0);
        Assert.True(status.Passed || status.PcaDistanceRatio <= 0.5,
            "an undefined RVR must not be the thing that fails the verdict");
    }

    [Fact]
    public void GenuineOverfittingIsStillFlagged()
    {
        // Reference barely improves, QC improves a lot -> a real, finite, large RVR.
        var status = Evaluate(refBefore: 0.40, refAfter: 0.399, qcBefore: 0.60, qcAfter: 0.30);

        Assert.False(double.IsNaN(status.RelativeVarianceReduction));
        Assert.True(status.RelativeVarianceReduction > 2.0);
        Assert.Contains(status.Warnings, w => w.Contains("possible overfitting", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void PoorGeneralizationIsStillFlagged()
    {
        // Reference improves a lot, QC barely -> small finite RVR.
        var status = Evaluate(refBefore: 0.60, refAfter: 0.30, qcBefore: 0.40, qcAfter: 0.399);

        Assert.False(double.IsNaN(status.RelativeVarianceReduction));
        Assert.True(status.RelativeVarianceReduction < 0.5);
        Assert.Contains(status.Warnings, w => w.Contains("may not generalize", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ABalancedImprovementPassesWithNoRatioWarnings()
    {
        var status = Evaluate(refBefore: 0.50, refAfter: 0.30, qcBefore: 0.50, qcAfter: 0.30);

        Assert.False(double.IsNaN(status.RelativeVarianceReduction));
        Assert.DoesNotContain(status.Warnings, w => w.Contains("RVR", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(status.Warnings, w => w.Contains("could not be evaluated", StringComparison.OrdinalIgnoreCase));
    }
}
