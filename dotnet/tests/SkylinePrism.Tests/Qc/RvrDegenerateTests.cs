using System;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The relative variance reduction (RVR = QC improvement / reference improvement) is an OBSERVATION -
/// a <see cref="ValidationStatus.Notes"/> entry - never a warning and never part of the verdict.
///
/// <para>Reference and QC are different materials injected at different amounts, so whichever started
/// with more excess variance has more of it to remove; the two improvements routinely differ by a lot
/// with nothing wrong. Failing a run on that asymmetry (the old <c>rvr &lt; 2.0</c> pass condition) was
/// excessive, and calling it "possible overfitting" named a cause the number cannot establish.</para>
///
/// <para>It is also only defined when the reference improved at all. When it did not - improvement zero
/// or NEGATIVE - the ratio used to be forced to +infinity, which tripped the "&gt; 2" branch and
/// announced <i>"QC improved much more than reference"</i> while the reference had in fact got WORSE.
/// RVR is NaN (undefined) there, and the reference degradation is reported on its own terms.</para>
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

        // The old behavior CLAIMED overfitting to the reference while the reference had degraded.
        Assert.DoesNotContain(status.Warnings.Concat(status.Notes),
            m => m.Contains("overfitting", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(status.Warnings, w => w.Contains("RVR", StringComparison.Ordinal));
        Assert.Contains(status.Notes, n => n.Contains("undefined", StringComparison.OrdinalIgnoreCase));
        // The genuine problem is still reported, as a warning.
        Assert.Contains(status.Warnings, w => w.Contains("Reference CV increased", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void AnUnevaluableRvrDoesNotByItselfFailTheVerdict()
    {
        // Previously rvr=+inf made "rvr < 2.0" false, so the run failed on a degenerate number rather
        // than on anything measured.
        var status = Evaluate(0.30, 0.34, 0.50, 0.48);

        Assert.True(status.QcCvImprovement > 0);
        Assert.True(status.Passed || status.PcaDistanceRatio < 0.5,
            "an undefined RVR must not be the thing that fails the verdict");
    }

    [Fact]
    public void ALargeRvrIsANoteAndDoesNotFailTheVerdict()
    {
        // Reference barely improves, QC improves a lot. The controls are different materials at
        // different injection amounts, so this asymmetry is ordinary - it is reported, not penalized.
        var status = Evaluate(refBefore: 0.40, refAfter: 0.399, qcBefore: 0.60, qcAfter: 0.30);

        Assert.True(status.RelativeVarianceReduction > 2.0);
        Assert.Contains(status.Notes, n => n.Contains("RVR=", StringComparison.Ordinal));
        Assert.DoesNotContain(status.Warnings, w => w.Contains("RVR", StringComparison.Ordinal));
        Assert.DoesNotContain(status.Warnings.Concat(status.Notes),
            m => m.Contains("overfitting", StringComparison.OrdinalIgnoreCase));
        Assert.True(status.Passed || status.PcaDistanceRatio < 0.5,
            "an asymmetric improvement must not be the thing that fails the verdict");
    }

    [Fact]
    public void ASmallRvrIsANoteAndDoesNotFailTheVerdict()
    {
        // Reference improves a lot, QC barely - the mirror image, and equally not a defect.
        var status = Evaluate(refBefore: 0.60, refAfter: 0.30, qcBefore: 0.40, qcAfter: 0.399);

        Assert.True(status.RelativeVarianceReduction < 0.5);
        Assert.Contains(status.Notes, n => n.Contains("RVR=", StringComparison.Ordinal));
        Assert.DoesNotContain(status.Warnings, w => w.Contains("RVR", StringComparison.Ordinal));
        Assert.True(status.Passed || status.PcaDistanceRatio < 0.5);
    }

    [Fact]
    public void ABalancedImprovementReportsNoRatioMessageAtAll()
    {
        var status = Evaluate(refBefore: 0.50, refAfter: 0.30, qcBefore: 0.50, qcAfter: 0.30);

        Assert.False(double.IsNaN(status.RelativeVarianceReduction));
        Assert.DoesNotContain(status.Warnings.Concat(status.Notes),
            m => m.Contains("RVR", StringComparison.OrdinalIgnoreCase));
    }
}
