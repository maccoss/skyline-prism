using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Dual-control validation verdict (validation.py:validate_correction / ValidationMetrics). Uses the
/// reference and QC samples to judge whether correction improved quality WITHOUT overfitting:
/// QC CV should improve, QC should not overfit toward the reference (relative variance reduction),
/// and QC/reference should not collapse together in PCA space.
/// </summary>
public sealed record ValidationStatus(
    double ReferenceCvBefore, double ReferenceCvAfter, double ReferenceCvImprovement,
    double QcCvBefore, double QcCvAfter, double QcCvImprovement,
    double RelativeVarianceReduction,
    double PcaDistanceBefore, double PcaDistanceAfter, double PcaDistanceRatio,
    bool Passed, IReadOnlyList<string> Warnings)
{
    /// <summary>
    /// Compute the verdict at one level from before/after LOG2 matrices [features, samples] and the
    /// reference / QC sample column indices. Returns null when either control group has &lt; 2 samples.
    /// </summary>
    public static ValidationStatus? Compute(
        double[,] before, double[,] after, IReadOnlyList<int> refIdx, IReadOnlyList<int> qcIdx)
    {
        if (refIdx.Count < 2 || qcIdx.Count < 2)
            return null;

        var refBefore = CvMetrics.MedianCv(before, refIdx);
        var refAfter = CvMetrics.MedianCv(after, refIdx);
        var qcBefore = CvMetrics.MedianCv(before, qcIdx);
        var qcAfter = CvMetrics.MedianCv(after, qcIdx);

        var refImp = refBefore > 0 ? (refBefore - refAfter) / refBefore : 0.0;
        var qcImp = qcBefore > 0 ? (qcBefore - qcAfter) / qcBefore : 0.0;

        // RVR only means something when the reference actually improved. If it did not - improvement zero
        // or NEGATIVE - the ratio is undefined, not infinite. Reporting +inf here made the ">2" branch
        // fire and announce "QC improved much more than reference - possible overfitting to the
        // reference", which is backwards: the reference got WORSE, which is the opposite of overfitting
        // to it. It also failed the whole verdict on a degenerate number. NaN keeps the ratio checks from
        // firing at all, and the situation is reported on its own terms below.
        var rvr = refImp > 0 ? qcImp / refImp : double.NaN;

        var distBefore = PcaCentroidDistance(before, refIdx, qcIdx);
        var distAfter = PcaCentroidDistance(after, refIdx, qcIdx);
        var pcaRatio = distBefore > 0 ? distAfter / distBefore : double.NaN;

        var warnings = new List<string>();
        if (qcImp < 0)
            warnings.Add("QC CV increased after normalization.");
        if (refImp < 0)
            warnings.Add("Reference CV increased after normalization.");
        // The ratio comparisons are meaningless unless the reference improved; see the RVR note above.
        if (double.IsNaN(rvr))
        {
            warnings.Add(
                "Reference CV did not improve, so the QC-vs-reference ratio (RVR) could not be evaluated - "
                + "the overfitting check was skipped, not passed.");
        }
        else
        {
            if (rvr > 2.0)
                warnings.Add($"QC improved much more than reference (RVR={rvr:0.00}) - possible overfitting to the reference.");
            if (rvr < 0.5)
                warnings.Add($"QC improved much less than reference (RVR={rvr:0.00}) - normalization may not generalize.");
        }
        if (pcaRatio < 0.5)
            warnings.Add($"QC-reference PCA distance decreased by {(1 - pcaRatio) * 100:0.0}% - control samples may be collapsing together.");

        // An unevaluable RVR must not silently fail the verdict - the reference-CV warning above already
        // reports that situation on its own terms.
        var passed = qcImp > 0 && pcaRatio > 0.5 && (double.IsNaN(rvr) || rvr < 2.0);

        return new ValidationStatus(
            refBefore, refAfter, refImp, qcBefore, qcAfter, qcImp, rvr,
            distBefore, distAfter, pcaRatio, passed, warnings);
    }

    // Euclidean distance between the QC and reference centroids in 2-D PCA space.
    private static double PcaCentroidDistance(double[,] featuresBySamples, IReadOnlyList<int> refIdx, IReadOnlyList<int> qcIdx)
    {
        var refSet = new HashSet<int>(refIdx);
        var qcSet = new HashSet<int>(qcIdx);
        var control = refIdx.Concat(qcIdx).Distinct().OrderBy(x => x).ToList();
        if (control.Count < 2)
            return double.NaN;

        var nF = featuresBySamples.GetLength(0);
        var sub = new double[control.Count, nF];
        for (var a = 0; a < control.Count; a++)
            for (var f = 0; f < nF; f++)
                sub[a, f] = featuresBySamples[f, control[a]];

        var scores = Pca.Fit2D(sub); // [control.Count, 2]

        double rx = 0, ry = 0, qx = 0, qy = 0;
        int nr = 0, nq = 0;
        for (var a = 0; a < control.Count; a++)
        {
            if (refSet.Contains(control[a])) { rx += scores[a, 0]; ry += scores[a, 1]; nr++; }
            else if (qcSet.Contains(control[a])) { qx += scores[a, 0]; qy += scores[a, 1]; nq++; }
        }
        if (nr == 0 || nq == 0)
            return double.NaN;
        rx /= nr; ry /= nr; qx /= nq; qy /= nq;
        var dx = qx - rx;
        var dy = qy - ry;
        return Math.Sqrt(dx * dx + dy * dy);
    }
}
