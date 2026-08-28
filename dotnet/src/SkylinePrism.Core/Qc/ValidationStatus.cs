using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Dual-control validation verdict (validation.py:validate_correction / ValidationMetrics). Uses the
/// reference and QC samples to judge whether correction damaged the data: the QC CV must be
/// measurable and must not get worse, and QC/reference must not collapse together in PCA space.
/// Every failing condition also produces a <see cref="Warnings"/> entry, so a FAILED verdict always
/// says what failed.
///
/// <para>The relative variance reduction (RVR = QC improvement / reference improvement) is reported as
/// a <see cref="Notes"/> observation, NOT as a warning and NOT as part of the verdict. Reference and QC
/// are different materials injected at different amounts, so one of them having more headroom to
/// improve than the other is ordinary - it does not imply overfitting, and failing a run on it is
/// excessive. A real problem shows up as a QC CV that got worse or as controls collapsing together,
/// both of which are checked on their own terms.</para>
/// </summary>
public sealed record ValidationStatus(
    double ReferenceCvBefore, double ReferenceCvAfter, double ReferenceCvImprovement,
    double QcCvBefore, double QcCvAfter, double QcCvImprovement,
    double RelativeVarianceReduction,
    double PcaDistanceBefore, double PcaDistanceAfter, double PcaDistanceRatio,
    bool Passed, IReadOnlyList<string> Warnings, IReadOnlyList<string> Notes)
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
        // or NEGATIVE - the ratio is undefined, not infinite: +inf used to read as "QC improved much
        // more than reference", when in fact the reference had got WORSE. NaN says "not measured", and
        // the reference degradation is reported as its own warning below.
        var rvr = refImp > 0 ? qcImp / refImp : double.NaN;

        var distBefore = PcaCentroidDistance(before, refIdx, qcIdx);
        var distAfter = PcaCentroidDistance(after, refIdx, qcIdx);
        var pcaRatio = distBefore > 0 ? distAfter / distBefore : double.NaN;

        // Every reason the verdict can fail must appear here, or a FAILED banner names nothing the
        // reader can act on. That includes a QC CV that could not be measured at all - qcImp is
        // forced to 0 in that case, which would otherwise fail the verdict silently.
        var qcMeasurable = qcBefore > 0 && !double.IsNaN(qcBefore) && !double.IsNaN(qcAfter);
        var warnings = new List<string>();
        if (!qcMeasurable)
            warnings.Add("QC CV could not be measured (no usable QC values), so the correction could not be validated.");
        else if (qcImp < 0)
            warnings.Add("QC CV increased after normalization.");
        if (refImp < 0)
            warnings.Add("Reference CV increased after normalization.");
        if (pcaRatio < 0.5)
            warnings.Add($"QC-reference PCA distance decreased by {(1 - pcaRatio) * 100:0.0}% - control samples may be collapsing together.");

        // The QC-vs-reference ratio is an observation, not a defect. The two control groups are
        // different materials at different injection amounts; whichever started with more excess
        // variance has more of it to remove, so the two improvements routinely differ by a lot with
        // nothing wrong. It is reported so the reader can see it, and it decides nothing.
        var notes = new List<string>();
        if (double.IsNaN(rvr))
        {
            notes.Add(
                "The QC-vs-reference improvement ratio (RVR) is undefined here: the reference CV did not "
                + "improve, so there is nothing to take a ratio against.");
        }
        else if (qcImp > 0 && (rvr > 2.0 || rvr < 0.5))
        {
            // Only when BOTH controls improved. A negative ratio means QC got worse while the
            // reference improved - "improved considerably more than QC" would be a strange way to say
            // that, and the QC-CV warning above already says it plainly.
            var qcImprovedMore = rvr > 2.0;
            var more = qcImprovedMore ? "QC" : "The reference";
            var less = qcImprovedMore ? "the reference" : "QC";
            notes.Add(
                $"{more} improved considerably more than {less} (RVR={rvr:0.00}). Reference and QC are "
                + "different materials injected at different amounts, so their CVs need not improve "
                + "together - on its own this is not a problem.");
        }

        // Only genuine damage fails the verdict: the independent control got worse, or the two control
        // groups collapsed onto each other. An unchanged QC CV (qcImp == 0) is not damage. A NaN PCA
        // ratio (degenerate geometry) is unmeasured, not failed - but an unmeasurable QC CV is,
        // because then nothing was validated; both cases are stated in the warnings above.
        var passed = qcMeasurable && qcImp >= 0 && !(pcaRatio < 0.5);

        return new ValidationStatus(
            refBefore, refAfter, refImp, qcBefore, qcAfter, qcImp, rvr,
            distBefore, distAfter, pcaRatio, passed, warnings, notes);
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
