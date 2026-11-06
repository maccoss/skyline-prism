using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Dual-control validation verdict (validation.py:validate_correction / ValidationMetrics). Uses the
/// reference and QC samples to judge whether processing damaged the data. Every failing condition is a
/// <see cref="Failures"/> entry and <see cref="Passed"/> is exactly "no failures", so a FAILED verdict
/// always says what failed. <see cref="Warnings"/> holds only what is worth a look but decided nothing,
/// such as a reference CV that rose: the banner names the failures, the box below the rest.
///
/// <para>The relative variance reduction (RVR = QC improvement / reference improvement) is reported as
/// a <see cref="Notes"/> observation, NOT as a warning and NOT as part of the verdict. Reference and QC
/// are different materials injected at different amounts, so one of them having more headroom to
/// improve than the other is ordinary - it does not imply overfitting, and failing a run on it is
/// excessive. A real problem shows up as a QC CV that got worse, which is checked on its own terms.</para>
///
/// <para>The QC-reference PCA distance ratio fails the verdict below <see cref="PcaCollapseRatio"/>
/// only when its "before" is the NORMALIZED, pre-ComBat matrix, so that it measures what batch
/// correction alone did to the separation of two different materials. Measured from the RAW rollup,
/// as it once was, the distance includes whatever loading difference separates the reference and QC
/// injections, which normalization removes by design: a pure loading offset drives the ratio to zero
/// with nothing wrong, and a 264-sample cohort whose reference and QC CVs both improved (+30%, +6%)
/// read FAILED on a ratio of 0.43. When no normalized matrix is available the raw ratio is still
/// computed, but it is only a <see cref="Notes"/> observation.</para>
/// </summary>
public sealed record ValidationStatus(
    double ReferenceCvBefore, double ReferenceCvAfter, double ReferenceCvImprovement,
    double QcCvBefore, double QcCvAfter, double QcCvImprovement,
    double RelativeVarianceReduction,
    double PcaDistanceBefore, double PcaDistanceAfter, double PcaDistanceRatio, bool PcaFromNormalized,
    IReadOnlyList<string> Failures, IReadOnlyList<string> Warnings, IReadOnlyList<string> Notes)
{
    /// <summary>True when nothing failed. Derived, so it cannot disagree with <see cref="Failures"/>.</summary>
    public bool Passed => Failures.Count == 0;

    /// <summary>True when the QC-reference PCA distance ratio is below <see cref="PcaCollapseRatio"/>.</summary>
    public bool PcaDistanceShrank => PcaDistanceRatio < PcaCollapseRatio;

    /// <summary>
    /// The PCA distance ratio (after/before) below which the control groups are taken to have
    /// collapsed: a failure when measured across batch correction, a note when measured from raw data.
    /// </summary>
    public const double PcaCollapseRatio = 0.5;

    /// <summary>
    /// Compute the verdict at one level. All matrices are LOG2 [features, samples] with the same sample
    /// columns in the same order: <paramref name="before"/> the raw rollup, <paramref name="after"/>
    /// the corrected output, and <paramref name="normalized"/> the normalized, pre-ComBat matrix
    /// (<c>peptides_log2_internal</c>), which when given is the PCA "before". The CVs are always raw
    /// against corrected. Returns null when either control group has &lt; 2 samples.
    /// </summary>
    public static ValidationStatus? Compute(
        double[,] before, double[,] after, IReadOnlyList<int> refIdx, IReadOnlyList<int> qcIdx,
        double[,]? normalized = null)
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

        var pcaFromNormalized = normalized is not null;
        var distBefore = PcaCentroidDistance(normalized ?? before, refIdx, qcIdx);
        var distAfter = PcaCentroidDistance(after, refIdx, qcIdx);
        var pcaRatio = distBefore > 0 ? distAfter / distBefore : double.NaN;

        // Every reason the verdict can fail must appear here, or a FAILED banner names nothing the
        // reader can act on. That includes a QC CV that could not be measured at all - qcImp is
        // forced to 0 in that case, which would otherwise fail the verdict silently.
        var qcMeasurable = qcBefore > 0 && !double.IsNaN(qcBefore) && !double.IsNaN(qcAfter);
        var failures = new List<string>();
        if (!qcMeasurable)
            failures.Add(
                "QC CV could not be measured (no usable QC values), so the correction could not be validated.");
        else if (qcImp < 0)
            failures.Add("QC CV increased after normalization.");

        // Measured across batch correction alone, a collapse is correction removing the real difference
        // between two materials - which is what happens when the controls are confounded with batch.
        if (pcaFromNormalized && pcaRatio < PcaCollapseRatio)
        {
            failures.Add(
                $"Batch correction moved the QC and reference groups {(1 - pcaRatio) * 100:0.0}% closer "
                + $"together in PCA space (distance ratio {pcaRatio:0.00}, after/before batch correction). "
                + "They are different materials, so correction that pulls them together is removing real "
                + "differences - check whether either control is confined to some of the batches.");
        }

        var warnings = new List<string>();
        if (refImp < 0)
            warnings.Add("Reference CV increased after normalization.");

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

        // Without the normalized matrix the ratio is measured from the raw rollup, so it cannot tell a
        // collapse from a loading difference that normalization removed - see the type's remarks.
        // Reported, never failed.
        if (!pcaFromNormalized && pcaRatio < PcaCollapseRatio)
        {
            notes.Add(
                $"The QC-reference distance in PCA space shrank by {(1 - pcaRatio) * 100:0.0}% "
                + $"(ratio {pcaRatio:0.00}). It is measured against the raw data, because this output "
                + "has no normalized pre-batch-correction peptides, so it includes any difference in "
                + "loading between the reference and QC injections, which normalization removes by "
                + "design - on its own this is not a problem. Check the PCA plots if the two materials "
                + "are expected to stay well apart.");
        }

        // Only genuine damage fails the verdict: the independent control got worse, or batch correction
        // collapsed the two controls. An unchanged QC CV (qcImp == 0) is not damage, but an
        // unmeasurable QC CV is, because then nothing was validated.
        return new ValidationStatus(
            refBefore, refAfter, refImp, qcBefore, qcAfter, qcImp, rvr,
            distBefore, distAfter, pcaRatio, pcaFromNormalized, failures, warnings, notes);
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
