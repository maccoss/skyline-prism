using System;
using System.Collections.Generic;

namespace SkylinePrism.Core.Qc;

/// <summary>Outcome of evaluating a ComBat correction against the control samples.</summary>
public readonly record struct BatchRevertDecision(
    bool Revert,
    string ControlName,
    double ControlCvBefore,
    double ControlCvAfter,
    string? ControlAsymmetryNote)
{
    public bool Evaluable => ControlName.Length > 0;
}

/// <summary>
/// Decides whether a ComBat correction should be reverted because it degraded control-sample quality.
/// Ports the decision in Python's legacy normalize_pipeline (evaluate_batch_correction): the primary
/// control is QC (else reference); revert if its median CV worsened by more than <c>worsenTolerance</c>.
/// A reference CV improving far more than the QC CV is reported alongside as an observation - it never
/// reverts anything. CVs are on the LINEAR scale (via <see cref="CvMetrics"/>).
/// </summary>
public static class BatchCorrectionEvaluator
{
    public static BatchRevertDecision Evaluate(
        double[,] preCombat,
        double[,] postCombat,
        IReadOnlyList<int> qcIdx,
        IReadOnlyList<int> refIdx,
        double worsenTolerance = 1.1,
        double maxAsymmetryRatio = 2.0)
    {
        var hasQc = qcIdx.Count >= 2;
        var hasRef = refIdx.Count >= 2;
        return Decide(
            qcBefore: hasQc ? CvMetrics.MedianCv(preCombat, qcIdx) : double.NaN,
            qcAfter: hasQc ? CvMetrics.MedianCv(postCombat, qcIdx) : double.NaN,
            refBefore: hasRef ? CvMetrics.MedianCv(preCombat, refIdx) : double.NaN,
            refAfter: hasRef ? CvMetrics.MedianCv(postCombat, refIdx) : double.NaN,
            hasQc, hasRef, worsenTolerance, maxAsymmetryRatio);
    }

    /// <summary>
    /// The decision itself, from the four median CVs. Split out so a streaming caller - which
    /// accumulates those CVs a row at a time and never holds either matrix - reaches the decision
    /// through exactly this code rather than a second copy of the thresholds.
    /// </summary>
    public static BatchRevertDecision Decide(
        double qcBefore, double qcAfter,
        double refBefore, double refAfter,
        bool hasQc, bool hasRef,
        double worsenTolerance = 1.1,
        double maxAsymmetryRatio = 2.0)
    {
        // Primary control for the revert decision: QC (independent) preferred, else reference.
        double before, after;
        string name;
        if (hasQc)
        {
            (before, after, name) = (qcBefore, qcAfter, "QC");
        }
        else if (hasRef)
        {
            (before, after, name) = (refBefore, refAfter, "reference");
        }
        else
        {
            return new BatchRevertDecision(false, "", double.NaN, double.NaN, null);
        }

        var revert = !double.IsNaN(before) && !double.IsNaN(after) && before > 0
                     && after > before * worsenTolerance;

        // Observation only - it changes nothing. One direction is worth stating: the anchoring control
        // improved far more than the INDEPENDENT one, which is the direction that would flatter a
        // reference-anchored correction. It is still only an observation, not a diagnosis: reference
        // and QC are different materials injected at different amounts, so whichever started with more
        // excess variance has more of it to remove, and the ratio alone cannot establish a cause.
        string? asymmetry = null;
        if (hasQc && hasRef)
        {
            var refImp = refBefore > 0 ? (refBefore - refAfter) / refBefore : 0;
            var qcImp = qcBefore > 0 ? (qcBefore - qcAfter) / qcBefore : 0;
            if (qcImp > 0 && refImp / qcImp > maxAsymmetryRatio)
                asymmetry = $"reference CV improved {refImp:P0} but QC only {qcImp:P0} "
                            + $"(ratio {refImp / qcImp:F1}) - the reference improved far more than the "
                            + "independent control";
        }

        return new BatchRevertDecision(revert, name, before, after, asymmetry);
    }
}
