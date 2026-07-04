using System;
using System.Collections.Generic;

namespace SkylinePrism.Core.Qc;

/// <summary>Outcome of evaluating a ComBat correction against the control samples.</summary>
public readonly record struct BatchRevertDecision(
    bool Revert,
    string ControlName,
    double ControlCvBefore,
    double ControlCvAfter,
    string? OverfittingWarning)
{
    public bool Evaluable => ControlName.Length > 0;
}

/// <summary>
/// Decides whether a ComBat correction should be reverted because it degraded control-sample quality.
/// Ports the decision in Python's legacy normalize_pipeline (evaluate_batch_correction): the primary
/// control is QC (else reference); revert if its median CV worsened by more than <c>worsenTolerance</c>,
/// and separately flag overfitting when the reference CV improves far more than the QC CV. CVs are on
/// the LINEAR scale (via <see cref="CvMetrics"/>).
/// </summary>
public static class BatchCorrectionEvaluator
{
    public static BatchRevertDecision Evaluate(
        double[,] preCombat,
        double[,] postCombat,
        IReadOnlyList<int> qcIdx,
        IReadOnlyList<int> refIdx,
        double worsenTolerance = 1.1,
        double maxOverfittingRatio = 2.0)
    {
        // Primary control for the revert decision: QC (independent) preferred, else reference.
        IReadOnlyList<int> control;
        string name;
        if (qcIdx.Count >= 2)
        {
            control = qcIdx;
            name = "QC";
        }
        else if (refIdx.Count >= 2)
        {
            control = refIdx;
            name = "reference";
        }
        else
        {
            return new BatchRevertDecision(false, "", double.NaN, double.NaN, null);
        }

        var before = CvMetrics.MedianCv(preCombat, control);
        var after = CvMetrics.MedianCv(postCombat, control);
        var revert = !double.IsNaN(before) && !double.IsNaN(after) && before > 0
                     && after > before * worsenTolerance;

        // Overfitting flag (warning only): reference CV improves much more than QC CV.
        string? overfit = null;
        if (qcIdx.Count >= 2 && refIdx.Count >= 2)
        {
            var refBefore = CvMetrics.MedianCv(preCombat, refIdx);
            var refAfter = CvMetrics.MedianCv(postCombat, refIdx);
            var qcBefore = CvMetrics.MedianCv(preCombat, qcIdx);
            var qcAfter = CvMetrics.MedianCv(postCombat, qcIdx);
            var refImp = refBefore > 0 ? (refBefore - refAfter) / refBefore : 0;
            var qcImp = qcBefore > 0 ? (qcBefore - qcAfter) / qcBefore : 0;
            if (qcImp > 0 && refImp / qcImp > maxOverfittingRatio)
                overfit = $"reference CV improved {refImp:P0} but QC only {qcImp:P0} "
                          + $"(ratio {refImp / qcImp:F1}) - possible overfitting";
        }

        return new BatchRevertDecision(revert, name, before, after, overfit);
    }
}
