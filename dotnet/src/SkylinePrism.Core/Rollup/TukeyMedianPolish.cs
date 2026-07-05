using System;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Tukey's median polish, ported line-for-line from rollup.py:tukey_median_polish so it
/// reproduces R's stats::medpolish (and the Python pipeline) to within tolerance.
///
/// SCALE: input and output are LOG2. The returned ColEffects are on the original scale
/// (overall + centered sample effect).
///
/// Parity-critical detail: both row and column effects are re-centered to median 0 each
/// iteration (Tukey 1977 normalization). A sum of median-0 vectors does not itself have
/// median 0, so without this re-centering the overall + col_effects partition drifts by
/// an unspecified per-matrix constant. See rollup.py:690-712 for the full rationale.
/// </summary>
public static class TukeyMedianPolish
{
    public static MedianPolishResult Run(double[,] matrix, int maxIter = 20, double tol = 1e-4)
    {
        var nRows = matrix.GetLength(0);
        var nCols = matrix.GetLength(1);

        // residuals = matrix.copy()
        var residuals = (double[,])matrix.Clone();
        var overall = 0.0;
        var rowEffects = new double[nRows];
        var colEffects = new double[nCols];

        // Scratch buffers reused across iterations.
        var rowBuf = new double[nCols];
        var colBuf = new double[nRows];

        var converged = false;
        var iteration = 0;

        for (; iteration < maxIter; iteration++)
        {
            // old_residuals = residuals.copy()
            var oldResiduals = (double[,])residuals.Clone();

            // Step 1: row sweep -- subtract per-row median (nanmedian axis=1).
            for (var i = 0; i < nRows; i++)
            {
                for (var j = 0; j < nCols; j++)
                    rowBuf[j] = residuals[i, j];
                var rowMedian = Stats.NanMedian(rowBuf);
                for (var j = 0; j < nCols; j++)
                    residuals[i, j] -= rowMedian;
                rowEffects[i] += rowMedian;
            }

            // Re-center col_effects so median(col_effects) == 0.
            var cdelta = Stats.NanMedian(colEffects);
            for (var j = 0; j < nCols; j++)
                colEffects[j] -= cdelta;
            overall += cdelta;

            // Step 2: column sweep -- subtract per-column median (nanmedian axis=0).
            for (var j = 0; j < nCols; j++)
            {
                for (var i = 0; i < nRows; i++)
                    colBuf[i] = residuals[i, j];
                var colMedian = Stats.NanMedian(colBuf);
                for (var i = 0; i < nRows; i++)
                    residuals[i, j] -= colMedian;
                colEffects[j] += colMedian;
            }

            // Re-center row_effects so median(row_effects) == 0.
            var rdelta = Stats.NanMedian(rowEffects);
            for (var i = 0; i < nRows; i++)
                rowEffects[i] -= rdelta;
            overall += rdelta;

            // Convergence: max |residual - old_residual| over non-NaN cells (nanmax).
            var maxChange = double.NaN;
            for (var i = 0; i < nRows; i++)
            {
                for (var j = 0; j < nCols; j++)
                {
                    var diff = Math.Abs(residuals[i, j] - oldResiduals[i, j]);
                    if (double.IsNaN(diff))
                        continue;
                    if (double.IsNaN(maxChange) || diff > maxChange)
                        maxChange = diff;
                }
            }

            // NaN < tol is false, matching numpy where an all-NaN nanmax stays unconverged.
            if (maxChange < tol)
            {
                converged = true;
                iteration++; // Python reports n_iterations = iteration + 1.
                break;
            }
        }

        // Report col_effects on the original scale (overall + sample effects).
        var colEffectsOriginal = new double[nCols];
        for (var j = 0; j < nCols; j++)
            colEffectsOriginal[j] = colEffects[j] + overall;

        return new MedianPolishResult
        {
            Overall = overall,
            RowEffects = rowEffects,
            ColEffects = colEffectsOriginal,
            Residuals = residuals,
            NIterations = converged ? iteration : maxIter,
            Converged = converged,
        };
    }
}
