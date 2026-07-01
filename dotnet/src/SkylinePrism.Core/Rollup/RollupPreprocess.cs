using System;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Per-peptide transition-matrix preprocessing, ported from
/// chunked_processing.py:_process_single_peptide (lines 180-201). Turns a raw LINEAR
/// transition x sample matrix (with NaN for missing) into a LOG2 matrix ready for the
/// rollup strategies.
///
/// Steps (order matters for parity):
///   1. clip(lower=0): negative measurements -> 0 (NaN preserved).
///   2. impute = max(percentile(positive_values, 1) * 0.5, 1.0), over ALL positive cells;
///      fallback 1.0 if no positive values.
///   3. fillna(impute) then replace(0, impute).
///   4. log2.
/// </summary>
public static class RollupPreprocess
{
    public readonly record struct Result(double[,] Log2Matrix, double ImputeValue);

    public static Result ImputeAndLog2(double[,] linearMatrix, bool logTransform = true)
    {
        var nRows = linearMatrix.GetLength(0);
        var nCols = linearMatrix.GetLength(1);
        var m = new double[nRows, nCols];

        // Step 1: clip(lower=0). Math.Max propagates NaN, matching pandas clip.
        for (var i = 0; i < nRows; i++)
        {
            for (var j = 0; j < nCols; j++)
                m[i, j] = Math.Max(linearMatrix[i, j], 0.0);
        }

        // Step 2: imputation value from positive cells.
        var positives = new System.Collections.Generic.List<double>(nRows * nCols);
        for (var i = 0; i < nRows; i++)
        {
            for (var j = 0; j < nCols; j++)
            {
                var v = m[i, j];
                if (!double.IsNaN(v) && v > 0.0)
                    positives.Add(v);
            }
        }

        double impute;
        if (positives.Count > 0)
        {
            var p1 = Stats.PercentileLinear(positives.ToArray(), 1);
            impute = Math.Max(p1 * 0.5, 1.0);
        }
        else
        {
            impute = 1.0;
        }

        // Step 3: fillna(impute) then replace(0, impute).
        for (var i = 0; i < nRows; i++)
        {
            for (var j = 0; j < nCols; j++)
            {
                var v = m[i, j];
                if (double.IsNaN(v) || v == 0.0)
                    m[i, j] = impute;
            }
        }

        // Step 4: log2.
        if (logTransform)
        {
            for (var i = 0; i < nRows; i++)
            {
                for (var j = 0; j < nCols; j++)
                    m[i, j] = Math.Log2(m[i, j]);
            }
        }

        return new Result(m, impute);
    }
}
