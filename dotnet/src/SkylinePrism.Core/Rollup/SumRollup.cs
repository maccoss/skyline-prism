using System;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Sum rollup, ported from chunked_processing.py:_process_single_peptide ("sum" branch,
/// lines 227-236): abundance_j = log2( clip( sum_i 2^m[i,j], min=1 ) ).
/// Input and output are LOG2.
/// </summary>
public sealed class SumRollup : IRollupMethod
{
    public double[] Aggregate(double[,] log2Matrix)
    {
        var nRows = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);
        var result = new double[nCols];

        for (var j = 0; j < nCols; j++)
        {
            double linearSum = 0.0;
            for (var i = 0; i < nRows; i++)
            {
                var v = log2Matrix[i, j];
                if (double.IsNaN(v))
                    continue;
                linearSum += Math.Pow(2.0, v);
            }
            // clip(lower=1) then log2.
            result[j] = Math.Log2(Math.Max(linearSum, 1.0));
        }
        return result;
    }
}
