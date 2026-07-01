using System;

namespace SkylinePrism.Core.Rollup;

public enum ProteinRollupMethod
{
    MedianPolish,
    Sum,
}

/// <summary>
/// Peptide-&gt;protein matrix rollup, porting rollup.py:rollup_protein_matrix (the single
/// source of truth). Dispatch by peptide count:
///   0 -&gt; NaN, 1 -&gt; peptide directly, &lt; min_peptides -&gt; sum_linear, else the method.
/// The median-polish branch uses col_effects WITHOUT the +log2(n) offset (unlike the
/// peptide stage). Input/output LOG2.
/// </summary>
public static class ProteinMatrixRollup
{
    public static double[] Aggregate(double[,] log2Matrix, ProteinRollupMethod method, int minPeptides)
    {
        var nPep = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);

        if (nPep == 0)
        {
            var nan = new double[nCols];
            for (var j = 0; j < nCols; j++)
                nan[j] = double.NaN;
            return nan;
        }

        if (nPep == 1)
        {
            var row = new double[nCols];
            for (var j = 0; j < nCols; j++)
                row[j] = log2Matrix[0, j];
            return row;
        }

        // sum_linear = log2(clip(sum(2^m), 1)); SumRollup computes exactly this.
        if (nPep < minPeptides)
            return new SumRollup().Aggregate(log2Matrix);

        return method switch
        {
            ProteinRollupMethod.MedianPolish =>
                new MedianPolishRollup(addLog2NOffset: false).Aggregate(log2Matrix),
            ProteinRollupMethod.Sum => new SumRollup().Aggregate(log2Matrix),
            _ => throw new NotSupportedException($"Unsupported protein method {method}"),
        };
    }
}
