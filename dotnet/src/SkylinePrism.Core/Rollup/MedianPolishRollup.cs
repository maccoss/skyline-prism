using System;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Median-polish rollup. Ported from chunked_processing.py:_process_single_peptide
/// ("median_polish" branch, lines 208-225) for the peptide stage, and shared with the
/// protein stage (rollup.py:rollup_protein_matrix) which uses the SAME polish but WITHOUT
/// the offset.
///
/// PARITY-CRITICAL: the transition-&gt;peptide stage adds +log2(n_transitions) so the
/// median-polish abundance is comparable in magnitude to a sum; the peptide-&gt;protein
/// stage does NOT add this offset. Controlled by <see cref="_addLog2NOffset"/>.
///
/// If the number of feature rows is below <see cref="_minTransitions"/>, all samples are
/// returned as NaN (matching the Python guard).
/// </summary>
public sealed class MedianPolishRollup : IRollupMethod
{
    private readonly bool _addLog2NOffset;
    private readonly int _minTransitions;

    public MedianPolishRollup(bool addLog2NOffset, int minTransitions = 1)
    {
        _addLog2NOffset = addLog2NOffset;
        _minTransitions = minTransitions;
    }

    public double[] Aggregate(double[,] log2Matrix) => Aggregate(log2Matrix, out _);

    /// <summary>
    /// As <see cref="Aggregate(double[,])"/>, additionally handing back the polish residuals
    /// (<c>[nRows, nCols]</c>) that the decomposition already computes.
    /// <para>
    /// Following Plubell et al. 2022 these are evidence, not waste - a row with a consistently
    /// large residual indicates interference, a PTM, or protein processing - so callers that want
    /// to persist them can, rather than the value being computed and dropped. <c>null</c> when the
    /// row count is below the minimum, since no polish ran.
    /// </para>
    /// </summary>
    public double[] Aggregate(double[,] log2Matrix, out double[,]? residuals)
    {
        var nRows = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);

        residuals = null;
        if (nRows < _minTransitions)
        {
            var nan = new double[nCols];
            for (var j = 0; j < nCols; j++)
                nan[j] = double.NaN;
            return nan;
        }

        var polish = TukeyMedianPolish.Run(log2Matrix);
        residuals = polish.Residuals;
        var abundances = new double[nCols];

        // scale_factor = log2(n_used) for the peptide stage; 0 for the protein stage.
        var scaleFactor = _addLog2NOffset && nRows > 0 ? Math.Log2(nRows) : 0.0;
        for (var j = 0; j < nCols; j++)
            abundances[j] = polish.ColEffects[j] + scaleFactor;

        return abundances;
    }
}
