namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Result of Tukey's median polish on a (row x col) matrix, mirroring the Python
/// MedianPolishResult (rollup.py). All values are on the same (LOG2) scale as the input.
///
/// Model: y_ij = overall + RowEffects[i] + ColEffects_centered[j] + Residuals[i,j].
/// <see cref="ColEffects"/> are reported on the ORIGINAL scale (overall + centered
/// sample effect), i.e. per-sample abundances, exactly as the Python code stores them.
/// </summary>
public sealed class MedianPolishResult
{
    public required double Overall { get; init; }

    /// <summary>Row (peptide / transition) effects, centered so median == 0.</summary>
    public required double[] RowEffects { get; init; }

    /// <summary>
    /// Column (sample) effects on the ORIGINAL scale: overall + centered column effect.
    /// These are the per-sample abundances used downstream.
    /// </summary>
    public required double[] ColEffects { get; init; }

    /// <summary>Residuals matrix (row-major [nRows, nCols]); preserved, not discarded.</summary>
    public required double[,] Residuals { get; init; }

    public required int NIterations { get; init; }

    public required bool Converged { get; init; }
}
