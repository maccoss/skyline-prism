namespace SkylinePrism.Core.Rollup;

/// <summary>
/// A rollup strategy aggregates a LOG2 (features x samples) matrix into per-sample LOG2
/// abundances (length = number of columns). Used for both transition-&gt;peptide (Layer 3)
/// and peptide-&gt;protein (Layer 7) aggregation; the concrete strategies differ per stage
/// (e.g. the peptide median-polish adds a +log2(n) offset that the protein stage omits).
/// </summary>
public interface IRollupMethod
{
    double[] Aggregate(double[,] log2Matrix);
}
