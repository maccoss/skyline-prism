using System.Collections.Generic;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// All transition-level rows for a single peptide, as read from the merged parquet.
/// Parallel lists (one entry per row) mirror the long-format DataFrame the Python
/// _process_single_peptide receives.
/// </summary>
public sealed class PeptideBlock
{
    public required string Peptide { get; init; }
    public List<string> Ion { get; } = new();
    public List<string> PrecursorCharge { get; } = new();
    public List<string> ProductCharge { get; } = new();
    public List<string> Sample { get; } = new();
    public List<double> Area { get; } = new();
    public List<double> RetentionTime { get; } = new();

    /// <summary>Product (fragment) m/z per row; populated only for library-assisted rollup.</summary>
    public List<double> ProductMz { get; } = new();

    public int RowCount => Ion.Count;
}
