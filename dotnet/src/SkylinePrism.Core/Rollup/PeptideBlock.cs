using System.Collections.Generic;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// One peptide's transition rows, as the rollup consumes them.
/// <para>
/// The columns here are what the rollup actually USES, not what the report stores, and that is
/// deliberate. Every field it can get as a number is a number: the reader hands back a transition id
/// already composed, a precomputed precursor flag, and a sample INDEX rather than a sample name -
/// because the producer is the rollup's bottleneck and its cost is dominated by allocating a string
/// per row. Pushing the composition into SQL took the read of a 15.5M-row partition from 5.0 s and
/// 4.5 GB allocated to 3.3 s and 1.5 GB. See <c>MergedParquetReader.StreamPeptideBlocks</c>.
/// </para>
/// </summary>
public sealed class PeptideBlock
{
    public required string Peptide { get; init; }

    /// <summary>
    /// Transition identity per row - <c>ion_z{precursor}_{product}</c> - composed once per distinct
    /// value rather than per row. Only distinctness matters to the rollup, but the string itself is
    /// written to <c>peptides_rollup_residuals.parquet</c>, so its exact form is an output contract.
    /// </summary>
    public List<string> TransitionId { get; } = new();

    /// <summary>Whether the row's fragment ion is a precursor (<c>transition_rollup.exclude_precursor</c>).</summary>
    public List<bool> IsPrecursor { get; } = new();

    /// <summary>Precursor charge as a number; used by the library-assisted path. 0 when unparseable.</summary>
    public List<int> PrecursorCharge { get; } = new();

    /// <summary>
    /// Index into the run's sample list, resolved by the reader. Rows whose sample is not in that list
    /// never arrive (the reader inner-joins), which is the same outcome the old name lookup produced by
    /// skipping them.
    /// </summary>
    public List<int> SampleIndex { get; } = new();

    public List<double> Area { get; } = new();
    public List<double> RetentionTime { get; } = new();

    /// <summary>Product (fragment) m/z per row; populated only for library-assisted rollup.</summary>
    public List<double> ProductMz { get; } = new();

    /// <summary>Shape correlation per row; populated only for topN correlation selection.</summary>
    public List<double> ShapeCorrelation { get; } = new();

    public int RowCount => TransitionId.Count;
}
