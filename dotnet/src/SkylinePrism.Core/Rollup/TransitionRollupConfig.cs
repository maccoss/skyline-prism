using System.Collections.Generic;
using SkylinePrism.Core.Library;

namespace SkylinePrism.Core.Rollup;

/// <summary>Transition-&gt;peptide rollup method.</summary>
public enum TransitionRollupMethod
{
    Sum,
    MedianPolish,
    LibraryAssist,
    TopN,
    Consensus,
}

/// <summary>
/// Configuration for the transition-&gt;peptide rollup (Stage 2), mirroring the relevant
/// fields of the Python ChunkedRollupConfig.
/// </summary>
public sealed record TransitionRollupConfig
{
    public TransitionRollupMethod Method { get; init; } = TransitionRollupMethod.Sum;

    /// <summary>Peptides with fewer than this many transitions are dropped from output.</summary>
    public int MinTransitions { get; init; } = 3;

    /// <summary>Transitions to keep for the topn method.</summary>
    public int TopNCount { get; init; } = 3;

    /// <summary>topn transition selection: "intensity" or "correlation" (needs Shape Correlation).</summary>
    public string TopNSelection { get; init; } = "intensity";

    /// <summary>topn transition weighting: "sum" or "sqrt".</summary>
    public string TopNWeighting { get; init; } = "sum";

    /// <summary>Regularization constant for the consensus method.</summary>
    public double ConsensusRegularization { get; init; } = 0.1;

    /// <summary>When false (default), MS1 "precursor" fragment ions are excluded.</summary>
    public bool UseMs1 { get; init; }

    public bool ExcludePrecursor => !UseMs1;

    public bool LogTransform { get; init; } = true;

    /// <summary>When set (median_polish only), write per-transition residuals to this parquet.</summary>
    public string? ResidualsPath { get; init; }

    /// <summary>Worker threads for per-peptide rollup: 0 = all cores, 1 = serial, N = cap at N.</summary>
    public int MaxDegreeOfParallelism { get; init; }

    /// <summary>Peptides buffered per streamed parquet row group (flush granularity).</summary>
    public int FlushRows { get; init; } = 2000;

    /// <summary>
    /// Ceiling on the buffer pool of the DuckDB connection this stage reads through, in MB
    /// (<c>processing.merge_memory_mb</c>); 0 = size it from the machine. The reader's ORDER BY is a
    /// blocking operator over every transition row, so this is the same kind of knob as it is for the
    /// merge - and the same one, so that bounding PRISM's footprint takes one setting rather than two.
    /// </summary>
    public int MemoryBudgetMb { get; init; }

    // --- Library-assisted rollup (method == LibraryAssist) ---

    /// <summary>Path to the spectral library (.blib) used for library-assisted rollup.</summary>
    public string? LibraryPath { get; init; }

    /// <summary>Minimum matched library fragments required for a library fit (default 3).</summary>
    public int LibraryMinFragments { get; init; } = 3;

    /// <summary>m/z tolerance (Da) for matching transitions to library fragments.</summary>
    public double LibraryMzTolerance { get; init; } = 0.02;

    /// <summary>Normalized-residual threshold above which a transition is treated as interfered.</summary>
    public double LibraryOutlierThreshold { get; init; } = 1.0;

    /// <summary>Iteratively remove interference (high-residual) fragments before the final scale.</summary>
    public bool LibraryRemoveOutliers { get; init; } = true;

    /// <summary>How a sample's scale is fitted (<c>transition_rollup.library_fitting_method</c>).</summary>
    public LibraryFitting LibraryFitting { get; init; } = LibraryFitting.NoiseGatedRatio;

    /// <summary>noise_gated_ratio: SDs of measured fragment noise an excess must also exceed; 0 = off.</summary>
    public double LibraryNoiseSigma { get; init; } = 3.0;

    /// <summary>
    /// noise_gated_ratio: the cohort's measured fragment noise. Null, or a curve with no points, turns
    /// the noise test off - the ratio test alone then decides.
    /// </summary>
    public FragmentNoiseCurve? LibraryNoise { get; init; }

    /// <summary>
    /// noise_gated_ratio: the replicate-pool groups to check each peptide's precision in, as indices
    /// into the run's sample list. Null when the run has no pools with 3 injections.
    /// </summary>
    public IReadOnlyList<NoiseGroup>? PrecisionGroups { get; init; }

    /// <summary>Where the per-peptide precision check is written; null = not written.</summary>
    public string? PrecisionPath { get; init; }
}

/// <summary>How library-assisted rollup fits a sample's scale (see <see cref="LibraryRollup"/>).</summary>
public enum LibraryFitting
{
    /// <summary>Ratio of kept fragment signal to library intensity, removal gated on measured noise.</summary>
    NoiseGatedRatio,

    /// <summary>The original median of log ratios with an ungated 2x removal.</summary>
    MedianPolish,
}
