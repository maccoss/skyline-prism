namespace SkylinePrism.Core.Rollup;

/// <summary>Transition-&gt;peptide rollup method.</summary>
public enum TransitionRollupMethod
{
    Sum,
    MedianPolish,
    LibraryAssist,
    TopN,
}

/// <summary>
/// Configuration for the transition-&gt;peptide rollup (Stage 2), mirroring the relevant
/// fields of the Python ChunkedRollupConfig.
/// </summary>
public sealed class TransitionRollupConfig
{
    public TransitionRollupMethod Method { get; init; } = TransitionRollupMethod.Sum;

    /// <summary>Peptides with fewer than this many transitions are dropped from output.</summary>
    public int MinTransitions { get; init; } = 3;

    /// <summary>Transitions to keep for the topn method.</summary>
    public int TopNCount { get; init; } = 3;

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

    // --- Library-assisted rollup (method == LibraryAssist) ---

    /// <summary>Path to the spectral library (.blib) used for library-assisted rollup.</summary>
    public string? LibraryPath { get; init; }

    /// <summary>Minimum matched library fragments required for a library fit (default 3).</summary>
    public int LibraryMinFragments { get; init; } = 3;

    /// <summary>m/z tolerance (Da) for matching transitions to library fragments.</summary>
    public double LibraryMzTolerance { get; init; } = 0.02;

    /// <summary>Normalized-residual threshold above which a transition is treated as interfered.</summary>
    public double LibraryOutlierThreshold { get; init; } = 1.0;
}
