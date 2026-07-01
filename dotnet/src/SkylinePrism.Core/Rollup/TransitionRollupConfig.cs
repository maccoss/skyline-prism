namespace SkylinePrism.Core.Rollup;

/// <summary>Transition-&gt;peptide rollup method.</summary>
public enum TransitionRollupMethod
{
    Sum,
    MedianPolish,
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

    /// <summary>When false (default), MS1 "precursor" fragment ions are excluded.</summary>
    public bool UseMs1 { get; init; }

    public bool ExcludePrecursor => !UseMs1;

    public bool LogTransform { get; init; } = true;
}
