using System.Collections.Generic;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>How the samples are related to each other - the design the contrast is estimated under.</summary>
public enum DifferentialDesign
{
    /// <summary>Two independent groups.</summary>
    Unpaired,

    /// <summary>
    /// Two measurements of the same subject. Fitted as a FIXED-effect subject block
    /// (<c>[1, treat, subject one-hot]</c>), which is what the lab's toolkit does - not a random
    /// effect. A random-intercept model is a separate thing and is not implemented yet.
    /// </summary>
    Paired,

    /// <summary>
    /// A slope against a numeric time or dose, testing whether it differs from zero. Assumes every
    /// sample is INDEPENDENT - one sample per subject, as in a dose-response where each animal
    /// receives one dose.
    /// </summary>
    LinearTrend,

    /// <summary>
    /// The same slope where the SAME subjects are followed across the trend column, fitted as
    /// <c>[1, x, subject one-hot]</c>.
    /// </summary>
    /// <remarks>
    /// This is <see cref="Paired"/> generalized from a two-level column to a numeric one, and it
    /// exists because <see cref="LinearTrend"/> is wrong for a repeated-measures study: treating one
    /// subject's several samples as independent understates the standard error, so the test reports
    /// more hits than the data support. The subject block absorbs each subject's overall level, so
    /// the slope is estimated purely WITHIN subject.
    ///
    /// <para>A fixed effect, not a random intercept - the same choice <see cref="Paired"/> makes,
    /// for the same reason. On a balanced design the two give very similar slopes; they diverge on
    /// unbalanced designs, on missing timepoints, and when a random SLOPE per subject is wanted. A
    /// mixed model is a different estimator and is still not implemented.</para>
    /// </remarks>
    LinearTrendWithinSubject,
}

/// <summary>The estimator applied to the design.</summary>
public enum DifferentialTest
{
    /// <summary>limma empirical-Bayes moderated t. The only test that uses a variance prior.</summary>
    ModeratedT,

    /// <summary>Two-sample t with unequal variances (Welch-Satterthwaite df).</summary>
    WelchT,

    /// <summary>Two-sample t with a pooled variance.</summary>
    StudentT,

    /// <summary>One-sample t on the within-subject differences.</summary>
    PairedT,

    /// <summary>Wilcoxon signed-rank on the within-subject differences.</summary>
    Wilcoxon,

    /// <summary>Mann-Whitney U, two-sided.</summary>
    MannWhitney,
}

/// <summary>
/// Where the moderated-t prior variance comes from. Only meaningful for
/// <see cref="DifferentialTest.ModeratedT"/>.
/// </summary>
public enum VariancePrior
{
    /// <summary>One global <c>(s0^2, d0)</c> for every feature - Smyth (2004).</summary>
    Global,

    /// <summary>
    /// The lab's default: a LOWESS of log(within-group variance) on log(within-group MEAN INTENSITY),
    /// fitted on RAW pre-log intensities, one point per (feature, group), converted back to log space
    /// by the delta method. That gives the prior's SHAPE; its level and the prior degrees of freedom
    /// are then fitted to the design's residuals around it.
    /// </summary>
    /// <remarks>
    /// This is <c>proteomics-toolkit</c>'s <c>moderation="intensity_trend"</c>, and it is NOT the same
    /// estimator as <see cref="LimmaTrend"/> despite both being called an intensity trend. See
    /// <c>docs/differential-analysis.md</c>: they differ in the smoother, the space and what
    /// contributes a point, and agree to a median 0.9% on p-values (at most 18%) on the trend golden.
    /// </remarks>
    IntensityTrend,

    /// <summary>
    /// limma's own <c>trend=TRUE</c>: a natural cubic spline of log(residual variance) on mean LOG2
    /// expression, re-estimating the prior degrees of freedom as well as the scale. Kept because it is
    /// what limma does and what the committed inmoose goldens pin, but it is not the lab's default.
    /// </summary>
    LimmaTrend,

    /// <summary>
    /// DEqMS (Zhu 2020): a LOWESS of log(residual variance) on log(peptide count). Needs a peptide
    /// count per feature, so it is protein-level only.
    /// </summary>
    PeptideCount,
}

/// <summary>How the raw p-values are corrected for multiple testing.</summary>
public enum MultipleTesting
{
    /// <summary>Benjamini-Hochberg step-up FDR.</summary>
    BenjaminiHochberg,

    /// <summary>Benjamini-Yekutieli, valid under arbitrary dependence.</summary>
    BenjaminiYekutieli,

    /// <summary>Bonferroni family-wise correction.</summary>
    Bonferroni,

    /// <summary>Holm step-down family-wise correction.</summary>
    Holm,

    /// <summary>No correction - the adjusted column repeats the raw p.</summary>
    None,
}

/// <summary>
/// Everything that selects WHICH analysis <see cref="Differential.Run"/> performs, as opposed to the
/// data it performs it on.
/// </summary>
/// <remarks>
/// <para>Three orthogonal axes, which is how the lab's toolkit models it too: a
/// <see cref="Design"/>, a <see cref="Test"/>, and - for the moderated test alone - a
/// <see cref="Prior"/>. Kept as one record rather than a widening parameter list because the
/// combinations are what a caller reasons about, and because a record can gain an axis without
/// breaking every call site.</para>
/// <para>The defaults are the lab's defaults: an unpaired moderated-t with the intensity-trend prior.
/// Note that <see cref="Prior"/> defaults to <see cref="VariancePrior.IntensityTrend"/> rather than
/// <see cref="VariancePrior.Global"/>, which is a deliberate change from the old <c>trend: false</c>
/// default - see the release notes.</para>
/// </remarks>
public sealed record DifferentialOptions
{
    /// <summary>The lab's defaults.</summary>
    public static readonly DifferentialOptions Default = new();

    /// <summary>The unpaired global-prior moderated t, which is what <c>trend: false</c> used to mean.</summary>
    public static readonly DifferentialOptions GlobalPrior = new() { Prior = VariancePrior.Global };

    public DifferentialDesign Design { get; init; } = DifferentialDesign.Unpaired;

    public DifferentialTest Test { get; init; } = DifferentialTest.ModeratedT;

    public VariancePrior Prior { get; init; } = VariancePrior.IntensityTrend;

    public MultipleTesting Correction { get; init; } = MultipleTesting.BenjaminiHochberg;

    /// <summary>
    /// Pairing key per sample column, for <see cref="DifferentialDesign.Paired"/> and for the paired
    /// tests. Indexed by the ORIGINAL matrix column, so it is read through the selected columns.
    /// </summary>
    public IReadOnlyList<string?>? SubjectLabels { get; init; }

    /// <summary>
    /// The name of the trend column, for the status line and the axis. Display only - the values
    /// come from <see cref="TimeValues"/>.
    /// </summary>
    public string? TrendColumn { get; init; }

    /// <summary>Numeric time or dose per sample column, for <see cref="DifferentialDesign.LinearTrend"/>.</summary>
    public IReadOnlyList<double>? TimeValues { get; init; }

    /// <summary>
    /// Peptides behind each FEATURE (not sample), for the count-based priors. Parallel to the feature
    /// ids. Null where the matrix does not carry <c>n_peptides</c>, which is the peptide-level case.
    /// </summary>
    public IReadOnlyList<double>? PeptideCounts { get; init; }

    /// <summary>
    /// Sample-column GROUPS to fit the variance prior on, instead of the contrast arms - the
    /// toolkit's <c>variance_prior_group_column</c>. Null uses the contrast arms.
    /// </summary>
    /// <remarks>
    /// <para>Pointing this at dedicated QC or reference replicates takes the intensity trend's SHAPE
    /// from samples with no biology in them, which trace noise against intensity cleanly. It does not
    /// set the trend's level: that is fitted to the design residuals whichever groups give the shape,
    /// because control pools lack the biology the residuals carry and would otherwise make every test
    /// optimistic (see <c>Differential.WithCalibratedLevel</c>).</para>
    /// <para>A LIST OF GROUPS, not one pooled set, and the distinction matters: the variance is
    /// computed WITHIN each group, so pooling QC and reference replicates into one would count the
    /// systematic difference between two different materials as measurement noise. Grouping by the
    /// nominated column's distinct values is what the toolkit does, and it also yields more points
    /// for the trend to be fitted from.</para>
    /// <para>These are ABSOLUTE sample-column indices into the full matrix, because the replicates
    /// nominated here usually take no part in the contrast and so appear nowhere else.</para>
    /// </remarks>
    public IReadOnlyList<IReadOnlyList<int>>? PriorGroupColumns { get; init; }

    /// <summary>
    /// The default <see cref="MinPerGroup"/>. Named rather than repeated as a literal because three
    /// places need to agree on it: this initializer, the CLI's <c>--min-per-group</c> fallback, and
    /// <see cref="QuantCommand"/>, which omits the flag from a recorded command when the run used the
    /// default. A drift between them would make a recorded command reproduce a different minimum.
    /// </summary>
    public const int DefaultMinPerGroup = 2;

    /// <summary>Minimum samples per arm before the contrast is refused.</summary>
    public int MinPerGroup { get; init; } = DefaultMinPerGroup;

    /// <summary>Covariates to adjust the contrast for. Ignored by the rank-based tests.</summary>
    public IReadOnlyList<Covariate>? Covariates { get; init; }

    /// <summary>A short human-readable name for what this asks for, for the status line and provenance.</summary>
    /// <param name="actualPrior">
    /// The prior that was actually fitted (<see cref="DifferentialResult.VariancePrior"/>), when the
    /// result is in hand. A requested prior can fall back - a trend design has no groups for the
    /// intensity trend, a peptide matrix has no peptide counts - and naming the requested one then
    /// would put a prior in the headline that did not produce the numbers underneath it.
    /// </param>
    public string Describe(string? actualPrior = null)
    {
        var priorName = actualPrior switch
        {
            null => Prior switch
            {
                VariancePrior.Global => "global prior",
                VariancePrior.IntensityTrend => "intensity-trend prior",
                VariancePrior.LimmaTrend => "limma-trend prior",
                VariancePrior.PeptideCount => "peptide-count prior",
                _ => "intensity-trend prior",
            },
            "global" => "global prior",
            "intensity-trend" => "intensity-trend prior",
            "intensity-trend from controls" => "intensity-trend prior from controls",
            "intensity-trend from design groups" => "intensity-trend prior from design groups",
            "limma-trend" => "limma-trend prior",
            "peptide-count" => "peptide-count prior",
            var other => other + " prior",
        };
        var test = Test switch
        {
            DifferentialTest.ModeratedT => $"moderated t ({priorName})",
            DifferentialTest.WelchT => "Welch t",
            DifferentialTest.StudentT => "Student t",
            DifferentialTest.PairedT => "paired t",
            DifferentialTest.Wilcoxon => "Wilcoxon signed-rank",
            _ => "Mann-Whitney U",
        };
        var over = TrendColumn is null ? string.Empty : $" over {TrendColumn}";
        var design = Design switch
        {
            DifferentialDesign.Paired => "paired",
            DifferentialDesign.LinearTrend => $"linear trend{over}",
            DifferentialDesign.LinearTrendWithinSubject => $"linear trend{over}, within subject",
            _ => "unpaired",
        };
        return $"{test}, {design}";
    }
}
