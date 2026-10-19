namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// The canonical <c>prism differential</c> flag value for each statistical choice - the words the CLI
/// accepts, and the words <c>quant_parameters.yaml</c> records, so a report's parameters read as the
/// command that would reproduce it.
/// </summary>
/// <remarks>
/// One table for both directions of use: the CLI prints these in its refusals ("That design runs:
/// moderated, paired-t, wilcoxon") and the report writes them, and two copies had already begun to
/// differ - the report was writing enum names such as <c>ModeratedT</c> that no flag accepts.
/// </remarks>
public static class DifferentialTokens
{
    /// <summary><c>--design</c>.</summary>
    public static string Design(DifferentialDesign design) => design switch
    {
        DifferentialDesign.Paired => "paired",
        DifferentialDesign.LinearTrend => "trend",
        DifferentialDesign.LinearTrendWithinSubject => "trend-within-subject",
        DifferentialDesign.BlockedBySubject => "blocked",
        _ => "unpaired",
    };

    /// <summary><c>--test</c>.</summary>
    public static string Test(DifferentialTest test) => test switch
    {
        DifferentialTest.WelchT => "welch",
        DifferentialTest.StudentT => "student",
        DifferentialTest.MannWhitney => "mann-whitney",
        DifferentialTest.PairedT => "paired-t",
        DifferentialTest.Wilcoxon => "wilcoxon",
        _ => "moderated",
    };

    /// <summary><c>--prior</c>: the prior that was REQUESTED (what actually ran is <see cref="DifferentialResult.VariancePrior"/>).</summary>
    public static string Prior(VariancePrior prior) => prior switch
    {
        VariancePrior.Global => "global",
        VariancePrior.LimmaTrend => "limma-trend",
        VariancePrior.PeptideCount => "peptide-count",
        VariancePrior.LimmaTrendIntensityNPeptides => "limma-trend-intensity-n-peptides",
        _ => "intensity-trend",
    };

    /// <summary>
    /// The prior a <c>--prior</c> value names, the inverse of <see cref="Prior(VariancePrior)"/>, or null
    /// for an unknown one. Case is ignored and underscores are accepted for the hyphens, as
    /// <c>proteomics-toolkit</c>'s <c>moderation</c> accepts them, so one spelling works in both tools.
    /// </summary>
    public static VariancePrior? ParsePrior(string token) =>
        token.Trim().ToLowerInvariant().Replace('_', '-') switch
        {
            "global" => VariancePrior.Global,
            "limma-trend" => VariancePrior.LimmaTrend,
            "limma-trend-intensity-n-peptides" => VariancePrior.LimmaTrendIntensityNPeptides,
            "peptide-count" => VariancePrior.PeptideCount,
            "intensity-trend" => VariancePrior.IntensityTrend,
            _ => null,
        };

    /// <summary>Whether a prior conditions on peptide count, and so is protein level only.</summary>
    public static bool UsesPeptideCounts(VariancePrior prior) =>
        prior is VariancePrior.PeptideCount or VariancePrior.LimmaTrendIntensityNPeptides;

    /// <summary><c>--correction</c>.</summary>
    public static string Correction(MultipleTesting correction) => correction switch
    {
        MultipleTesting.BenjaminiYekutieli => "by",
        MultipleTesting.Holm => "holm",
        MultipleTesting.Bonferroni => "bonferroni",
        MultipleTesting.None => "none",
        _ => "bh",
    };

    /// <summary><c>--level</c>.</summary>
    public static string Level(FeatureLevel level) =>
        level == FeatureLevel.Peptide ? "peptide" : "protein";
}
