using System;
using System.Globalization;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// Which features count as hits: a p-value cut, whether that cut is applied to the raw or the
/// adjusted p, and a minimum effect size.
/// </summary>
/// <remarks>
/// <para>One rule object rather than a pair of loose doubles, because the same decision is read in
/// four places - the status line's count, the volcano's coloring AND its guide lines, the hit list
/// handed to enrichment, and <see cref="SignificanceScan"/> - and a volcano whose colored hits
/// disagree with its own threshold lines is worse than either convention on its own.</para>
///
/// <para><b>Raw versus adjusted is a real choice, not a display one.</b> Adjusted is the honest
/// default and stays the default. Raw is offered because a small pilot cohort can have nothing
/// surviving correction at all, and seeing the uncorrected ranking is a legitimate way to decide
/// whether a full study is worth running - as long as it is labeled. <see cref="Describe"/> and
/// <see cref="YAxisLabel"/> exist so it always is.</para>
/// </remarks>
public sealed record SignificanceRule
{
    /// <summary>The conventional rule: adjusted p below 0.05 and at least a two-fold change.</summary>
    public static SignificanceRule Default { get; } = new();

    /// <summary>The p-value cut. Applied to the adjusted p unless <see cref="UseAdjusted"/> is false.</summary>
    public double PThreshold { get; init; } = 0.05;

    /// <summary>
    /// Whether <see cref="PThreshold"/> is applied to the adjusted p (the default) or the raw one.
    /// </summary>
    public bool UseAdjusted { get; init; } = true;

    /// <summary>
    /// Minimum absolute log2 effect. Zero disables the effect filter, leaving the p-value alone to
    /// decide - which is what a reader wants when the effect is a slope they have no prior for.
    /// </summary>
    /// <remarks>
    /// For a two-arm contrast this is the log2 fold change between the arms. For a TREND design it
    /// is the modeled log2 change across the observed range of the trend column, so the same number
    /// means the same thing - "at least a two-fold change over what was measured" - whatever the
    /// units of that column are. See <see cref="DifferentialRow.LogFc"/>.
    /// </remarks>
    public double Log2FcThreshold { get; init; } = 1.0;

    /// <summary>The p-value this rule judges <paramref name="row"/> by.</summary>
    public double PValueOf(DifferentialRow row) => UseAdjusted ? row.AdjPValue : row.PValue;

    /// <summary>Whether this feature is a hit. A NaN in either quantity is never one.</summary>
    public bool IsSignificant(DifferentialRow row)
    {
        var p = PValueOf(row);
        return p < PThreshold && Math.Abs(row.LogFc) >= Log2FcThreshold;
    }

    /// <summary>
    /// The y-axis name, which must say which p is plotted.
    /// </summary>
    /// <param name="corrected">
    /// Whether a multiple-testing correction ran at all. With <c>Correct = None</c> the adjusted
    /// column simply holds the raw p, so calling it adjusted would be the plainest kind of
    /// mislabeling.
    /// </param>
    public string YAxisLabel(bool corrected) =>
        UseAdjusted && corrected ? "-log10(adjusted p-value)" : "-log10(p-value)";

    /// <summary>
    /// The rule as a reader would write it - "adj.P &lt; 0.05, |log2FC| >= 1". Goes in the status
    /// line and in the "nothing to enrich" message, so a hit count is never reported without the
    /// rule that produced it.
    /// </summary>
    /// <param name="effectName">
    /// What the effect is called on this design: <c>log2FC</c> for a two-arm contrast, or something
    /// like <c>log2 change across week</c> for a trend.
    /// </param>
    public string Describe(string effectName = "log2FC")
    {
        var p = $"{(UseAdjusted ? "adj.P" : "P")} < {Fmt(PThreshold)}";
        return Log2FcThreshold > 0
            ? $"{p}, |{effectName}| >= {Fmt(Log2FcThreshold)}"
            : p;
    }

    private static string Fmt(double v) => v.ToString("0.####", CultureInfo.InvariantCulture);
}
