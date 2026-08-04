using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.App;

/// <summary>
/// Decides which samples a QC plot shows, given the values ticked in the Group-by value dropdown.
///
/// <para>Several values can be ticked at once, which is what lets the control-correlation heatmap show
/// Quality Control AND Standard together while leaving the unknowns out - the useful reading there is
/// that each control type correlates with itself but not with the other, and unknowns bury it.</para>
///
/// <para>Kept out of the window because it decides which data reaches a plot: getting it wrong shows a
/// plausible plot of the wrong samples, with nothing to indicate a problem.</para>
/// </summary>
public static class QcGroupFilter
{
    /// <summary>
    /// Values meaning "this sample is a control", in both vocabularies: the Replicates report carries
    /// Skyline's spellings, while PRISM's synthetic Sample Type column uses its own mapped names.
    /// </summary>
    public static readonly string[] ControlValues =
        { "Standard", "Quality Control", "QC", "reference", "qc" };

    /// <summary>Comparer used throughout: annotation spellings vary in case between sources.</summary>
    public static StringComparer Comparer => StringComparer.OrdinalIgnoreCase;

    /// <summary>
    /// Indices of the samples to plot. An EMPTY selection means "no filter" - every sample - which is how
    /// an untouched dropdown behaves; it is not the same as "nothing matched".
    /// </summary>
    /// <param name="annotationOf">Sample index -> its value in the Group-by column.</param>
    public static List<int> Matching(int sampleCount, Func<int, string> annotationOf, IReadOnlySet<string> selected)
    {
        if (selected.Count == 0)
            return Enumerable.Range(0, sampleCount).ToList();
        return Enumerable.Range(0, sampleCount)
            .Where(i => selected.Contains(annotationOf(i)))
            .ToList();
    }

    /// <summary>Description of the current filter for plot titles ("all samples", "QC + Standard").</summary>
    public static string Describe(IReadOnlyCollection<string> selected) =>
        selected.Count == 0
            ? "all samples"
            : string.Join(" + ", selected.OrderBy(x => x, StringComparer.Ordinal));

    /// <summary>
    /// Closed-state text for the dropdown. Everything ticked reads the same as nothing ticked, because
    /// both show every sample - saying "All" is clearer than listing every value back.
    /// </summary>
    public static string Summarize(IReadOnlyList<string> selected, int totalValues) =>
        selected.Count == 0 || (totalValues > 0 && selected.Count == totalValues)
            ? "All"
            : string.Join(", ", selected);

    /// <summary>Whether a Group-by value denotes a control sample.</summary>
    public static bool IsControlValue(string? value) =>
        value is not null && ControlValues.Contains(value, Comparer);

    /// <summary>
    /// The control values present in <paramref name="available"/>. Empty when the column is not a
    /// sample-type column at all (e.g. a Condition annotation), in which case the caller should leave the
    /// selection alone rather than tick nothing and show an empty plot.
    /// </summary>
    public static IReadOnlyList<string> ControlsAmong(IEnumerable<string> available) =>
        available.Where(IsControlValue).ToList();
}
