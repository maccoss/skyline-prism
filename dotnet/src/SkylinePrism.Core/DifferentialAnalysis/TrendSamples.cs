using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// Which samples a trend design can actually use, and the subject each belongs to.
/// </summary>
/// <param name="Columns">Matrix column per usable sample.</param>
/// <param name="X">Trend value per usable sample, parallel to <paramref name="Columns"/>.</param>
/// <param name="SubjectOf">
/// Subject index per usable sample, or an empty list when the design is independent.
/// </param>
/// <param name="SubjectCount">Distinct subjects retained.</param>
/// <param name="Messages">What was excluded and why, for the status line.</param>
public sealed record TrendSelection(
    IReadOnlyList<int> Columns,
    IReadOnlyList<double> X,
    IReadOnlyList<int> SubjectOf,
    int SubjectCount,
    IReadOnlyList<string> Messages);

/// <summary>
/// Resolves the samples a linear-trend design runs over: those with a usable trend value, and - for
/// a within-subject trend - those belonging to a subject that can carry a slope at all.
/// </summary>
public static class TrendSamples
{
    /// <summary>
    /// Select the usable samples. <paramref name="xValues"/> and <paramref name="subjectLabels"/>
    /// are indexed by MATRIX COLUMN, not by position in <paramref name="columns"/>, because that is
    /// how a metadata column arrives.
    /// </summary>
    /// <param name="withinSubject">
    /// Whether the samples are repeated measures on shared subjects. When true,
    /// <paramref name="subjectLabels"/> is required and subjects that cannot carry a within-subject
    /// slope are excluded.
    /// </param>
    public static TrendSelection Resolve(
        IReadOnlyList<int> columns, double[] xValues, string?[]? subjectLabels, bool withinSubject)
    {
        var messages = new List<string>();

        // A sample with no trend value is not evidence about the slope. Dropped first, so the
        // subject rules below see only samples that could contribute.
        var usable = columns.Where(c => c < xValues.Length && double.IsFinite(xValues[c])).ToList();
        var noValue = columns.Count - usable.Count;
        if (noValue > 0)
            messages.Add($"{noValue} sample(s) have no value in the trend column and were left out.");

        if (!withinSubject)
        {
            var xs = usable.Select(c => xValues[c]).ToList();
            return new TrendSelection(usable, xs, Array.Empty<int>(), 0, messages);
        }

        if (subjectLabels is null)
            // Worded for the person reading it, not the caller: both front ends show this message
            // as it stands, and it used to name a property of DifferentialOptions.
            throw new ArgumentException(
                "A within-subject trend needs a subject column: the metadata column that "
                + "identifies each subject, so each one's level can be taken out of the slope.");

        // Group by subject, then keep only subjects that can carry a slope. A subject needs at
        // least two DISTINCT trend values: with one sample, or with several all at the same x, its
        // dummy absorbs every point it has and it contributes nothing to the slope while still
        // costing a parameter. Excluded out loud rather than silently carried.
        var bySubject = usable
            .Where(c => c < subjectLabels.Length && !string.IsNullOrEmpty(subjectLabels[c]))
            .GroupBy(c => subjectLabels[c]!, StringComparer.Ordinal)
            .ToList();

        var unlabeled = usable.Count - bySubject.Sum(g => g.Count());
        if (unlabeled > 0)
            messages.Add($"{unlabeled} sample(s) have no subject and were left out.");

        var keptColumns = new List<int>();
        var keptX = new List<double>();
        var subjectOf = new List<int>();
        var dropped = new List<string>();
        var subjectIndex = 0;
        foreach (var group in bySubject.OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            var cols = group.ToList();
            var distinctX = cols.Select(c => xValues[c]).Distinct().Count();
            if (cols.Count < 2 || distinctX < 2)
            {
                dropped.Add(group.Key);
                continue;
            }

            foreach (var c in cols)
            {
                keptColumns.Add(c);
                keptX.Add(xValues[c]);
                subjectOf.Add(subjectIndex);
            }

            subjectIndex++;
        }

        if (dropped.Count > 0)
            messages.Add(
                $"{dropped.Count} subject(s) have fewer than two distinct trend values, so they "
                + $"carry no within-subject slope, and were left out ({NamePreview.Of(dropped, 3)}).");

        return new TrendSelection(keptColumns, keptX, subjectOf, subjectIndex, messages);
    }
}
