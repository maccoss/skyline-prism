using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>The samples a <see cref="DifferentialDesign.BlockedBySubject"/> contrast runs over.</summary>
/// <param name="A">Arm A's columns that have a subject, in the order picked.</param>
/// <param name="B">Arm B's columns that have a subject, in the order picked.</param>
/// <param name="SubjectOf">Each kept sample's subject, in the order [A..., B...].</param>
/// <param name="Dropped">Picked samples left out for having no subject.</param>
public sealed record BlockedSelection(
    IReadOnlyList<int> A,
    IReadOnlyList<int> B,
    IReadOnlyList<string> SubjectOf,
    int Dropped);

/// <summary>
/// Which samples a blocked contrast uses and which subject each belongs to - the one place that
/// decides it, so the test, the plots drawn beside it (<see cref="PairedSamples.ColumnsUsed"/>) and the
/// detection view (<see cref="Detection.DetectionAnalysis"/>) cannot disagree about the samples.
/// </summary>
/// <remarks>
/// A sample with no subject is left out rather than made a subject of its own: a blank in the subject
/// column is far more often a missing annotation than a genuinely unrelated sample, and a guess either
/// way changes the correlation. Labels are compared trimmed, so <c>"P1 "</c> and <c>"P1"</c> are one
/// subject.
/// </remarks>
public static class BlockedSamples
{
    /// <summary>The subject of matrix column <paramref name="column"/>, trimmed, or null when it has none.</summary>
    public static string? SubjectOf(IReadOnlyList<string?> subjectLabels, int column) =>
        column >= 0 && column < subjectLabels.Count && !string.IsNullOrWhiteSpace(subjectLabels[column])
            ? subjectLabels[column]!.Trim()
            : null;

    /// <summary>The arms less any sample with no subject, and each kept sample's subject.</summary>
    public static BlockedSelection Resolve(IReadOnlyList<string?> subjectLabels,
        IReadOnlyList<int> groupA, IReadOnlyList<int> groupB)
    {
        var a = groupA.Where(c => SubjectOf(subjectLabels, c) is not null).ToList();
        var b = groupB.Where(c => SubjectOf(subjectLabels, c) is not null).ToList();
        var subjectOf = a.Concat(b).Select(c => SubjectOf(subjectLabels, c)!).ToList();
        return new BlockedSelection(a, b, subjectOf, groupA.Count + groupB.Count - a.Count - b.Count);
    }

    /// <summary>Whether any subject has more than one sample across the two arms.</summary>
    public static bool RepeatsASubject(BlockedSelection selection) =>
        selection.SubjectOf.Distinct(StringComparer.Ordinal).Count() < selection.SubjectOf.Count;
}
