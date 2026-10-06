using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// One subject measured once in each arm: the unit a paired analysis is actually over.
/// </summary>
/// <param name="Subject">The pairing key's value.</param>
/// <param name="AColumn">That subject's sample column in arm A.</param>
/// <param name="BColumn">That subject's sample column in arm B.</param>
public readonly record struct SamplePair(string Subject, int AColumn, int BColumn);

/// <summary>
/// Matching samples into pairs by a subject key, and saying what could not be matched.
/// </summary>
/// <remarks>
/// Shared by the paired t, the Wilcoxon signed-rank and the paired moderated design, because all
/// three mean the same thing by "paired" and must agree on which subjects took part. A subject that
/// silently took part in one of them and not another would make two numbers on the same screen
/// describe different sample sets.
/// </remarks>
// Public, not internal: the GUI resolves the same pairs to decide which samples the per-feature
// boxplot should draw. Its doc promises every consumer agrees on which subjects took part, and that
// promise is only keepable if they can all call it.
public static class PairedSamples
{
    /// <summary>
    /// The subjects measured exactly once in each arm, in the order they first appear in arm A.
    /// </summary>
    /// <remarks>
    /// <para>Two kinds of subject are dropped, and both are reported rather than absorbed:</para>
    /// <list type="bullet">
    /// <item><b>Present in only one arm.</b> There is nothing to difference it against. Keeping it
    /// would quietly turn a paired analysis into a partly-unpaired one.</item>
    /// <item><b>More than one sample in an arm.</b> Which replicate pairs with which is not
    /// determined by the metadata, and picking one arbitrarily would make the result depend on row
    /// order. A cohort with technical replicates needs them collapsed first, deliberately.</item>
    /// </list>
    /// <para>A sample with no subject value at all cannot be paired either, and is counted
    /// separately because the cause is different - a gap in the metadata column rather than an
    /// unbalanced design.</para>
    /// </remarks>
    public static (IReadOnlyList<SamplePair> Pairs, IReadOnlyList<string> Messages) Resolve(
        IReadOnlyList<string?> subjectLabels,
        IReadOnlyList<int> groupAColumns,
        IReadOnlyList<int> groupBColumns)
    {
        var messages = new List<string>();
        var unlabeled = 0;

        Dictionary<string, List<int>> BySubject(IReadOnlyList<int> cols)
        {
            var map = new Dictionary<string, List<int>>(StringComparer.Ordinal);
            foreach (var c in cols)
            {
                var subject = c < subjectLabels.Count ? subjectLabels[c] : null;
                if (string.IsNullOrEmpty(subject))
                {
                    unlabeled++;
                    continue;
                }

                if (!map.TryGetValue(subject, out var list))
                    map[subject] = list = new List<int>();
                list.Add(c);
            }

            return map;
        }

        var inA = BySubject(groupAColumns);
        var inB = BySubject(groupBColumns);

        var pairs = new List<SamplePair>();
        var ambiguous = new List<string>();
        var unmatched = new List<string>();

        // Seen, not "already paired": the loop is over COLUMNS, so a subject with two samples in an
        // arm comes round twice. Probing `pairs` only skips the ones that succeeded, so a rejected
        // subject was counted - and previewed - once per column it occupied, reporting one ambiguous
        // subject as two.
        var handled = new HashSet<string>(StringComparer.Ordinal);

        foreach (var c in groupAColumns)
        {
            var subject = c < subjectLabels.Count ? subjectLabels[c] : null;
            if (string.IsNullOrEmpty(subject) || !handled.Add(subject))
                continue;

            var a = inA[subject];
            if (!inB.TryGetValue(subject, out var b))
            {
                unmatched.Add(subject);
                continue;
            }

            if (a.Count > 1 || b.Count > 1)
            {
                ambiguous.Add(subject);
                continue;
            }

            pairs.Add(new SamplePair(subject, a[0], b[0]));
        }

        // Subjects seen only in B are unmatched too, and the count is what a reader needs.
        foreach (var subject in inB.Keys)
            if (!inA.ContainsKey(subject))
                unmatched.Add(subject);

        if (unlabeled > 0)
            messages.Add($"{unlabeled} sample(s) have no value in the pairing column and were left out.");
        if (unmatched.Count > 0)
            messages.Add($"{unmatched.Count} subject(s) appear in only one arm and were left out "
                + $"({Preview(unmatched)}).");
        if (ambiguous.Count > 0)
            messages.Add($"{ambiguous.Count} subject(s) have more than one sample in an arm, so the "
                + $"pairing is ambiguous, and were left out ({Preview(ambiguous)}).");

        return (pairs, messages);
    }

    /// <summary>
    /// The sample columns a contrast actually ran over: under a paired design with subject labels,
    /// the matched subjects' columns; otherwise the arms as picked. What the per-feature boxplot and the
    /// quant report's raw-value table show, so neither draws a subject that took no part in the test.
    /// </summary>
    /// <remarks>
    /// A paired design where nothing could be matched gives the picked arms back: the test has then
    /// fallen back too, and showing no samples at all would be the worse answer. A blocked design
    /// leaves out only the samples with no subject, as the test does.
    /// </remarks>
    public static (List<int> A, List<int> B) ColumnsUsed(DifferentialDesign design,
        IReadOnlyList<string?>? subjectLabels, IReadOnlyList<int> pickedA, IReadOnlyList<int> pickedB)
    {
        if (design == DifferentialDesign.BlockedBySubject && subjectLabels is not null)
        {
            var blocked = BlockedSamples.Resolve(subjectLabels, pickedA, pickedB);
            return (blocked.A.ToList(), blocked.B.ToList());
        }

        if (design != DifferentialDesign.Paired || subjectLabels is null)
            return (pickedA.ToList(), pickedB.ToList());

        var (pairs, _) = Resolve(subjectLabels, pickedA, pickedB);
        return pairs.Count == 0
            ? (pickedA.ToList(), pickedB.ToList())
            : (pairs.Select(p => p.AColumn).ToList(), pairs.Select(p => p.BColumn).ToList());
    }

    /// <summary>A few names rather than all of them - a status line with 200 subject ids in it is unreadable.</summary>
    private static string Preview(IReadOnlyList<string> names) => NamePreview.Of(names, show: 3);
}
