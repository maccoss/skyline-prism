using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// A marker-panel evaluation: the row z-scored log2 abundance of a protein list's members, summarized as
/// a marker x group (or marker x sample) heatmap, plus a per-group panel-score distribution for a
/// boxplot. Ported from the explorer's "EV markers" tab, generalized to any protein list and any grouping
/// column. Matching is by the same rules as the Dynamic Range protein lists (accession / gene / name).
/// </summary>
public sealed record MarkerPanelResult(
    string[] MarkerLabels,
    string[] MarkerFeatureIds,
    string[] ColumnLabels,
    double[,] Heatmap,
    double SymmetricMax,
    string[] GroupNames,
    double[][] PanelScoreByGroup,
    IReadOnlyList<string> NotDetected,
    int Found,
    int Total)
{
    /// <summary>
    /// The group of each heatmap column, as an index into <see cref="GroupNames"/>: one column per
    /// group in the group-means view, one per sample - ordered by group - in the per-sample view.
    /// </summary>
    /// <remarks>
    /// What lets a per-sample heatmap name its GROUPS under each run of columns rather than every
    /// sample id, which at cohort size is unreadable and is not what the view is read for. An
    /// init-only property, not a positional one, so every existing construction keeps compiling.
    /// </remarks>
    public int[] ColumnGroups { get; init; } = Array.Empty<int>();
}

/// <summary>Builds a <see cref="MarkerPanelResult"/> from a loaded differential matrix and a protein list.</summary>
public static class MarkerPanel
{
    /// <summary>
    /// Evaluate <paramref name="panel"/> against the LOG2 matrix. Samples whose grouping label is null or
    /// empty are excluded. Each matched feature's row is z-scored across the included samples (population
    /// sd); a row with fewer than two finite values, or zero variance, is left as NaN. When
    /// <paramref name="perSample"/> is false the heatmap columns are group means, otherwise individual
    /// samples ordered by group. The boxplot data is, per group, each sample's mean marker z-score.
    /// </summary>
    /// <remarks>
    /// <paramref name="identities"/> rather than a pair of id/label arrays because a panel is written
    /// in accessions, genes or protein names and a feature must be findable under every one of them.
    /// Matching on (id, label) alone meant a protein row was offered only its group id and gene - so
    /// an accession- or name-based list matched nothing - and a peptide row was offered its modified
    /// sequence as BOTH fields, which is neither. Valid panels reported their members as not
    /// detected. <see cref="FeatureIdentity"/> carries what the matcher needs at both levels.
    /// </remarks>
    public static MarkerPanelResult Evaluate(double[,] exprLog2,
        IReadOnlyList<FeatureIdentity> identities,
        string?[] sampleGroups, string[] sampleIds, ProteinList panel, bool perSample)
    {
        var nFeatures = exprLog2.GetLength(0);
        var nSamples = exprLog2.GetLength(1);

        // Included samples: those with a grouping label.
        var included = new List<int>();
        for (var s = 0; s < nSamples; s++)
            if (!string.IsNullOrEmpty(sampleGroups[s]))
                included.Add(s);

        // Matched features.
        var matcher = ProteinListSet.MatcherFor(panel);
        var matched = new List<int>();
        for (var f = 0; f < nFeatures; f++)
            if (Claims(matcher, identities[f]))
                matched.Add(f);

        var markerLabels = matched
            .Select(f => string.IsNullOrEmpty(identities[f].Label) ? identities[f].FeatureId : identities[f].Label)
            .ToArray();
        var markerFeatureIds = matched.Select(f => identities[f].FeatureId).ToArray();

        // Per-member detection, for the "found N/total" report.
        //
        // Inverted: collect every identifier the FEATURES carry once, then test each member against
        // that set. The straightforward loop is members x features with a matcher built per member,
        // which is 1,273 members (all 65 shipped panels) against a peptide matrix of tens of
        // thousands - tens of millions of comparisons on the UI thread. The test itself is
        // unchanged, because both sides still come from ProteinListMatcher.
        var present = new HashSet<string>(ProteinListMatcher.TokenComparer);
        for (var f = 0; f < nFeatures; f++)
        {
            var id = identities[f];
            foreach (var a in id.Accessions)
                present.UnionWith(ProteinListMatcher.CandidatesFor(a, null, null));
            foreach (var g in id.Genes)
                present.UnionWith(ProteinListMatcher.CandidatesFor(null, g, null));
            foreach (var n in id.ProteinNames)
                present.UnionWith(ProteinListMatcher.CandidatesFor(null, null, n));
            // The id/label fallback, matching what Claims tests second.
            present.UnionWith(ProteinListMatcher.CandidatesFor(id.FeatureId, id.Label, null));
        }

        var total = 0;
        var notDetected = new List<string>();
        foreach (var member in panel.Members)
        {
            var token = ProteinList.MatchToken(member);
            if (string.IsNullOrEmpty(token))
                continue;
            total++;
            if (!ProteinListMatcher.MemberTokens(member).Any(present.Contains))
                notDetected.Add(ProteinList.DisplayName(member));
        }

        var found = total - notDetected.Count;

        // Row z-scores over the included samples: z[m, includedIndex].
        var z = new double[matched.Count, included.Count];
        for (var m = 0; m < matched.Count; m++)
        {
            var f = matched[m];
            var vals = new List<double>(included.Count);
            foreach (var s in included)
            {
                var v = exprLog2[f, s];
                if (double.IsFinite(v))
                    vals.Add(v);
            }

            double mean = 0, sd = 0;
            if (vals.Count >= 2)
            {
                mean = vals.Average();
                sd = Math.Sqrt(vals.Sum(v => (v - mean) * (v - mean)) / vals.Count); // population sd
            }

            for (var k = 0; k < included.Count; k++)
            {
                var v = exprLog2[f, included[k]];
                z[m, k] = double.IsFinite(v) && sd > 0 ? (v - mean) / sd : double.NaN;
            }
        }

        // Distinct groups, sorted.
        var groupNames = included.Select(s => sampleGroups[s]!).Distinct()
            .OrderBy(g => g, StringComparer.Ordinal).ToArray();
        var groupOf = new int[included.Count];
        var groupIndex = groupNames.Select((g, i) => (g, i)).ToDictionary(x => x.g, x => x.i);
        for (var k = 0; k < included.Count; k++)
            groupOf[k] = groupIndex[sampleGroups[included[k]]!];

        // Heatmap.
        double[,] heat;
        string[] columnLabels;
        int[] columnGroups;
        if (!perSample)
        {
            heat = new double[matched.Count, groupNames.Length];
            columnLabels = groupNames;
            columnGroups = Enumerable.Range(0, groupNames.Length).ToArray();
            for (var m = 0; m < matched.Count; m++)
                for (var g = 0; g < groupNames.Length; g++)
                {
                    double sum = 0;
                    var n = 0;
                    for (var k = 0; k < included.Count; k++)
                        if (groupOf[k] == g && double.IsFinite(z[m, k]))
                        {
                            sum += z[m, k];
                            n++;
                        }

                    heat[m, g] = n > 0 ? sum / n : double.NaN;
                }
        }
        else
        {
            // Columns are samples ordered by group, then by sample id.
            var order = Enumerable.Range(0, included.Count)
                .OrderBy(k => groupOf[k])
                .ThenBy(k => sampleIds[included[k]], StringComparer.Ordinal)
                .ToArray();
            heat = new double[matched.Count, order.Length];
            columnLabels = order.Select(k => sampleIds[included[k]]).ToArray();
            columnGroups = order.Select(k => groupOf[k]).ToArray();
            for (var m = 0; m < matched.Count; m++)
                for (var c = 0; c < order.Length; c++)
                    heat[m, c] = z[m, order[c]];
        }

        // Boxplot: per group, each sample's mean marker z-score.
        var panelScoreByGroup = new double[groupNames.Length][];
        var buckets = new List<double>[groupNames.Length];
        for (var g = 0; g < groupNames.Length; g++)
            buckets[g] = new List<double>();
        for (var k = 0; k < included.Count; k++)
        {
            double sum = 0;
            var n = 0;
            for (var m = 0; m < matched.Count; m++)
                if (double.IsFinite(z[m, k]))
                {
                    sum += z[m, k];
                    n++;
                }

            if (n > 0)
                buckets[groupOf[k]].Add(sum / n);
        }

        for (var g = 0; g < groupNames.Length; g++)
            panelScoreByGroup[g] = buckets[g].ToArray();

        // Symmetric color range: the largest finite magnitude in the heatmap.
        var symMax = 0.0;
        foreach (var v in heat)
            if (double.IsFinite(v) && Math.Abs(v) > symMax)
                symMax = Math.Abs(v);
        if (symMax <= 0)
            symMax = 1.0;

        return new MarkerPanelResult(markerLabels, markerFeatureIds, columnLabels, heat, symMax, groupNames,
            panelScoreByGroup, notDetected, found, total)
        {
            ColumnGroups = columnGroups,
        };
    }

    /// <summary>
    /// Whether the list claims this feature, under any accession, gene or protein name it carries -
    /// plus its own id and display label, which is how a panel written in protein-group ids or in
    /// peptide sequences still matches.
    /// </summary>
    private static bool Claims(ProteinListMatcher matcher, FeatureIdentity identity)
        => matcher.MatchAny(identity.Accessions, identity.Genes, identity.ProteinNames) is not null
           || matcher.Match(identity.FeatureId, identity.Label, null) is not null;
}
