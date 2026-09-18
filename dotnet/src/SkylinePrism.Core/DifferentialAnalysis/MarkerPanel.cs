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
    string[] ColumnLabels,
    double[,] Heatmap,
    double SymmetricMax,
    string[] GroupNames,
    double[][] PanelScoreByGroup,
    IReadOnlyList<string> NotDetected,
    int Found,
    int Total);

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
    public static MarkerPanelResult Evaluate(double[,] exprLog2, string[] featureIds, string[] featureLabels,
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
            if (matcher.Match(featureIds[f], featureLabels[f], null) is not null)
                matched.Add(f);

        var markerLabels = matched
            .Select(f => string.IsNullOrEmpty(featureLabels[f]) ? featureIds[f] : featureLabels[f])
            .ToArray();

        // Per-member detection, for the "found N/total" report.
        var total = 0;
        var notDetected = new List<string>();
        foreach (var member in panel.Members)
        {
            var token = ProteinList.MatchToken(member);
            if (string.IsNullOrEmpty(token))
                continue;
            total++;
            var memberMatcher = ProteinListSet.MatcherFor(new ProteinList { Members = { member } });
            var hit = false;
            for (var f = 0; f < nFeatures && !hit; f++)
                if (memberMatcher.Match(featureIds[f], featureLabels[f], null) is not null)
                    hit = true;
            if (!hit)
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
        if (!perSample)
        {
            heat = new double[matched.Count, groupNames.Length];
            columnLabels = groupNames;
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

        return new MarkerPanelResult(markerLabels, columnLabels, heat, symMax, groupNames,
            panelScoreByGroup, notDetected, found, total);
    }
}
