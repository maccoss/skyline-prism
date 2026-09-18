using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>How a metadata column's levels are partitioned into contrasts for the Finder scan.</summary>
public enum ScanScope
{
    /// <summary>Each level versus all the others.</summary>
    OneVsRest,

    /// <summary>Every unordered pair of levels.</summary>
    Pairwise,

    /// <summary>Every disjoint non-empty (A, B) subset pair; levels may be excluded.</summary>
    SubsetPairs,
}

/// <summary>One contrast's Finder result.</summary>
public sealed record ScanRow(
    string A,
    string B,
    int NA,
    int NB,
    int NSig,
    double MinQ,
    IReadOnlyList<string> ALevels,
    IReadOnlyList<string> BLevels);

/// <summary>Result of <see cref="SignificanceScan.Run"/>: ranked contrasts and whether the scan was truncated.</summary>
public sealed class ScanResult
{
    internal ScanResult(IReadOnlyList<ScanRow> rows, bool truncated)
    {
        Rows = rows;
        Truncated = truncated;
    }

    /// <summary>Contrasts ranked by significant-feature count (desc) then smallest adjusted p (asc).</summary>
    public IReadOnlyList<ScanRow> Rows { get; }

    /// <summary>True if the scan hit its <c>maxTests</c> budget.</summary>
    public bool Truncated { get; }
}

/// <summary>Headline counts for a <see cref="ScanResult"/>.</summary>
public sealed record ScanSummary(int NContrasts, int BestNSig, int NWithHits, string? BestA, string? BestB);

/// <summary>
/// The Finder: run the moderated-t contrast (<see cref="Differential"/>) over many partitions of a
/// metadata column and rank them by how many features come out significant. Ported from the PRISM
/// Differential Explorer's <c>significance_scan</c> / <c>_enumerate_splits</c> / <c>scan_summary</c>.
/// This is an exploratory search over contrasts and inflates false positives; calibrate a winner with
/// the permutation nulls (<see cref="ScanPermutationNull"/>) and validate on independent data.
/// </summary>
public static class SignificanceScan
{
    /// <summary>
    /// Scan the partitions of <paramref name="sampleLabels"/> (a per-column metadata value, aligned to
    /// the abundance matrix columns) at LOG2. <paramref name="poolColumns"/> restricts the samples
    /// considered (e.g. experimental only); null uses all columns. A level must have at least
    /// <paramref name="minN"/> samples to take part, and each contrast arm at least <paramref name="minN"/>.
    /// </summary>
    public static ScanResult Run(
        double[,] exprLog2FeaturesBySamples,
        IReadOnlyList<string> featureIds,
        IReadOnlyList<string?> sampleLabels,
        IReadOnlyList<int>? poolColumns,
        ScanScope scope,
        double qCut = 0.05,
        double lfcCut = 1.0,
        int minN = 3,
        int maxTests = 400,
        IReadOnlyList<Covariate>? covariates = null)
    {
        var pool = poolColumns ?? Enumerable.Range(0, exprLog2FeaturesBySamples.GetLength(1)).ToList();

        // Levels present in the pool with >= minN members, ordered by count desc then first appearance
        // (matching pandas value_counts tie order).
        var counts = new Dictionary<string, int>();
        var firstSeen = new Dictionary<string, int>();
        var order = 0;
        var idsByLevel = new Dictionary<string, List<int>>();
        foreach (var c in pool)
        {
            var label = sampleLabels[c];
            if (label is null)
                continue;
            if (!counts.ContainsKey(label))
            {
                counts[label] = 0;
                firstSeen[label] = order++;
                idsByLevel[label] = new List<int>();
            }

            counts[label]++;
            idsByLevel[label].Add(c);
        }

        var levels = counts.Keys
            .Where(lv => counts[lv] >= minN)
            .OrderByDescending(lv => counts[lv])
            .ThenBy(lv => firstSeen[lv])
            .ToList();

        var rows = new List<ScanRow>();
        var nDone = 0;
        var truncated = false;
        foreach (var (aLevels, bLevels) in EnumerateSplits(levels, scope, maxTests))
        {
            var ga = aLevels.SelectMany(lv => idsByLevel[lv]).ToList();
            var gb = bLevels.SelectMany(lv => idsByLevel[lv]).ToList();
            if (ga.Count < minN || gb.Count < minN)
                continue;

            nDone++;
            if (nDone > maxTests)
            {
                truncated = true;
                break;
            }

            DifferentialResult res;
            try
            {
                res = Differential.Run(exprLog2FeaturesBySamples, featureIds, ga, gb, minN, covariates);
            }
            catch (Exception)
            {
                continue; // a degenerate split is skipped, not fatal (matches the reference's except Exception)
            }

            var nSig = res.Rows.Count(r => r.AdjPValue < qCut && Math.Abs(r.LogFc) >= lfcCut);
            var minQ = res.Rows.Min(r => r.AdjPValue);
            rows.Add(new ScanRow(string.Join(" + ", aLevels), string.Join(" + ", bLevels),
                ga.Count, gb.Count, nSig, minQ, aLevels, bLevels));
        }

        var ranked = rows
            .OrderByDescending(r => r.NSig)
            .ThenBy(r => r.MinQ)
            .ToList();

        return new ScanResult(ranked, truncated);
    }

    /// <summary>Headline counts for a scan result (rows already ranked by <see cref="SignificanceScan.Run"/>).</summary>
    public static ScanSummary Summarize(ScanResult scan)
    {
        if (scan.Rows.Count == 0)
            return new ScanSummary(0, 0, 0, null, null);

        var top = scan.Rows[0];
        return new ScanSummary(
            scan.Rows.Count,
            scan.Rows.Max(r => r.NSig),
            scan.Rows.Count(r => r.NSig > 0),
            top.A,
            top.B);
    }

    /// <summary>
    /// Enumerate (A-levels, B-levels) partitions for the scope. SubsetPairs assigns each level to
    /// A / B / excluded (base-3, level 0 most significant, matching itertools.product order), skips
    /// empty arms, de-duplicates A-B mirror images, and stops after <paramref name="maxTests"/>.
    /// </summary>
    public static IEnumerable<(IReadOnlyList<string> A, IReadOnlyList<string> B)> EnumerateSplits(
        IReadOnlyList<string> levels, ScanScope scope, int maxTests)
    {
        var k = levels.Count;
        switch (scope)
        {
            case ScanScope.OneVsRest:
                foreach (var lv in levels)
                    yield return (new[] { lv }, levels.Where(x => x != lv).ToArray());
                break;

            case ScanScope.Pairwise:
                for (var i = 0; i < k; i++)
                for (var j = i + 1; j < k; j++)
                    yield return (new[] { levels[i] }, new[] { levels[j] });
                break;

            default: // SubsetPairs
            {
                var count = 0;
                var seen = new HashSet<string>();
                long total = 1;
                for (var d = 0; d < k; d++)
                    total *= 3;

                for (long code = 0; code < total; code++)
                {
                    var a = new List<string>();
                    var b = new List<string>();
                    var rem = code;
                    // Extract base-3 digits with level 0 most significant.
                    for (var pos = 0; pos < k; pos++)
                    {
                        var place = 1L;
                        for (var d = 0; d < k - 1 - pos; d++)
                            place *= 3;
                        var digit = (int)(rem / place % 3);
                        if (digit == 0)
                            a.Add(levels[pos]);
                        else if (digit == 1)
                            b.Add(levels[pos]);
                    }

                    if (a.Count == 0 || b.Count == 0)
                        continue;

                    var key = MirrorKey(a, b);
                    if (!seen.Add(key))
                        continue;

                    yield return (a, b);
                    count++;
                    if (count >= maxTests)
                        yield break;
                }

                break;
            }
        }
    }

    /// <summary>
    /// Null distribution of the significant-feature count under random label shuffles of one split's
    /// samples (keeping the group sizes). Calibrates a single contrast, not the search over many
    /// partitions - use <see cref="ScanPermutationNull"/> for that. Ported from the explorer's
    /// <c>permute_calibrate</c>; the shuffle is a seeded C# Fisher-Yates (reproducible and
    /// method-equivalent to the reference, though not identical to numpy's PCG64 draws).
    /// </summary>
    public static int[] PermuteCalibrate(
        double[,] exprLog2FeaturesBySamples,
        IReadOnlyList<string> featureIds,
        IReadOnlyList<int> groupAColumns,
        IReadOnlyList<int> groupBColumns,
        double qCut = 0.05,
        double lfcCut = 1.0,
        int nPerm = 200,
        int minN = 3,
        int seed = 12345,
        IReadOnlyList<Covariate>? covariates = null)
    {
        var pool = groupAColumns.Concat(groupBColumns).ToList();
        var nA = groupAColumns.Count;
        var rng = new Random(seed);
        var null_ = new List<int>(nPerm);
        for (var iter = 0; iter < nPerm; iter++)
        {
            Shuffle(pool, rng);
            var pa = pool.Take(nA).ToList();
            var pb = pool.Skip(nA).ToList();
            try
            {
                var res = Differential.Run(exprLog2FeaturesBySamples, featureIds, pa, pb, minN, covariates);
                null_.Add(res.Rows.Count(r => r.AdjPValue < qCut && Math.Abs(r.LogFc) >= lfcCut));
            }
            catch (Exception)
            {
                // skip a degenerate shuffle, as the reference does (except Exception)
            }
        }

        return null_.ToArray();
    }

    /// <summary>
    /// Family-wise null for a Finder search: permute the column labels across the pool, re-run the
    /// whole scan, and record the maximum significant-feature count per permutation. The observed top
    /// hit's empirical p against this null accounts for the selection over all partitions. Ported from
    /// the explorer's <c>scan_permutation_null</c> (seeded C# shuffle, see <see cref="PermuteCalibrate"/>).
    /// </summary>
    public static int[] ScanPermutationNull(
        double[,] exprLog2FeaturesBySamples,
        IReadOnlyList<string> featureIds,
        IReadOnlyList<string?> sampleLabels,
        IReadOnlyList<int>? poolColumns,
        ScanScope scope,
        double qCut = 0.05,
        double lfcCut = 1.0,
        int minN = 3,
        int maxTests = 400,
        int nPerm = 100,
        int seed = 12345,
        IReadOnlyList<Covariate>? covariates = null)
    {
        var ids = poolColumns ?? Enumerable.Range(0, exprLog2FeaturesBySamples.GetLength(1)).ToList();
        var labels = ids.Select(c => sampleLabels[c]).ToList();
        var rng = new Random(seed);
        var nullMax = new int[nPerm];
        for (var iter = 0; iter < nPerm; iter++)
        {
            Shuffle(labels, rng);
            var permuted = sampleLabels.ToArray();
            for (var i = 0; i < ids.Count; i++)
                permuted[ids[i]] = labels[i];

            var scan = Run(exprLog2FeaturesBySamples, featureIds, permuted, poolColumns, scope,
                qCut, lfcCut, minN, maxTests, covariates);
            nullMax[iter] = scan.Rows.Count > 0 ? scan.Rows.Max(r => r.NSig) : 0;
        }

        return nullMax;
    }

    /// <summary>In-place Fisher-Yates shuffle using the given generator.</summary>
    private static void Shuffle<T>(IList<T> list, Random rng)
    {
        for (var i = list.Count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    /// <summary>Order-independent key for an {A, B} pair so A-vs-B and B-vs-A collapse to one.</summary>
    private static string MirrorKey(List<string> a, List<string> b)
    {
        // NUL / SOH separators so a level name containing ',' or '|' cannot forge a collision.
        var sa = string.Join('\0', a.OrderBy(x => x, StringComparer.Ordinal));
        var sb = string.Join('\0', b.OrderBy(x => x, StringComparer.Ordinal));
        return string.CompareOrdinal(sa, sb) <= 0 ? sa + '' + sb : sb + '' + sa;
    }
}
