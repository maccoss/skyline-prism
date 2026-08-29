using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Rollup;

namespace SkylinePrism.Core.Qc;

/// <summary>What a re-rolled dynamic-range view should compute, independent of what the run configured.</summary>
public sealed class DynamicRangeRollupOptions
{
    public ProteinRollupMethod Method { get; init; } = ProteinRollupMethod.MedianPolish;

    /// <summary>Below this many peptides a group falls back to a linear sum, exactly as Stage 4 does.</summary>
    public int MinPeptides { get; init; } = 3;

    public int TopN { get; init; } = 3;

    public string TopNSelection { get; init; } = "median_abundance";

    /// <summary>
    /// Theoretical peptide counts per leading protein, for iBAQ. Null (or a missing protein) falls back
    /// to the observed count, which is what Stage 4 does when no FASTA is configured - but it makes iBAQ
    /// close to a per-peptide mean rather than an absolute-abundance estimate, so the caller should say
    /// so rather than presenting the two as the same plot.
    /// </summary>
    public IReadOnlyDictionary<string, int>? TheoreticalCounts { get; init; }
}

/// <summary>
/// Re-roll protein abundances from a peptide matrix, so the dynamic-range plot can be viewed under a
/// rollup method the run did not use.
///
/// <para>The plot's shape depends on the rollup in ways that are easy to misread: a summing method scales
/// with how many peptides a protein has, median polish estimates the level of a typical one, and iBAQ
/// divides by the theoretical peptide count and is the only one of them meant for comparing one protein
/// against another. Being able to switch turns that from a caveat in the axis label into something the
/// user can see.</para>
///
/// <para>This is a VIEW, not a re-run. It reads the peptide matrix as written and applies Stage 4's own
/// per-group aggregation (<see cref="ProteinMatrixRollup"/>, the same code path), but it does not re-run
/// parsimony, protein-level normalization or protein-level batch correction. So the numbers match the
/// pipeline's own output in shape and ordering rather than cell for cell, and the caller is expected to
/// label them as recomputed.</para>
/// </summary>
public static class DynamicRangeRollup
{
    /// <summary>
    /// Whether a peptide matrix carries what this needs: the protein-group membership stamped onto
    /// corrected_peptides. A run whose peptide output was disabled, or one from before those columns
    /// existed, cannot be re-rolled - and the caller should say that rather than plotting nothing.
    /// </summary>
    public static bool CanRecompute(ParquetTable peptides) => peptides.HasColumn("protein_group");

    /// <summary>
    /// Roll <paramref name="peptides"/> (a LINEAR corrected peptide matrix) up to protein groups under
    /// <paramref name="options"/> and rank the result, most abundant first.
    /// </summary>
    /// <param name="sampleColumns">Replicates to average; all of them when null or empty.</param>
    /// <param name="progress">Fraction of groups completed, reported occasionally rather than per group.</param>
    public static List<AbundanceEntry> Recompute(
        ParquetTable peptides,
        DynamicRangeRollupOptions options,
        IReadOnlyList<string>? sampleColumns = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var samples = sampleColumns is { Count: > 0 }
            ? sampleColumns.Where(peptides.HasColumn).ToList()
            : DynamicRange.SampleColumns(peptides, AbundanceLevel.Peptide);
        if (samples.Count == 0 || !CanRecompute(peptides))
            return new List<AbundanceEntry>();

        var groups = BuildGroups(peptides);
        if (groups.Count == 0)
            return new List<AbundanceEntry>();

        // Read the replicate columns once, as a dense double[] per sample. The peptide matrix is the
        // largest thing this touches, and every group would otherwise re-index the nullable columns.
        var columns = new double[samples.Count][];
        for (var j = 0; j < samples.Count; j++)
        {
            var raw = peptides.GetDouble(samples[j]);
            var dense = new double[peptides.RowCount];
            for (var i = 0; i < peptides.RowCount; i++)
            {
                // LINEAR in, LOG2 to the rollup. A null, NaN or non-positive cell is a missing
                // measurement rather than a zero, and log2 of it must stay NaN so the aggregation skips
                // it instead of pulling the group down to a floor value.
                var v = raw[i];
                dense[i] = v is { } value && !double.IsNaN(value) && value > 0
                    ? Math.Log2(value)
                    : double.NaN;
            }
            columns[j] = dense;
        }

        var results = new AbundanceEntry?[groups.Count];
        var done = 0;
        // Progress in ~100 steps: a per-group report on 20,000 groups is 20,000 dispatcher hops for a
        // bar that only has so many pixels.
        var step = Math.Max(1, groups.Count / 100);

        Parallel.For(0, groups.Count,
            new ParallelOptions
            {
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount),
            },
            gi =>
            {
                var group = groups[gi];
                var sub = new double[group.Rows.Count, samples.Count];
                for (var p = 0; p < group.Rows.Count; p++)
                {
                    var row = group.Rows[p];
                    for (var j = 0; j < samples.Count; j++)
                        sub[p, j] = columns[j][row];
                }

                var nTheoretical = -1;
                if (options.TheoreticalCounts is not null
                    && group.LeadingProtein is { Length: > 0 } accession
                    && options.TheoreticalCounts.TryGetValue(accession, out var count))
                {
                    nTheoretical = count;
                }

                var log2 = ProteinMatrixRollup.Aggregate(
                    sub, options.Method, options.MinPeptides, options.TopN, nTheoretical,
                    options.TopNSelection);

                // Back to linear and averaged there, matching DynamicRange.Compute - a mean of logs is a
                // geometric mean, which is not the quantity this plot shows.
                double sum = 0;
                var used = 0;
                foreach (var v in log2)
                {
                    if (double.IsNaN(v) || double.IsInfinity(v))
                        continue;
                    var linear = Math.Pow(2, v);
                    if (linear <= 0 || double.IsInfinity(linear))
                        continue;
                    sum += linear;
                    used++;
                }
                if (used > 0)
                {
                    var mean = sum / used;
                    results[gi] = new AbundanceEntry(
                        group.Name, group.Label, group.Accession, group.Gene, group.Name,
                        mean, Math.Log10(mean), 0, used)
                    {
                        ProteinGroups = new[] { group.GroupId },
                        ProteinNames = new[] { group.Name },
                    };
                }

                if (progress is not null && Interlocked.Increment(ref done) % step == 0)
                    progress.Report(Math.Min(1.0, (double)done / groups.Count));
            });

        var entries = results.Where(e => e is not null).Select(e => e!).ToList();
        entries.Sort((a, b) => b.Log10Abundance.CompareTo(a.Log10Abundance));
        for (var i = 0; i < entries.Count; i++)
            entries[i] = entries[i] with { Rank = i + 1 };
        progress?.Report(1.0);
        return entries;
    }

    /// <summary>Which leading proteins the plot will need counts for - the iBAQ digest's accession filter.</summary>
    public static HashSet<string> LeadingProteins(ParquetTable peptides) =>
        BuildGroups(peptides)
            .Select(g => g.LeadingProtein)
            .Where(p => !string.IsNullOrWhiteSpace(p))
            .ToHashSet(StringComparer.Ordinal)!;

    private sealed class GroupRows
    {
        public required string GroupId { get; init; }
        public string Name { get; set; } = "";
        public string Label { get; set; } = "";
        public string? Accession { get; set; }
        public string? Gene { get; set; }
        public string LeadingProtein { get; set; } = "";
        public List<int> Rows { get; } = new();
    }

    /// <summary>
    /// Invert corrected_peptides' group membership: group id -> the peptide rows in it.
    /// <para>
    /// A shared peptide names every group it maps to, ';'-separated, and the identity columns are joined
    /// in the SAME order - so position k of protein_group and of leading_protein describe one group.
    /// Every group a peptide names gets that peptide, which is the pipeline's default
    /// <c>all_groups</c> handling. A run configured for <c>unique_only</c> or <c>razor</c> assigned
    /// fewer, and this view does not reproduce that: it cannot, because which group parsimony gave a
    /// shared peptide to is not recorded per peptide in this file.
    /// </para>
    /// </summary>
    private static List<GroupRows> BuildGroups(ParquetTable peptides)
    {
        var groupIds = peptides.GetString("protein_group");
        var names = peptides.HasColumn("leading_name") ? peptides.GetString("leading_name") : null;
        var accessions = peptides.HasColumn("leading_protein") ? peptides.GetString("leading_protein") : null;
        var genes = peptides.HasColumn("leading_gene_name") ? peptides.GetString("leading_gene_name") : null;

        var byId = new Dictionary<string, GroupRows>(StringComparer.Ordinal);
        var order = new List<GroupRows>();
        for (var row = 0; row < peptides.RowCount; row++)
        {
            var ids = DynamicRange.SplitGroups(At(groupIds, row));
            if (ids.Count == 0)
                continue;
            var rowNames = DynamicRange.SplitGroups(At(names, row));
            var rowAccessions = DynamicRange.SplitGroups(At(accessions, row));
            var rowGenes = DynamicRange.SplitGroups(At(genes, row));

            for (var k = 0; k < ids.Count; k++)
            {
                if (!byId.TryGetValue(ids[k], out var group))
                {
                    group = new GroupRows { GroupId = ids[k] };
                    byId[ids[k]] = group;
                    order.Add(group);
                }
                // Filled from the first peptide that carries them: every peptide of a group repeats the
                // same identity, and a shared peptide may list fewer names than ids if a name was blank.
                if (group.Name.Length == 0)
                {
                    var accession = k < rowAccessions.Count ? rowAccessions[k] : null;
                    var gene = k < rowGenes.Count ? rowGenes[k] : null;
                    var name = k < rowNames.Count ? rowNames[k] : null;
                    group.Name = name ?? accession ?? ids[k];
                    group.Accession = accession;
                    group.Gene = gene;
                    group.LeadingProtein = accession ?? "";
                    // Gene names read on a crowded plot where "sp|P02768|ALBU_HUMAN" does not - the same
                    // preference DynamicRange.ReadIdentity applies to the protein matrix.
                    group.Label = gene ?? accession ?? group.Name;
                }
                group.Rows.Add(row);
            }
        }
        return order;
    }

    private static string? At(string?[]? column, int row) =>
        column is null || row >= column.Length ? null : column[row];
}
