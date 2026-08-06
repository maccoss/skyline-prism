using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Qc;

/// <summary>Which corrected matrix the dynamic-range plot is built from.</summary>
public enum AbundanceLevel
{
    Protein,
    Peptide,
}

/// <summary>
/// One point on the dynamic-range plot: a protein group or peptide, its abundance averaged over the
/// selected replicates, and its rank within that ordering.
/// </summary>
/// <param name="Key">Identity used to build a Skyline element locator (protein name / peptide sequence).</param>
/// <param name="Label">What a point label shows: gene name, falling back to accession, then Key.</param>
/// <param name="Log10Abundance">log10 of the mean LINEAR abundance - the plot's y value.</param>
/// <param name="Rank">1 = most abundant. The plot's x value.</param>
/// <param name="SamplesUsed">How many of the selected replicates actually had a value.</param>
public sealed record AbundanceEntry(
    string Key,
    string Label,
    string? Accession,
    string? Gene,
    string? ProteinName,
    double MeanAbundance,
    double Log10Abundance,
    int Rank,
    int SamplesUsed)
{
    /// <summary>
    /// Every protein group this entry belongs to. A peptide shared between groups lists them all - which
    /// group "owns" it is a quantification decision, not a fact about the peptide - so the UI can show the
    /// full set rather than implying a single assignment. One entry for a protein row.
    /// </summary>
    public IReadOnlyList<string> ProteinGroups { get; init; } = Array.Empty<string>();

    /// <summary>Protein names of <see cref="ProteinGroups"/>, in the same order.</summary>
    public IReadOnlyList<string> ProteinNames { get; init; } = Array.Empty<string>();

    /// <summary>True when this peptide maps to more than one protein group.</summary>
    public bool IsShared => ProteinGroups.Count > 1;
}

/// <summary>
/// The "Dynamic Range" plot's data: log10 abundance against abundance rank, the shape Skyline calls a
/// Relative Abundance plot. Built from the CORRECTED PRISM matrices only - the whole point is to show the
/// dynamic range of the normalized, batch-corrected result the user will actually analyse.
/// </summary>
/// <remarks>
/// Abundances are averaged on the LINEAR scale and only then log-transformed, because the parquet outputs
/// are linear by contract and a mean of logs is a geometric mean, which is not the quantity being plotted.
/// A protein with no measurement in any selected replicate is dropped rather than plotted at zero.
/// </remarks>
public static class DynamicRange
{
    /// <summary>Non-sample columns of the corrected matrices, by level (everything else is a replicate).</summary>
    public static readonly string[] ProteinMetadataColumns =
    {
        "protein_group", "leading_protein", "leading_name", "leading_uniprot_id", "leading_gene_name",
        "n_peptides", "confidence", "quant_method",
    };

    public static readonly string[] PeptideMetadataColumns =
    {
        "Peptide Modified Sequence Unimod Ids", "Peptide Modified Sequence", "Peptide", "Modified Sequence",
        "n_transitions", "mean_rt",
        // Protein grouping stamped onto the corrected peptide output. These MUST be listed here or they
        // would be mistaken for replicate columns and averaged as abundances.
        "protein_group", "leading_protein", "leading_name", "leading_gene_name",
    };

    /// <summary>
    /// Separator between the protein groups of a shared peptide, matching what the pipeline writes. A
    /// peptide mapping to several groups lists them all.
    /// </summary>
    public const string GroupSeparator = ";";

    /// <summary>First of a ';'-separated list, or null when empty - the group used to identify a peptide.</summary>
    public static string? FirstGroup(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;
        var first = value!.Split(GroupSeparator, StringSplitOptions.RemoveEmptyEntries).FirstOrDefault();
        return string.IsNullOrWhiteSpace(first) ? null : first.Trim();
    }

    /// <summary>Replicate columns of a corrected matrix: everything that is not a known metadata column.</summary>
    public static List<string> SampleColumns(ParquetTable table, AbundanceLevel level)
    {
        var meta = new HashSet<string>(
            level == AbundanceLevel.Protein ? ProteinMetadataColumns : PeptideMetadataColumns,
            StringComparer.OrdinalIgnoreCase);
        return table.ColumnNames.Where(c => !meta.Contains(c)).ToList();
    }

    /// <summary>
    /// Rank the rows of a corrected matrix by mean abundance over <paramref name="sampleColumns"/>
    /// (all replicates when empty). Returns entries ordered most abundant first, rank 1..N.
    /// </summary>
    public static List<AbundanceEntry> Compute(
        ParquetTable table, AbundanceLevel level, IReadOnlyList<string>? sampleColumns = null)
    {
        var samples = sampleColumns is { Count: > 0 }
            ? sampleColumns.Where(table.HasColumn).ToList()
            : SampleColumns(table, level);
        if (samples.Count == 0)
            return new List<AbundanceEntry>();

        var columns = samples.Select(table.GetDouble).ToList();
        var identity = ReadIdentity(table, level);

        var entries = new List<AbundanceEntry>(table.RowCount);
        for (var row = 0; row < table.RowCount; row++)
        {
            double sum = 0;
            var used = 0;
            foreach (var column in columns)
            {
                var v = column[row];
                // Linear abundances: a null, NaN or non-positive value is a missing measurement, not a
                // zero-abundance protein, and averaging it in would drag the whole curve down.
                if (v is { } value && !double.IsNaN(value) && value > 0)
                {
                    sum += value;
                    used++;
                }
            }
            if (used == 0)
                continue;

            var mean = sum / used;
            var id = identity(row);
            entries.Add(new AbundanceEntry(
                id.Key, id.Label, id.Accession, id.Gene, id.ProteinName,
                mean, Math.Log10(mean), 0, used)
            {
                ProteinGroups = id.Groups,
                ProteinNames = id.ProteinNames,
            });
        }

        entries.Sort((a, b) => b.Log10Abundance.CompareTo(a.Log10Abundance));
        for (var i = 0; i < entries.Count; i++)
            entries[i] = entries[i] with { Rank = i + 1 };
        return entries;
    }

    private readonly record struct Identity(
        string Key, string Label, string? Accession, string? Gene, string? ProteinName,
        IReadOnlyList<string> Groups, IReadOnlyList<string> ProteinNames);

    /// <summary>Split a ';'-separated multi-group value into its parts.</summary>
    public static IReadOnlyList<string> SplitGroups(string? value) =>
        string.IsNullOrWhiteSpace(value)
            ? Array.Empty<string>()
            : value!.Split(GroupSeparator, StringSplitOptions.RemoveEmptyEntries)
                .Select(v => v.Trim())
                .Where(v => v.Length > 0)
                .ToArray();

    private static Func<int, Identity> ReadIdentity(ParquetTable table, AbundanceLevel level)
    {
        if (level == AbundanceLevel.Protein)
        {
            var names = Strings(table, "leading_name", "leading_protein", "protein_group");
            var accessions = Strings(table, "leading_uniprot_id", "leading_protein");
            var genes = Strings(table, "leading_gene_name");
            var groups = Strings(table, "protein_group");
            return row =>
            {
                var name = Value(names, row) ?? Value(groups, row) ?? $"row{row}";
                var accession = Value(accessions, row);
                var gene = Value(genes, row);
                var group = Value(groups, row);
                // Labels prefer the gene name: "ALB" reads on a crowded plot where
                // "sp|P02768|ALBU_HUMAN" does not. Accession, then the name, are the fallbacks.
                return new Identity(
                    name, gene ?? accession ?? name, accession, gene, name,
                    group is null ? Array.Empty<string>() : new[] { group },
                    new[] { name });
            };
        }

        var peptides = Strings(
            table, "Peptide Modified Sequence Unimod Ids", "Peptide Modified Sequence", "Modified Sequence",
            "Peptide");
        var proteinNames = Strings(table, "leading_name");
        var peptideAccessions = Strings(table, "leading_protein");
        var peptideGenes = Strings(table, "leading_gene_name");
        var peptideGroups = Strings(table, "protein_group");
        return row =>
        {
            var peptide = Value(peptides, row) ?? $"row{row}";
            // A shared peptide maps to several groups. All of them are carried (so the UI can show every
            // protein it is present in); the FIRST identifies it for navigation and list matching.
            var allGroups = SplitGroups(Value(peptideGroups, row));
            var allNames = SplitGroups(Value(proteinNames, row));
            var protein = allNames.FirstOrDefault();
            var accession = FirstGroup(Value(peptideAccessions, row));
            var gene = FirstGroup(Value(peptideGenes, row));
            // The label stays the sequence - a peptide plot labelled by gene would collide for every
            // peptide of the same protein.
            return new Identity(peptide, peptide, accession, gene, protein, allGroups, allNames);
        };
    }

    // First of the candidate columns that exists, as strings; null when none do.
    private static string?[]? Strings(ParquetTable table, params string[] candidates)
    {
        var actual = SkylineColumns.FindColumn(table.ColumnNames.ToList(), candidates);
        return actual is null ? null : table.GetString(actual);
    }

    private static string? Value(string?[]? column, int row)
    {
        if (column is null || row >= column.Length)
            return null;
        var v = column[row];
        return string.IsNullOrWhiteSpace(v) ? null : v;
    }
}
