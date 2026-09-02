using System;
using System.Collections.Generic;
using System.IO;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Which peptides the run kept, and which selected protein lists claim them.
///
/// <para><b>Identity only.</b> Pipeline outputs are read here for their ROW SET and their protein-group
/// columns, never for a value: the magnitudes all come from <c>merged_data/</c>'s raw areas. Reading a
/// number out of <c>corrected_peptides.parquet</c> would make the signal accounting move with
/// normalization and ComBat settings that have nothing to do with what the instrument acquired.</para>
/// </summary>
public static class Ms2SignalPeptides
{
    /// <summary>Group columns the peptide output carries, in the order the matcher wants them.</summary>
    private const string AccessionColumn = "leading_protein";
    private const string GeneColumn = "leading_gene_name";
    private const string NameColumn = "leading_name";

    private static readonly string[] MetaColumns = { "n_transitions", "mean_rt" };

    /// <param name="Classes">Keyed by the peptide string as <c>merged_data/</c> spells it.</param>
    /// <param name="ListNames">Selected lists, aligned with the bits of every
    /// <see cref="Ms2SignalRegions.PeptideClass.ListMask"/> and with
    /// <see cref="Ms2SignalUnion.Result.ListArea"/>.</param>
    /// <param name="PerListPeptides">How many peptides each list claimed. A list that claimed none
    /// produces a zero bar, which is a real answer and has to be distinguishable from a missing one.</param>
    /// <param name="HasGroupColumns">False for a peptide output written before the protein-group columns
    /// existed, in which case no list can be matched and only the assigned total is meaningful.</param>
    public sealed record Classified(
        IReadOnlyDictionary<string, Ms2SignalRegions.PeptideClass> Classes,
        IReadOnlyList<string> ListNames,
        int AssignedPeptides,
        IReadOnlyList<int> PerListPeptides,
        bool HasGroupColumns);

    /// <summary>
    /// Read the assigned peptide set and its list membership out of an output directory.
    ///
    /// <para>The assigned set is the row set of <c>peptides_rollup.parquet</c> - the peptides that
    /// actually reached the peptide matrix. It is not re-derived from <c>transition_rollup</c> settings,
    /// and it is not inferred from matrix VALUES: <c>RollupPreprocess.ImputeAndLog2</c> floors every NaN
    /// and zero, so every cell is positive and a value test would mark everything present.</para>
    /// </summary>
    /// <param name="lists">Selected lists, in the order their bars should appear. More than
    /// <see cref="Ms2SignalUnion.MaxLists"/> is rejected rather than silently truncated.</param>
    public static Classified Classify(string outputDir, IReadOnlyList<ProteinList> lists)
    {
        if (lists.Count > Ms2SignalUnion.MaxLists)
            throw new ArgumentOutOfRangeException(nameof(lists), lists.Count,
                $"At most {Ms2SignalUnion.MaxLists} protein lists can be accounted for at once.");

        var rollup = Path.Combine(outputDir, "peptides_rollup.parquet");
        var assigned = ReadPeptideKeys(rollup);

        var listNames = new List<string>(lists.Count);
        foreach (var list in lists)
            listNames.Add(list.Name);

        var masks = ReadListMasks(
            Path.Combine(outputDir, "corrected_peptides.parquet"), lists, out var hasGroupColumns);

        var classes = new Dictionary<string, Ms2SignalRegions.PeptideClass>(
            assigned.Count, StringComparer.Ordinal);
        var perList = new int[lists.Count];
        foreach (var peptide in assigned)
        {
            var mask = masks.GetValueOrDefault(peptide, 0u);
            classes[peptide] = new Ms2SignalRegions.PeptideClass(true, mask);
            for (var l = 0; l < lists.Count; l++)
                if ((mask & (1u << l)) != 0)
                    perList[l]++;
        }

        return new Classified(classes, listNames, assigned.Count, perList, hasGroupColumns);
    }

    /// <summary>
    /// The peptide keys of a wide peptide parquet, in file order. Only the key column is decoded - the
    /// sample columns are 75,202 x 192 doubles on a real cohort and none of them is wanted here.
    /// </summary>
    private static List<string> ReadPeptideKeys(string path)
    {
        var keys = new List<string>();
        if (!File.Exists(path))
            return keys;

        using var reader = ParquetColumnReader.Open(path);
        var column = PeptideKeyColumn(reader.ColumnNames);
        if (column is null)
            return keys;

        foreach (var value in reader.ReadStrings(column))
            if (!string.IsNullOrEmpty(value))
                keys.Add(value);
        return keys;
    }

    /// <summary>
    /// The peptide identifier column: the first column that is neither known metadata nor a sample.
    /// Sample columns carry the <c>__@__</c> separator <c>DuckDbMerge</c> builds sample ids with, which
    /// is what distinguishes them from a metadata column without hard-coding replicate names.
    /// </summary>
    private static string? PeptideKeyColumn(IReadOnlyList<string> columns)
    {
        foreach (var name in columns)
        {
            if (name.Contains("__@__", StringComparison.Ordinal))
                continue;
            if (name == AccessionColumn || name == GeneColumn || name == NameColumn)
                continue;
            if (Array.IndexOf(MetaColumns, name) >= 0)
                continue;
            return name;
        }
        return null;
    }

    /// <summary>
    /// Peptide to list bitmask, from the group columns only <c>corrected_peptides.parquet</c> carries.
    /// A missing file or missing group columns is not an error - it means no list can be matched, which
    /// the caller reports rather than silently showing empty bars.
    /// </summary>
    private static Dictionary<string, uint> ReadListMasks(
        string path, IReadOnlyList<ProteinList> lists, out bool hasGroupColumns)
    {
        hasGroupColumns = false;
        var masks = new Dictionary<string, uint>(StringComparer.Ordinal);
        if (lists.Count == 0 || !File.Exists(path))
            return masks;

        using var reader = ParquetColumnReader.Open(path);
        var keyColumn = PeptideKeyColumn(reader.ColumnNames);
        if (keyColumn is null)
            return masks;

        var accession = Strings(reader, AccessionColumn);
        var gene = Strings(reader, GeneColumn);
        var name = Strings(reader, NameColumn);
        if (accession is null && gene is null && name is null)
            return masks;   // an older peptide file with no group columns
        hasGroupColumns = true;

        // One matcher per list, never a combined one: every shipped panel ships Visible = false, and a
        // combined matcher honours visibility - so it would match nothing at all.
        var matchers = new ProteinListMatcher[lists.Count];
        for (var l = 0; l < lists.Count; l++)
            matchers[l] = ProteinListSet.MatcherFor(lists[l]);

        var keys = reader.ReadStrings(keyColumn);
        for (var i = 0; i < keys.Length; i++)
        {
            var peptide = keys[i];
            if (string.IsNullOrEmpty(peptide))
                continue;

            uint mask = 0;
            for (var l = 0; l < matchers.Length; l++)
            {
                // A shared peptide names every group it belongs to; being in one of a list's groups is
                // enough, which is what MatchPeptide encodes.
                if (matchers[l].MatchPeptide(Cell(accession, i), Cell(gene, i), Cell(name, i)) is not null)
                    mask |= 1u << l;
            }
            if (mask != 0)
                masks[peptide] = masks.GetValueOrDefault(peptide, 0u) | mask;
        }
        return masks;
    }

    private static string[]? Strings(ParquetColumnReader reader, string column) =>
        reader.HasColumn(column) && !reader.IsNumericColumn(column) ? reader.ReadStrings(column) : null;

    private static string? Cell(string[]? column, int row) =>
        column is null || row >= column.Length ? null : column[row];
}
