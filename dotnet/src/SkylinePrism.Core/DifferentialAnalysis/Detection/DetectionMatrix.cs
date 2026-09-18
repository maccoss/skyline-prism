using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.DifferentialAnalysis.Detection;

/// <summary>A binary peptide x sample detection matrix.</summary>
public sealed class DetectionMatrixData
{
    internal DetectionMatrixData(string[] peptideIds, string[] sampleIds, double[,] matrix)
    {
        PeptideIds = peptideIds;
        SampleIds = sampleIds;
        Matrix = matrix;
    }

    /// <summary>Peptide ids, sorted ascending (matrix rows).</summary>
    public string[] PeptideIds { get; }

    /// <summary>Sample ids, sorted ascending (matrix columns).</summary>
    public string[] SampleIds { get; }

    /// <summary>Detection indicator, <c>[peptide, sample]</c>: 1 if detected, else 0.</summary>
    public double[,] Matrix { get; }
}

/// <summary>
/// Builds the binary peptide x sample detection matrix from a PRISM run's transition-level
/// <c>merged_data</c> (a hash-partitioned parquet dataset), ported from the explorer's
/// <c>cryptic_detection_matrix</c> / <c>cryptic_peptide_map</c> / <c>cryptic_short_label</c>. A cell is
/// 1 iff the peptide was genuinely detected in that sample (<c>DetectionQValue</c> present and below the
/// threshold), the on/off signal the dense abundance matrix cannot give. Reads with DuckDB, filtering
/// server-side.
/// </summary>
public static class DetectionMatrix
{
    /// <summary>
    /// Load the detection matrix from the run's output directory or a <c>merged_data</c> root.
    /// <paramref name="term"/> restricts to peptides whose <c>Protein</c> contains it (the cryptic
    /// flag, e.g. "cryptic"); pass null to cover all peptides. A cell is 1 when
    /// <c>DetectionQValue &lt; qThreshold</c> in that sample.
    /// </summary>
    public static DetectionMatrixData Load(string outputDirOrMergedRoot, double qThreshold = 0.01,
        string? term = "cryptic")
    {
        var dataset = OpenDataset(outputDirOrMergedRoot);
        var cols = ResolveColumns(dataset);
        var thr = qThreshold.ToString(CultureInfo.InvariantCulture);
        var where = $"\"{cols.Peptide}\" IS NOT NULL"
            + (term is null ? string.Empty : $" AND \"{cols.Protein}\" ILIKE '%{Esc(term)}%'");
        var sql =
            $"SELECT \"{cols.Peptide}\" AS pep, \"{cols.Sample}\" AS samp, " +
            $"MAX(CASE WHEN \"{cols.DetectionQ}\" IS NOT NULL AND \"{cols.DetectionQ}\" < {thr} " +
            "THEN 1 ELSE 0 END) AS det " +
            $"FROM {MergedParquetReader.Scan(dataset.ScanTarget)} WHERE {where} GROUP BY pep, samp";

        // Intern peptide/sample strings once and keep only compact (int, int, byte) triples, so a
        // cohort-scale merged_data (tens of millions of pep x samp groups) does not materialize two
        // fresh strings per group.
        var pepIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var pepList = new List<string>();
        var sampIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var sampList = new List<string>();
        var triples = new List<(int P, int S, byte D)>();
        using (var conn = OpenBounded(dataset))
        using (var cmd = DuckDbTuning.StreamingCommand(conn, sql))
        using (var reader = cmd.ExecuteReader())
        {
            while (reader.Read())
            {
                var pep = reader.GetString(0);
                var samp = reader.GetString(1);
                var det = (byte)Convert.ToInt32(reader.GetValue(2), CultureInfo.InvariantCulture);
                if (!pepIndex.TryGetValue(pep, out var pi))
                {
                    pi = pepList.Count;
                    pepIndex[pep] = pi;
                    pepList.Add(pep);
                }

                if (!sampIndex.TryGetValue(samp, out var si))
                {
                    si = sampList.Count;
                    sampIndex[samp] = si;
                    sampList.Add(samp);
                }

                triples.Add((pi, si, det));
            }
        }

        var peptides = pepList.OrderBy(p => p, StringComparer.Ordinal).ToArray();
        var samples = sampList.OrderBy(s => s, StringComparer.Ordinal).ToArray();
        var pepToSorted = new int[pepList.Count];
        var sampToSorted = new int[sampList.Count];
        for (var i = 0; i < peptides.Length; i++)
            pepToSorted[pepIndex[peptides[i]]] = i;
        for (var i = 0; i < samples.Length; i++)
            sampToSorted[sampIndex[samples[i]]] = i;

        var matrix = new double[peptides.Length, samples.Length];
        foreach (var (p, s, d) in triples)
            matrix[pepToSorted[p], sampToSorted[s]] = d;

        return new DetectionMatrixData(peptides, samples, matrix);
    }

    /// <summary>
    /// Map each peptide whose <c>Protein</c> contains <paramref name="term"/> to a representative such
    /// protein string (first occurrence wins). Cryptic peptides carry the flag in their Protein field,
    /// which the protein rollup discards, so this recovers it from the transition table.
    /// </summary>
    public static IReadOnlyDictionary<string, string> CrypticPeptideMap(string outputDirOrMergedRoot,
        string term = "cryptic")
    {
        var dataset = OpenDataset(outputDirOrMergedRoot);
        var cols = ResolveColumns(dataset);
        var sql =
            $"SELECT DISTINCT \"{cols.Peptide}\" AS pep, \"{cols.Protein}\" AS prot " +
            $"FROM {MergedParquetReader.Scan(dataset.ScanTarget)} " +
            $"WHERE \"{cols.Peptide}\" IS NOT NULL AND \"{cols.Protein}\" ILIKE '%{Esc(term)}%'";

        var map = new Dictionary<string, string>();
        using var conn = OpenBounded(dataset);
        using var cmd = DuckDbTuning.StreamingCommand(conn, sql);
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var pep = reader.GetString(0);
            if (!map.ContainsKey(pep)) // setdefault: first occurrence wins
                map[pep] = reader.GetString(1);
        }

        return map;
    }

    /// <summary>
    /// Shorten a cryptic protein string to a readable name (e.g. "S35U4_HUMAN"): the first
    /// pipe-delimited part ending in "_HUMAN", else the second part, else the string itself.
    /// </summary>
    public static string CrypticShortLabel(string? proteinString)
    {
        var parts = (proteinString ?? string.Empty).Split('|');
        foreach (var p in parts)
            if (p.EndsWith("_HUMAN", StringComparison.Ordinal))
                return p;
        return parts.Length > 1 ? parts[1] : proteinString ?? string.Empty;
    }

    private static MergedDataset OpenDataset(string outputDirOrMergedRoot)
    {
        // Accept either a run output directory (holding a merged_data child) or a merged_data root
        // (a directory of _pep_bucket partitions, or a legacy single file) passed directly.
        var root = MergedDataset.Locate(outputDirOrMergedRoot) ?? outputDirOrMergedRoot;
        return MergedDataset.Open(root);
    }

    private static DuckDBConnection OpenBounded(MergedDataset dataset)
    {
        var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(conn, DuckDbMerge.AutoMemoryBudgetMb(),
            DuckDbMerge.ResolveTempDirectory(dataset.Root));
        return conn;
    }

    /// <summary>
    /// Resolve the merged_data column names this loader needs, tolerating naming variants (the current
    /// pipeline writes "PeptideModifiedSequenceUnimodIds"/"DetectionQValue"; older exports use the
    /// spaced Skyline headers "Peptide Modified Sequence Unimod Ids"/"Detection Q Value"). Matching
    /// ignores case, spaces and punctuation.
    /// </summary>
    private static (string Peptide, string Sample, string DetectionQ, string Protein) ResolveColumns(
        MergedDataset dataset)
    {
        var names = ParquetTable.ReadColumnNames(dataset.RepresentativeFile());
        var byNorm = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var n in names)
            byNorm.TryAdd(Normalize(n), n);

        string Find(string logical) => byNorm.TryGetValue(Normalize(logical), out var actual)
            ? actual
            : throw new InvalidOperationException(
                $"merged_data has no column matching '{logical}'. Columns: {string.Join(", ", names)}");

        return (Find("PeptideModifiedSequenceUnimodIds"), Find("Sample ID"),
            Find("DetectionQValue"), Find("Protein"));
    }

    private static string Normalize(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (var ch in s)
            if (char.IsLetterOrDigit(ch))
                sb.Append(char.ToLowerInvariant(ch));
        return sb.ToString();
    }

    private static string Esc(string s) => s.Replace("'", "''");
}
