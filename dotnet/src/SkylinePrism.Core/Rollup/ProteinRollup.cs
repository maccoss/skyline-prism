using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Parsimony;

namespace SkylinePrism.Core.Rollup;

/// <summary>Configuration for the peptide-&gt;protein rollup (Stage 4).</summary>
public sealed class ProteinRollupConfig
{
    public ProteinRollupMethod Method { get; init; } = ProteinRollupMethod.MedianPolish;
    public int MinPeptides { get; init; } = 3;

    /// <summary>Peptides to average for the topn method.</summary>
    public int TopN { get; init; } = 3;

    /// <summary>Peptide selection for the topn method: "median_abundance" (default) or "frequency".</summary>
    public string TopNSelection { get; init; } = "median_abundance";

    /// <summary>
    /// Which peptide set feeds each protein: "all_groups" (all mapped peptides, shared go to every
    /// group), "unique_only" (unique peptides only), or "razor" (parsimony-assigned = unique + razor).
    /// </summary>
    public string SharedPeptideHandling { get; init; } = "all_groups";
}

/// <summary>
/// Stage 4 peptide-&gt;protein rollup driver, porting chunked_processing.rollup_proteins_streaming
/// + _process_single_protein. Reads the LOG2 peptide matrix (peptides_log2_internal), rolls up
/// each protein group's available peptides, and writes the wide LOG2 proteins_raw parquet.
/// </summary>
public sealed class ProteinRollup
{
    private static readonly string[] MetaCols =
        { "n_transitions", "mean_rt" };

    public sealed record Result(int NProteins, int NSkipped, IReadOnlyList<string> Samples);

    public static Result Run(
        string peptideLog2Parquet,
        IReadOnlyList<ProteinGroup> groups,
        ProteinRollupConfig cfg,
        string peptideCol,
        string outputPath,
        IReadOnlyList<string>? samples = null,
        IReadOnlyDictionary<string, int>? theoreticalCounts = null,
        int maxDegreeOfParallelism = 0)
    {
        var table = ParquetTable.Load(peptideLog2Parquet);

        var meta = new HashSet<string>(MetaCols, StringComparer.Ordinal) { peptideCol };
        var sampleCols = samples?.ToList()
            ?? table.ColumnNames.Where(c => !meta.Contains(c)).ToList();

        // Peptide matrix: peptide -> row of sample values.
        var pepKeys = table.GetString(peptideCol);
        var pepIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < pepKeys.Length; i++)
            pepIndex[pepKeys[i]!] = i;

        var sampleData = sampleCols.Select(table.GetDouble).ToList();

        // Per-group rollup is pure (reads the shared peptide matrix read-only, allocates its own
        // submatrix), so groups run in parallel; results are written into a preallocated array by
        // group index to preserve order without locking.
        var results = new ProteinRow?[groups.Count];
        var nSkipped = 0;
        var dop = maxDegreeOfParallelism <= 0
            ? Math.Max(1, Environment.ProcessorCount)
            : Math.Min(maxDegreeOfParallelism, Math.Max(1, Environment.ProcessorCount));

        Parallel.For(0, groups.Count, new ParallelOptions { MaxDegreeOfParallelism = dop }, gi =>
        {
            var group = groups[gi];
            var source = cfg.SharedPeptideHandling switch
            {
                "unique_only" => group.UniquePeptides,
                "razor" => group.Peptides, // parsimony-assigned = unique + razor
                _ => group.AllMappedPeptides,
            };
            var available = source.Where(pepIndex.ContainsKey).ToList();
            if (available.Count == 0)
            {
                Interlocked.Increment(ref nSkipped);
                return;
            }

            // Submatrix [nAvailable, nSamples] in group peptide order.
            var sub = new double[available.Count, sampleCols.Count];
            for (var a = 0; a < available.Count; a++)
            {
                var ri = pepIndex[available[a]];
                for (var j = 0; j < sampleCols.Count; j++)
                    sub[a, j] = sampleData[j][ri] ?? double.NaN;
            }

            var nTheo = -1;
            if (theoreticalCounts is not null && theoreticalCounts.TryGetValue(group.LeadingProtein, out var c))
                nTheo = c;
            var vals = ProteinMatrixRollup.Aggregate(sub, cfg.Method, cfg.MinPeptides, cfg.TopN, nTheo, cfg.TopNSelection);

            results[gi] = new ProteinRow
            {
                Group = group,
                NPeptides = available.Count,
                NUniquePeptides = group.UniquePeptides.Count,
                LowConfidence = available.Count < cfg.MinPeptides,
                Values = vals,
            };
        });

        var rows = results.Where(r => r is not null).Select(r => r!).ToList();
        WriteOutput(outputPath, sampleCols, rows);
        return new Result(rows.Count, nSkipped, sampleCols);
    }

    private sealed class ProteinRow
    {
        public required ProteinGroup Group { get; init; }
        public required int NPeptides { get; init; }
        public required int NUniquePeptides { get; init; }
        public required bool LowConfidence { get; init; }
        public required double[] Values { get; init; }
    }

    private static void WriteOutput(string outputPath, IReadOnlyList<string> samples, List<ProteinRow> rows)
    {
        var n = rows.Count;
        var groupId = new string[n];
        var leadingProtein = new string[n];
        var leadingName = new string[n];
        var leadingUniprot = new string[n];
        var leadingGene = new string[n];
        var leadingDesc = new string[n];
        var nPeptides = new long[n];
        var nUnique = new long[n];
        var lowConf = new bool[n];
        var sampleColumns = new double[samples.Count][];
        for (var s = 0; s < samples.Count; s++)
            sampleColumns[s] = new double[n];

        for (var r = 0; r < n; r++)
        {
            var row = rows[r];
            groupId[r] = row.Group.GroupId;
            leadingProtein[r] = row.Group.LeadingProtein;
            leadingName[r] = row.Group.LeadingName;
            leadingUniprot[r] = row.Group.LeadingUniProtId;
            leadingGene[r] = row.Group.LeadingGeneName;
            leadingDesc[r] = row.Group.LeadingDescription;
            nPeptides[r] = row.NPeptides;
            nUnique[r] = row.NUniquePeptides;
            lowConf[r] = row.LowConfidence;
            for (var s = 0; s < samples.Count; s++)
                sampleColumns[s][r] = row.Values[s];
        }

        var metaColumns = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("protein_group", groupId),
            ParquetWideWriter.Strings("leading_protein", leadingProtein),
            ParquetWideWriter.Strings("leading_name", leadingName),
            ParquetWideWriter.Strings("leading_uniprot_id", leadingUniprot),
            ParquetWideWriter.Strings("leading_gene_name", leadingGene),
            ParquetWideWriter.Strings("leading_description", leadingDesc),
            ParquetWideWriter.Longs("n_peptides", nPeptides),
            ParquetWideWriter.Longs("n_unique_peptides", nUnique),
            ParquetWideWriter.Bools("low_confidence", lowConf),
        };
        ParquetWideWriter.Write(outputPath, metaColumns, samples, sampleColumns, n);
    }
}
