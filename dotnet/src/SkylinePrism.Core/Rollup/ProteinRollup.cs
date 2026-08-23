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

    /// <summary>
    /// Where to write the per-peptide median-polish residuals, or null to not write them.
    /// <para>
    /// Only <see cref="ProteinRollupMethod.MedianPolish"/> produces residuals, and only for groups
    /// that actually reach the polish - see
    /// <see cref="ProteinMatrixRollup.Aggregate(double[,], ProteinRollupMethod, int, int, int, string, out double[,])"/>.
    /// The file is one row per (protein group x peptide), one column per sample, on the LOG2 scale.
    /// </para>
    /// </summary>
    public string? ResidualsPath { get; init; }
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

    /// <summary>
    /// The metadata columns of the protein matrices (<c>proteins_raw</c> and, carried through Stage
    /// 4b/4c, <c>corrected_proteins</c>), in output order.
    /// <para>
    /// This is the single source of truth on purpose. Every reader of those files identifies
    /// replicates as "every column that is not one of these", so a name missing from the list is
    /// silently treated as a replicate and its text parsed as an abundance - which is exactly how
    /// the Dynamic Range tab came to throw on <c>leading_description</c> and render blank. Add a
    /// column to the writer below and it appears here automatically; anything else that enumerates
    /// these columns must reference this array rather than repeat it.
    /// </para>
    /// </summary>
    public static readonly string[] MetadataColumns =
    {
        "protein_group", "leading_protein", "leading_name", "leading_uniprot_id", "leading_gene_name",
        "leading_description", "n_peptides", "n_unique_peptides", "low_confidence",
    };

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
        // Column-at-a-time rather than ParquetTable.Load: the whole-table load holds every sample
        // column as a nullable double?[], 16 bytes per cell against the 8 these need - 11 GB versus
        // 6 GB on a 100-document cohort, for the same values. Reading straight into double[] also
        // spares the per-cell null check in the inner loop below.
        using var reader = ParquetColumnReader.Open(peptideLog2Parquet);

        var meta = new HashSet<string>(MetaCols, StringComparer.Ordinal) { peptideCol };
        var sampleCols = samples?.ToList()
            ?? reader.ColumnNames.Where(c => !meta.Contains(c)).ToList();

        // Peptide matrix: peptide -> row of sample values.
        var pepKeys = reader.ReadStrings(peptideCol);
        var pepIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < pepKeys.Length; i++)
            pepIndex[pepKeys[i]] = i;

        var sampleData = sampleCols.Select(reader.ReadDoubles).ToList();

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
                    sub[a, j] = sampleData[j][ri];
            }

            var nTheo = -1;
            if (theoreticalCounts is not null && theoreticalCounts.TryGetValue(group.LeadingProtein, out var c))
                nTheo = c;
            var vals = ProteinMatrixRollup.Aggregate(
                sub, cfg.Method, cfg.MinPeptides, cfg.TopN, nTheo, cfg.TopNSelection, out var resid);

            results[gi] = new ProteinRow
            {
                Group = group,
                NPeptides = available.Count,
                NUniquePeptides = group.UniquePeptides.Count,
                LowConfidence = available.Count < cfg.MinPeptides,
                Values = vals,
                // Held per group and written after the parallel loop: residuals are [nPeptides x
                // nSamples], so the peptide names are needed alongside them to key the output rows.
                ResidualPeptides = resid is null ? null : available,
                Residuals = cfg.ResidualsPath is null ? null : resid,
            };
        });

        var rows = results.Where(r => r is not null).Select(r => r!).ToList();
        WriteOutput(outputPath, sampleCols, rows);
        // Only write when something was actually decomposed. A method that does not polish (sum,
        // topN, maxLFQ, iBAQ) would otherwise leave a zero-row file behind, which reads as "no
        // peptide deviated" rather than "residuals were never computed" - the more misleading of
        // the two. Python guards this the same way (`if save_residuals and residual_rows`).
        if (cfg.ResidualsPath is not null && rows.Any(r => r.Residuals is not null))
            WriteResiduals(cfg.ResidualsPath, peptideCol, sampleCols, rows);
        return new Result(rows.Count, nSkipped, sampleCols);
    }

    /// <summary>
    /// Write the per-peptide polish residuals: one row per (protein group x peptide), one column
    /// per sample, LOG2. Groups that never reached a polish contribute nothing, so the file is
    /// shorter than the peptide count whenever some groups fall below <c>min_peptides</c>.
    /// </summary>
    private static void WriteResiduals(
        string path, string peptideCol, IReadOnlyList<string> samples, List<ProteinRow> rows)
    {
        var meta = new (string, Type)[] { ("protein_group", typeof(string)), (peptideCol, typeof(string)) };
        using var writer = StreamingWideWriter.Create(path, meta, samples);

        var n = rows.Sum(r => r.Residuals is null ? 0 : r.ResidualPeptides!.Count);
        var groupIds = new string[n];
        var peptides = new string[n];
        var sampleColumns = new double[samples.Count][];
        for (var j = 0; j < samples.Count; j++)
            sampleColumns[j] = new double[n];

        var at = 0;
        foreach (var row in rows)
        {
            if (row.Residuals is null)
                continue;
            for (var p = 0; p < row.ResidualPeptides!.Count; p++, at++)
            {
                groupIds[at] = row.Group.GroupId;
                peptides[at] = row.ResidualPeptides[p];
                for (var j = 0; j < samples.Count; j++)
                    sampleColumns[j][at] = row.Residuals[p, j];
            }
        }

        writer.WriteRowGroup(new Array[] { groupIds, peptides }, sampleColumns);
    }

    private sealed class ProteinRow
    {
        public required ProteinGroup Group { get; init; }
        public required int NPeptides { get; init; }
        public required int NUniquePeptides { get; init; }
        public required bool LowConfidence { get; init; }
        public required double[] Values { get; init; }

        /// <summary>Peptide names for <see cref="Residuals"/>' rows; null when no polish ran.</summary>
        public List<string>? ResidualPeptides { get; init; }

        /// <summary>[nPeptides, nSamples] polish residuals, or null when not captured.</summary>
        public double[,]? Residuals { get; init; }
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
            ParquetWideWriter.Strings(MetadataColumns[0], groupId),
            ParquetWideWriter.Strings(MetadataColumns[1], leadingProtein),
            ParquetWideWriter.Strings(MetadataColumns[2], leadingName),
            ParquetWideWriter.Strings(MetadataColumns[3], leadingUniprot),
            ParquetWideWriter.Strings(MetadataColumns[4], leadingGene),
            ParquetWideWriter.Strings(MetadataColumns[5], leadingDesc),
            ParquetWideWriter.Longs(MetadataColumns[6], nPeptides),
            ParquetWideWriter.Longs(MetadataColumns[7], nUnique),
            ParquetWideWriter.Bools(MetadataColumns[8], lowConf),
        };
        ParquetWideWriter.Write(outputPath, metaColumns, samples, sampleColumns, n);
    }
}
