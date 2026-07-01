using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using DuckDB.NET.Data;
using SkylinePrism.Core.BatchCorrection;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Core.Parsimony;
using SkylinePrism.Core.Rollup;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// End-to-end PRISM pipeline (cmd_run), orchestrating Stage 1..5: merge -> transition rollup
/// -> peptide normalization -> peptide ComBat -> parsimony -> protein rollup -> protein
/// normalization -> protein ComBat -> outputs. Produces the same output files as the Python
/// pipeline. QC report generation (Stage 5b) is Layer 8.
/// </summary>
public sealed class PrismPipeline
{
    public sealed record Result(
        int NPeptides, int NProteins, int NSamples, IReadOnlyList<string> Batches);

    public static Result Run(IReadOnlyList<string> inputs, string outputDir, PrismConfig config)
    {
        Directory.CreateDirectory(outputDir);

        // Stage 1: merge.
        var mergedPath = Path.Combine(outputDir, "merged_data.parquet");
        DuckDbMerge.MergeAndSort(inputs, mergedPath);

        var cols = SkylineColumns.Detect(ParquetTable.Load(mergedPath).ColumnNames.ToHashSet());
        var samples = MergedParquetReader.GetSortedSamples(mergedPath, cols.Sample);
        var batchMap = GetBatchMap(mergedPath, cols);
        var batchLabels = samples.Select(s => batchMap.GetValueOrDefault(s, "batch1")).ToList();
        var batches = batchLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
        var combatEnabled = config.BatchCorrection.Enabled && batches.Count >= 2;

        // Stage 2: transition -> peptide.
        var peptidesRollupPath = Path.Combine(outputDir, "peptides_rollup.parquet");
        var transitionCfg = new TransitionRollupConfig
        {
            Method = config.TransitionRollup.Method is "median_polish"
                ? TransitionRollupMethod.MedianPolish : TransitionRollupMethod.Sum,
            MinTransitions = config.TransitionRollup.MinTransitions,
            UseMs1 = config.TransitionRollup.UseMs1,
        };
        TransitionRollup.Run(mergedPath, cols, transitionCfg, peptidesRollupPath, samples);

        // Stage 2b/2c: peptide normalization + ComBat -> peptides_log2_internal (LOG2) +
        // corrected_peptides (LINEAR).
        var internalPath = Path.Combine(outputDir, "peptides_log2_internal.parquet");
        var correctedPepPath = Path.Combine(outputDir, "corrected_peptides." + config.Output.Format);
        var nPeptides = NormalizeAndCorrect(
            peptidesRollupPath,
            new[] { (cols.Peptide, MetaType.Str), ("n_transitions", MetaType.Long), ("mean_rt", MetaType.Double) },
            samples, batchLabels, combatEnabled, config.GlobalNormalization.Method,
            internalPath, correctedPepPath);

        // Stage 3: parsimony.
        var groups = ParsimonyEngine.Run(mergedPath, cols);
        ProteinGroupsCsv.Write(groups, Path.Combine(outputDir, "protein_groups.csv"));

        // Stage 4: peptide -> protein.
        var proteinsRawPath = Path.Combine(outputDir, "proteins_raw.parquet");
        var proteinCfg = new ProteinRollupConfig
        {
            Method = config.ProteinRollup.Method is "sum"
                ? ProteinRollupMethod.Sum : ProteinRollupMethod.MedianPolish,
            MinPeptides = config.ProteinRollup.MinPeptides,
        };
        var protResult = ProteinRollup.Run(internalPath, groups, proteinCfg, cols.Peptide, proteinsRawPath, samples);

        // Stage 4b/4c: protein normalization + ComBat -> corrected_proteins (LINEAR).
        var correctedProtPath = Path.Combine(outputDir, "corrected_proteins." + config.Output.Format);
        var proteinMeta = new (string, MetaType)[]
        {
            ("protein_group", MetaType.Str), ("leading_protein", MetaType.Str), ("leading_name", MetaType.Str),
            ("leading_uniprot_id", MetaType.Str), ("leading_gene_name", MetaType.Str),
            ("leading_description", MetaType.Str), ("n_peptides", MetaType.Long),
            ("n_unique_peptides", MetaType.Long), ("low_confidence", MetaType.Bool),
        };
        var nProteins = NormalizeAndCorrect(
            proteinsRawPath, proteinMeta, samples, batchLabels, combatEnabled,
            config.ProteinNormalization.Method, internalLog2Path: null, correctedLinearPath: correctedProtPath);

        WriteSampleMetadata(Path.Combine(outputDir, "sample_metadata.csv"), samples, batchMap, config);

        return new Result(nPeptides, nProteins, samples.Count, batches);
    }

    private enum MetaType { Str, Long, Double, Bool }

    /// <summary>
    /// Load a wide LOG2 parquet, drop all-NaN feature rows, median-normalize, optionally
    /// ComBat, then write the LOG2 "internal" parquet (if a path is given) and the LINEAR
    /// corrected output. Returns the number of features written.
    /// </summary>
    private static int NormalizeAndCorrect(
        string wideParquet,
        IReadOnlyList<(string Name, MetaType Type)> metaSpec,
        IReadOnlyList<string> samples,
        IReadOnlyList<string> batchLabels,
        bool combatEnabled,
        string normMethod,
        string? internalLog2Path,
        string correctedLinearPath)
    {
        var table = ParquetTable.Load(wideParquet);
        var nAll = table.RowCount;

        // Read matrix + meta.
        var matrixAll = new double[nAll, samples.Count];
        for (var j = 0; j < samples.Count; j++)
        {
            var col = table.GetDouble(samples[j]);
            for (var i = 0; i < nAll; i++)
                matrixAll[i, j] = col[i] ?? double.NaN;
        }

        // Drop all-NaN rows.
        var keep = new List<int>(nAll);
        for (var i = 0; i < nAll; i++)
        {
            var any = false;
            for (var j = 0; j < samples.Count && !any; j++)
                any = !double.IsNaN(matrixAll[i, j]);
            if (any)
                keep.Add(i);
        }

        var n = keep.Count;
        var matrix = new double[n, samples.Count];
        for (var r = 0; r < n; r++)
            for (var j = 0; j < samples.Count; j++)
                matrix[r, j] = matrixAll[keep[r], j];

        var normalized = normMethod is "none"
            ? matrix
            : Normalizer.MedianNormalize(matrix);
        var corrected = combatEnabled ? ComBat.Run(normalized, batchLabels) : normalized;

        // Meta columns (filtered to kept rows).
        var metaCols = new List<ParquetWideWriter.MetaColumn>();
        foreach (var (name, type) in metaSpec)
        {
            switch (type)
            {
                case MetaType.Str:
                    var sv = table.GetString(name);
                    metaCols.Add(ParquetWideWriter.Strings(name, keep.Select(i => sv[i] ?? "").ToArray()));
                    break;
                case MetaType.Long:
                    var lv = table.GetLong(name);
                    metaCols.Add(ParquetWideWriter.Longs(name, keep.Select(i => lv[i]).ToArray()));
                    break;
                case MetaType.Double:
                    var dv = table.GetDouble(name);
                    metaCols.Add(ParquetWideWriter.Doubles(name, keep.Select(i => dv[i] ?? double.NaN).ToArray()));
                    break;
                case MetaType.Bool:
                    var bv = table.GetBool(name);
                    metaCols.Add(ParquetWideWriter.Bools(name, keep.Select(i => bv[i]).ToArray()));
                    break;
            }
        }

        var sampleCols = new double[samples.Count][];
        for (var j = 0; j < samples.Count; j++)
        {
            sampleCols[j] = new double[n];
            for (var r = 0; r < n; r++)
                sampleCols[j][r] = corrected[r, j];
        }

        if (internalLog2Path is not null)
            ParquetWideWriter.Write(internalLog2Path, metaCols, samples, sampleCols, n);

        // Corrected output is LINEAR (2^log2).
        var linearCols = new double[samples.Count][];
        for (var j = 0; j < samples.Count; j++)
        {
            linearCols[j] = new double[n];
            for (var r = 0; r < n; r++)
                linearCols[j][r] = Math.Pow(2.0, corrected[r, j]);
        }

        if (correctedLinearPath.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase))
            ParquetWideWriter.Write(correctedLinearPath, metaCols, samples, linearCols, n);
        else
            WriteDelimited(correctedLinearPath, metaCols, samples, linearCols, n);

        return n;
    }

    private static Dictionary<string, string> GetBatchMap(string mergedPath, SkylineColumns cols)
    {
        var batchCol = cols.Batch ?? "Batch";
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT DISTINCT \"{cols.Sample}\" AS s, \"{batchCol}\" AS b FROM read_parquet('{mergedPath.Replace("'", "''")}')";
        using var reader = cmd.ExecuteReader();
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        while (reader.Read())
        {
            if (reader.IsDBNull(0))
                continue;
            map[reader.GetString(0)] = reader.IsDBNull(1)
                ? "batch1"
                : Convert.ToString(reader.GetValue(1), CultureInfo.InvariantCulture) ?? "batch1";
        }
        return map;
    }

    private static void WriteSampleMetadata(
        string path, IReadOnlyList<string> samples, IReadOnlyDictionary<string, string> batchMap, PrismConfig config)
    {
        var sb = new StringBuilder("sample_id,sample,sample_type,batch\n");
        foreach (var sampleId in samples)
        {
            var replicate = SampleIdToReplicate(sampleId);
            var batch = batchMap.GetValueOrDefault(sampleId, "batch1");
            var type = ClassifySampleType(sampleId, replicate, config);
            sb.Append(Csv(sampleId)).Append(',').Append(Csv(replicate)).Append(',')
              .Append(type).Append(',').Append(Csv(batch)).Append('\n');
        }
        File.WriteAllText(path, sb.ToString());
    }

    private static string ClassifySampleType(string sampleId, string replicate, PrismConfig config)
    {
        bool Matches(IEnumerable<string> patterns) =>
            patterns.Any(p => sampleId.Contains(p, StringComparison.Ordinal)
                              || replicate.Contains(p, StringComparison.Ordinal));
        if (Matches(config.SampleAnnotations.ReferencePattern))
            return "reference";
        if (Matches(config.SampleAnnotations.QcPattern))
            return "qc";
        return "experimental";
    }

    private static string SampleIdToReplicate(string sampleId)
    {
        const string sep = "__@__";
        var idx = sampleId.IndexOf(sep, StringComparison.Ordinal);
        return idx >= 0 ? sampleId[..idx] : sampleId;
    }

    private static void WriteDelimited(
        string path, IReadOnlyList<ParquetWideWriter.MetaColumn> meta,
        IReadOnlyList<string> samples, IReadOnlyList<double[]> sampleCols, int n)
    {
        var delim = path.EndsWith(".tsv", StringComparison.OrdinalIgnoreCase) ? '\t' : ',';
        var sb = new StringBuilder();
        var headers = meta.Select(m => m.Name).Concat(samples);
        sb.Append(string.Join(delim, headers)).Append('\n');
        for (var r = 0; r < n; r++)
        {
            var fields = new List<string>();
            foreach (var m in meta)
                fields.Add(Convert.ToString(m.Values.GetValue(r), CultureInfo.InvariantCulture) ?? "");
            for (var j = 0; j < samples.Count; j++)
                fields.Add(sampleCols[j][r].ToString("R", CultureInfo.InvariantCulture));
            sb.Append(string.Join(delim, fields)).Append('\n');
        }
        File.WriteAllText(path, sb.ToString());
    }

    private static string Csv(string s) =>
        s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
}
