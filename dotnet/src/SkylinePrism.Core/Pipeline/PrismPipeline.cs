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
using SkylinePrism.Core.Qc;
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

    public static Result Run(
        IReadOnlyList<string> inputs, string outputDir, PrismConfig config,
        string? metadataPath = null, Action<string>? log = null)
    {
        Directory.CreateDirectory(outputDir);
        var report = log ?? (_ => { });

        report("============================================================");
        report("Stage 1: Merge / prepare input");
        report("============================================================");
        var mergedPath = Path.Combine(outputDir, "merged_data.parquet");
        var merge = DuckDbMerge.MergeAndSort(inputs, mergedPath);
        report($"  Merged {inputs.Count} report(s) -> {merge.TotalRows:N0} transition rows.");

        var cols = SkylineColumns.Detect(ParquetTable.Load(mergedPath).ColumnNames.ToHashSet());
        var samples = MergedParquetReader.GetSortedSamples(mergedPath, cols.Sample);
        report($"  Columns: peptide='{cols.Peptide}', sample='{cols.Sample}', abundance='{cols.Abundance}'.");
        report($"  Samples: {samples.Count}.");

        // Resolve per-sample batch and type: prefer the Replicates metadata (Batch annotation /
        // Skyline Sample Type), else fall back to the Source Document batch + name patterns.
        var sourceBatchMap = GetBatchMap(mergedPath, cols);
        var metadata = ReplicateMetadata.TryLoad(
            metadataPath, report, config.Metadata.SampleTypeColumn, config.Metadata.BatchColumn);
        var resolvedBatch = new Dictionary<string, string>(StringComparer.Ordinal);
        var resolvedType = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var s in samples)
        {
            var rep = SampleIdToReplicate(s);
            resolvedBatch[s] = metadata?.BatchByReplicate.GetValueOrDefault(rep)
                ?? sourceBatchMap.GetValueOrDefault(s, "batch1");
            resolvedType[s] = metadata?.TypeByReplicate.GetValueOrDefault(rep)
                ?? ClassifySampleType(s, rep, config);
        }

        var batchLabels = samples.Select(s => resolvedBatch[s]).ToList();
        var batches = batchLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
        var multiBatch = batches.Count >= 2;
        var peptideCombat = config.BatchCorrection.Enabled && config.BatchCorrection.PeptideLevel && multiBatch;
        var proteinCombat = config.BatchCorrection.Enabled && config.BatchCorrection.ProteinLevel && multiBatch;
        var combatNote = !multiBatch ? "skipped (needs >= 2 batches)"
            : !config.BatchCorrection.Enabled ? "disabled"
            : $"peptide={(peptideCombat ? "on" : "off")}, protein={(proteinCombat ? "on" : "off")}";
        report($"Batches: {batches.Count} ({string.Join(", ", batches)}); ComBat {combatNote}.");

        // Stage 2: transition -> peptide.
        report("============================================================");
        report($"Stage 2: Transition -> peptide rollup ({config.TransitionRollup.Method})");
        report("============================================================");
        var peptidesRollupPath = Path.Combine(outputDir, "peptides_rollup.parquet");
        var transitionCfg = new TransitionRollupConfig
        {
            Method = config.TransitionRollup.Method is "median_polish"
                ? TransitionRollupMethod.MedianPolish : TransitionRollupMethod.Sum,
            MinTransitions = config.TransitionRollup.MinTransitions,
            UseMs1 = config.TransitionRollup.UseMs1,
        };
        var t2 = TransitionRollup.Run(mergedPath, cols, transitionCfg, peptidesRollupPath, samples);
        report($"  Rolled up to {t2.NPeptides:N0} peptides ({t2.NFiltered:N0} filtered below min_transitions).");

        // Stage 2b/2c: peptide normalization + ComBat -> peptides_log2_internal (LOG2) +
        // corrected_peptides (LINEAR).
        report($"Stage 2b: Peptide normalization ({config.GlobalNormalization.Method})"
            + (peptideCombat ? " + 2c: ComBat batch correction" : "") + "...");
        var internalPath = Path.Combine(outputDir, "peptides_log2_internal.parquet");
        var correctedPepPath = Path.Combine(outputDir, "corrected_peptides." + config.Output.Format);
        var nPeptides = NormalizeAndCorrect(
            peptidesRollupPath,
            new[] { (cols.Peptide, MetaType.Str), ("n_transitions", MetaType.Long), ("mean_rt", MetaType.Double) },
            samples, batchLabels, peptideCombat, config.GlobalNormalization.Method,
            internalPath, correctedPepPath, rtColumn: "mean_rt");
        report($"  Wrote {nPeptides:N0} corrected peptides.");

        // Stage 3: parsimony.
        report("============================================================");
        report(config.Parsimony.Enabled ? "Stage 3: Protein parsimony" : "Stage 3: Protein grouping (parsimony disabled)");
        report("============================================================");
        var groups = ParsimonyEngine.Run(mergedPath, cols, config.Parsimony.Enabled);
        ProteinGroupsCsv.Write(groups, Path.Combine(outputDir, "protein_groups.csv"));
        report($"  {(config.Parsimony.Enabled ? "Computed" : "Built")} {groups.Count:N0} protein groups.");

        // Stage 4: peptide -> protein.
        report("============================================================");
        report($"Stage 4: Peptide -> protein rollup ({config.ProteinRollup.Method})");
        report("============================================================");
        var proteinsRawPath = Path.Combine(outputDir, "proteins_raw.parquet");
        var proteinCfg = new ProteinRollupConfig
        {
            Method = config.ProteinRollup.Method is "sum"
                ? ProteinRollupMethod.Sum : ProteinRollupMethod.MedianPolish,
            MinPeptides = config.ProteinRollup.MinPeptides,
        };
        var protResult = ProteinRollup.Run(internalPath, groups, proteinCfg, cols.Peptide, proteinsRawPath, samples);
        report($"  Rolled up to {protResult.NProteins:N0} proteins.");
        report($"Stage 4b: Protein normalization ({config.ProteinNormalization.Method})"
            + (proteinCombat ? " + 4c: ComBat" : "") + "...");

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
            proteinsRawPath, proteinMeta, samples, batchLabels, proteinCombat,
            config.ProteinNormalization.Method, internalLog2Path: null, correctedLinearPath: correctedProtPath);

        report("============================================================");
        report("Stage 5: Output generation");
        report("============================================================");
        WriteSampleMetadata(Path.Combine(outputDir, "sample_metadata.csv"), samples, resolvedBatch, resolvedType);
        var nRef = resolvedType.Values.Count(t => t == "reference");
        var nQc = resolvedType.Values.Count(t => t == "qc");
        report($"  Sample types: {nRef} reference, {nQc} qc, {resolvedType.Count - nRef - nQc} experimental.");
        report($"  Wrote corrected_peptides / corrected_proteins ({config.Output.Format}, linear) and sample_metadata.csv.");

        // Stage 5b: QC report.
        if (config.QcReport.Enabled)
        {
            report("Stage 5b: Generating QC report (qc_report.html)...");
            QcReport.Generate(outputDir, config, savePlots: config.QcReport.SavePlots);
        }

        report($"PRISM complete: {nPeptides:N0} peptides, {nProteins:N0} proteins, {samples.Count:N0} samples, "
            + $"{batches.Count} batch(es).");
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
        string correctedLinearPath,
        string? rtColumn = null)
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

        double[,] normalized;
        if (normMethod is "rt_lowess" && rtColumn is not null && table.HasColumn(rtColumn))
        {
            var rtAll = table.GetDouble(rtColumn);
            var rtKept = new double[n];
            for (var r = 0; r < n; r++)
                rtKept[r] = rtAll[keep[r]] ?? double.NaN;
            normalized = Normalizer.RtLowessNormalize(matrix, rtKept);
        }
        else if (normMethod is "none")
        {
            normalized = matrix;
        }
        else
        {
            normalized = Normalizer.MedianNormalize(matrix);
        }
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
        string path, IReadOnlyList<string> samples,
        IReadOnlyDictionary<string, string> resolvedBatch, IReadOnlyDictionary<string, string> resolvedType)
    {
        var sb = new StringBuilder("sample_id,sample,sample_type,batch\n");
        foreach (var sampleId in samples)
        {
            var replicate = SampleIdToReplicate(sampleId);
            var batch = resolvedBatch.GetValueOrDefault(sampleId, "batch1");
            var type = resolvedType.GetValueOrDefault(sampleId, "experimental");
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
