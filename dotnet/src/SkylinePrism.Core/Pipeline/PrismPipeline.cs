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
        IReadOnlyList<string>? metadataPaths = null, Action<string>? log = null,
        bool forceReprocess = false)
    {
        Directory.CreateDirectory(outputDir);
        var report = log ?? (_ => { });

        report("============================================================");
        report("Stage 1: Merge / prepare input");
        report("============================================================");
        var mergedPath = Path.Combine(outputDir, "merged_data.parquet");
        var cachePath = mergedPath + ".cache.json";
        var fingerprint = SourceFingerprint.Compute(inputs);

        DuckDbMerge.MergeResult merge;
        var cached = forceReprocess ? null : SourceFingerprint.TryRead(cachePath);
        if (cached is not null && cached.Fingerprint == fingerprint && File.Exists(mergedPath))
        {
            merge = new DuckDbMerge.MergeResult(mergedPath, cached.SortColumn, cached.TotalRows);
            report($"  Reusing cached merge ({merge.TotalRows:N0} rows; inputs unchanged - "
                + "pass --force-reprocess to rebuild).");
        }
        else
        {
            merge = DuckDbMerge.MergeAndSort(inputs, mergedPath, replicateColumn: config.Data.SampleColumn);
            SourceFingerprint.Write(cachePath,
                new SourceFingerprint.CacheEntry(fingerprint, merge.TotalRows, merge.SortColumn));
            report($"  Merged {inputs.Count} report(s) -> {merge.TotalRows:N0} transition rows.");
        }

        // Schema-only read: never materialize the (potentially huge, 200-report) merged table
        // just to detect column names.
        var cols = SkylineColumns.Detect(
            ParquetTable.ReadColumnNames(mergedPath).ToHashSet(), config.Data.ToOverrides());
        var samples = MergedParquetReader.GetSortedSamples(mergedPath, cols.Sample);
        report($"  Columns: peptide='{cols.Peptide}', sample='{cols.Sample}', abundance='{cols.Abundance}'.");
        report($"  Samples: {samples.Count}.");

        // Resolve per-sample batch and type: prefer the Replicates metadata (Batch annotation /
        // Skyline Sample Type), else fall back to the Source Document batch + name patterns.
        var sourceBatchMap = GetBatchMap(mergedPath, cols);
        // Metadata files are matched to inputs positionally so each file's rows can be qualified by its
        // source document ("<replicate>__@__<document>"). Without that, a replicate name reused across
        // documents (every plate has a "Ref_01") would take the LAST file's type/batch for every document.
        // Only 1:1 metadata-per-input is unambiguous; otherwise fall back to bare replicate keys.
        var documentLabels = metadataPaths is not null && metadataPaths.Count == inputs.Count && inputs.Count > 1
            ? inputs.Select(p => Path.GetFileNameWithoutExtension(p) ?? p).ToList()
            : null;
        var metadata = ReplicateMetadata.TryLoad(
            metadataPaths, report,
            config.Data.SampleTypeColumn ?? config.Metadata.SampleTypeColumn,
            config.Data.BatchColumn ?? config.Metadata.BatchColumn,
            documentLabels);
        var resolvedBatch = new Dictionary<string, string>(StringComparer.Ordinal);
        var resolvedType = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var s in samples)
        {
            var rep = SampleIdToReplicate(s);
            resolvedBatch[s] = metadata?.BatchFor(s, rep)
                ?? sourceBatchMap.GetValueOrDefault(s, "batch1");
            resolvedType[s] = metadata?.TypeFor(s, rep)
                ?? ClassifySampleType(s, rep, config);
        }

        // If neither metadata nor Source Document distinguishes batches (all one label), fall back
        // to estimating batches from acquisition-time gaps (ports cli.py:estimate_batches_from_parquet).
        var estMethod = (config.BatchEstimation.Method ?? "auto").ToLowerInvariant();
        if (resolvedBatch.Values.Distinct().Count() <= 1
            && estMethod is not ("none" or "source")
            && cols.AcquiredTime is not null)
        {
            var est = BatchEstimator.Estimate(
                mergedPath, cols.Sample, cols.AcquiredTime, estMethod,
                config.BatchEstimation.NBatches, config.BatchEstimation.GapIqrMultiplier);
            var nEst = est.Values.Distinct().Count();
            if (nEst > 1)
            {
                foreach (var s in samples)
                {
                    var rep = SampleIdToReplicate(s);
                    var hasMeta = metadata?.HasBatchFor(s, rep) == true;
                    if (!hasMeta && est.TryGetValue(s, out var b))
                        resolvedBatch[s] = b;
                }
                report($"  Batch estimation: {nEst} batches from acquisition-time gaps "
                    + $"(method={estMethod}).");
            }
        }

        var batchLabels = samples.Select(s => resolvedBatch[s]).ToList();
        var batches = batchLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
        var multiBatch = batches.Count >= 2;
        var peptideCombat = config.BatchCorrection.Enabled && config.BatchCorrection.PeptideLevel && multiBatch;
        var proteinCombat = config.BatchCorrection.Enabled && config.BatchCorrection.ProteinLevel && multiBatch;
        var combatNote = !multiBatch ? "skipped (needs >= 2 batches)"
            : !config.BatchCorrection.Enabled ? "disabled"
            : $"peptide={(peptideCombat ? "on" : "off")}, protein={(proteinCombat ? "on" : "off")}";
        // Which source supplied the batch labels (metadata Batch column > per-file Source Document >
        // acquisition-time estimation > a single default label).
        var metaProvidedBatch = metadata is not null
            && samples.Any(s => metadata.HasBatchFor(s, SampleIdToReplicate(s)));
        var batchSource = metaProvidedBatch ? "metadata report (Batch column)"
            : sourceBatchMap.Values.Distinct().Count() > 1 ? "Source Document (per-file)"
            : batches.Count > 1 ? "acquisition-time estimation"
            : "single label (no batch annotation)";
        report($"Batches: {batches.Count} from {batchSource} ({string.Join(", ", batches)}); ComBat {combatNote}.");

        // Stage 2: transition -> peptide.
        report("============================================================");
        report($"Stage 2: Transition -> peptide rollup ({config.TransitionRollup.Method})");
        report("============================================================");
        var peptidesRollupPath = Path.Combine(outputDir, "peptides_rollup.parquet");
        var transitionCfg = new TransitionRollupConfig
        {
            Method = config.TransitionRollup.Method switch
            {
                "median_polish" => TransitionRollupMethod.MedianPolish,
                "topn" => TransitionRollupMethod.TopN,
                "consensus" => TransitionRollupMethod.Consensus,
                "library_assist" or "library-assisted" or "library_assisted" => TransitionRollupMethod.LibraryAssist,
                _ => TransitionRollupMethod.Sum,
            },
            MinTransitions = config.TransitionRollup.MinTransitions,
            TopNCount = config.TransitionRollup.TopnCount,
            TopNSelection = config.TransitionRollup.TopnSelection,
            TopNWeighting = config.TransitionRollup.TopnWeighting,
            ConsensusRegularization = config.TransitionRollup.ConsensusRegularization,
            UseMs1 = config.TransitionRollup.UseMs1,
            LibraryPath = config.TransitionRollup.LibraryPath,
            LibraryMinFragments = config.TransitionRollup.LibraryMinFragments,
            LibraryMzTolerance = config.TransitionRollup.LibraryMzTolerance,
            LibraryOutlierThreshold = config.TransitionRollup.LibraryOutlierThreshold,
            LibraryRemoveOutliers = config.TransitionRollup.LibraryRemoveOutliers,
            ResidualsPath = config.Output.IncludeResiduals
                ? Path.Combine(outputDir, "peptide_residuals.parquet")
                : null,
            MaxDegreeOfParallelism = config.Processing.NWorkers,
            FlushRows = config.Processing.PeptideBatchSize,
        };
        if (transitionCfg.Method == TransitionRollupMethod.LibraryAssist)
            report($"  Library-assisted rollup using spectral library: {config.TransitionRollup.LibraryPath}");
        var dop = config.Processing.NWorkers <= 0
            ? Environment.ProcessorCount
            : Math.Min(config.Processing.NWorkers, Environment.ProcessorCount);
        report($"  Rollup workers: {dop} thread(s) (streamed to parquet in row-group batches of "
            + $"{Math.Max(1, config.Processing.PeptideBatchSize):N0}).");
        var t2 = TransitionRollup.Run(mergedPath, cols, transitionCfg, peptidesRollupPath, samples);
        report($"  Rolled up to {t2.NPeptides:N0} peptides ({t2.NFiltered:N0} filtered below min_transitions).");
        if (transitionCfg.ResidualsPath is not null && transitionCfg.Method == TransitionRollupMethod.MedianPolish)
            report("  Wrote peptide_residuals.parquet (per-transition median-polish residuals).");

        // Stage 2a: peptide-matrix density diagnostic + optional sample outlier detection.
        {
            var pepTable = ParquetTable.Load(peptidesRollupPath);
            var m = new double[pepTable.RowCount, samples.Count];
            long nanCells = 0;
            for (var j = 0; j < samples.Count; j++)
            {
                var col = pepTable.GetDouble(samples[j]);
                for (var i = 0; i < pepTable.RowCount; i++)
                {
                    m[i, j] = col[i] ?? double.NaN;
                    if (double.IsNaN(m[i, j]))
                        nanCells++;
                }
            }
            // Should be ~0: Skyline integrates every transition at (imputed) peak boundaries, so
            // its export is already complete; PRISM only floors the rare 0 / #N/A to a small value
            // before rollup. A non-trivial count therefore flags a real data issue (a report
            // missing transitions, a bad column mapping, etc.), not normal missingness.
            var totalCells = (long)pepTable.RowCount * samples.Count;
            report($"  Peptide matrix: {nanCells:N0} missing of {totalCells:N0} cells "
                + $"({(totalCells > 0 ? 100.0 * nanCells / totalCells : 0):0.###}%) "
                + (nanCells == 0 ? "- fully dense, as expected." : "- unexpected; investigate."));

            if (config.SampleOutlierDetection.Enabled)
            {
            var odMethod = config.SampleOutlierDetection.Method == "fold_median"
                ? OutlierDetector.Method.FoldMedian : OutlierDetector.Method.Iqr;
            var od = OutlierDetector.Detect(m, samples, odMethod,
                config.SampleOutlierDetection.IqrMultiplier, config.SampleOutlierDetection.FoldThreshold);

            if (od.Outliers.Count == 0)
            {
                report("  Sample outlier detection: no low-signal outliers.");
            }
            else if (config.SampleOutlierDetection.Action == "exclude")
            {
                report($"  Sample outlier detection: EXCLUDING {od.Outliers.Count} low-signal sample(s): "
                    + string.Join(", ", od.Outliers));
                var excluded = new HashSet<string>(od.Outliers, StringComparer.Ordinal);
                var keptSamples = new List<string>(samples.Count);
                var keptBatch = new List<string>(samples.Count);
                for (var j = 0; j < samples.Count; j++)
                {
                    if (excluded.Contains(samples[j]))
                        continue;
                    keptSamples.Add(samples[j]);
                    keptBatch.Add(batchLabels[j]);
                }
                samples = keptSamples;
                batchLabels = keptBatch;
                batches = batchLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
                var mb = batches.Count >= 2;
                peptideCombat = config.BatchCorrection.Enabled && config.BatchCorrection.PeptideLevel && mb;
                proteinCombat = config.BatchCorrection.Enabled && config.BatchCorrection.ProteinLevel && mb;
            }
            else
            {
                report($"  Sample outlier detection: {od.Outliers.Count} low-signal sample(s) flagged "
                    + "(report only, kept): " + string.Join(", ", od.Outliers));
            }
            }
        }

        // Stage 2b/2c: peptide normalization + ComBat -> peptides_log2_internal (LOG2) +
        // corrected_peptides (LINEAR).
        // Reference-sample mask (for reference-anchored ComBat), aligned to the current samples.
        var refType = config.BatchCorrection.ReferenceType ?? "reference";
        var referenceAnchored = config.BatchCorrection.ReferenceAnchored;
        var refMask = samples
            .Select(s => string.Equals(resolvedType[s], refType, StringComparison.OrdinalIgnoreCase))
            .ToList();

        // Control-sample column indices for the before/after median-CV report (computed on linear scale).
        var refIdx = Enumerable.Range(0, samples.Count)
            .Where(j => string.Equals(resolvedType[samples[j]], "reference", StringComparison.OrdinalIgnoreCase))
            .ToList();
        var qcIdx = Enumerable.Range(0, samples.Count)
            .Where(j => string.Equals(resolvedType[samples[j]], "qc", StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (referenceAnchored && (peptideCombat || proteinCombat))
        {
            var nRefSamples = refMask.Count(x => x);
            report(nRefSamples > 0
                ? $"  ComBat: reference-anchored using {nRefSamples} '{refType}' sample(s)."
                : $"  ComBat: reference-anchored requested but no '{refType}' samples found; using standard ComBat.");
        }

        report($"Stage 2b: Peptide normalization ({config.GlobalNormalization.Method})"
            + (peptideCombat ? " + 2c: ComBat batch correction" : "") + "...");
        var internalPath = Path.Combine(outputDir, "peptides_log2_internal.parquet");
        var correctedPepPath = Path.Combine(outputDir, "corrected_peptides." + config.Output.Format);
        var nPeptides = NormalizeAndCorrect(
            peptidesRollupPath,
            new[] { (cols.Peptide, MetaType.Str), ("n_transitions", MetaType.Long), ("mean_rt", MetaType.Double) },
            samples, batchLabels, peptideCombat, config.GlobalNormalization.Method,
            internalPath, correctedPepPath,
            report, refIdx, qcIdx,
            referenceAnchored: referenceAnchored, referenceMask: refMask, rtColumn: "mean_rt",
            rtLowessFrac: config.GlobalNormalization.RtLowess.Frac,
            rtLowessGridPoints: config.GlobalNormalization.RtLowess.NGridPoints,
            autoRevert: config.BatchCorrection.AutoRevert);
        report($"  Wrote {nPeptides:N0} corrected peptides.");

        // Stage 3: parsimony.
        report("============================================================");
        report(config.Parsimony.Enabled ? "Stage 3: Protein parsimony" : "Stage 3: Protein grouping (parsimony disabled)");
        report("============================================================");
        if (!string.IsNullOrWhiteSpace(config.Parsimony.FastaPath))
            report($"  FASTA-based parsimony map: {config.Parsimony.FastaPath} "
                + $"(enzyme={config.Parsimony.Enzyme}, specificity={config.Parsimony.EnzymeSpecificity})");
        var groups = ParsimonyEngine.Run(
            mergedPath, cols, config.Parsimony.Enabled, config.Parsimony.FastaPath,
            config.Parsimony.Enzyme, config.Parsimony.EnzymeSpecificity);
        ProteinGroupsCsv.Write(groups, Path.Combine(outputDir, "protein_groups.csv"));
        report($"  {(config.Parsimony.Enabled ? "Computed" : "Built")} {groups.Count:N0} protein groups.");

        // Stage 4: peptide -> protein.
        report("============================================================");
        report($"Stage 4: Peptide -> protein rollup ({config.ProteinRollup.Method})");
        report("============================================================");
        var proteinsRawPath = Path.Combine(outputDir, "proteins_raw.parquet");
        var proteinCfg = new ProteinRollupConfig
        {
            Method = config.ProteinRollup.Method switch
            {
                "sum" => ProteinRollupMethod.Sum,
                "topn" => ProteinRollupMethod.TopN,
                "maxlfq" => ProteinRollupMethod.MaxLfq,
                "ibaq" => ProteinRollupMethod.Ibaq,
                _ => ProteinRollupMethod.MedianPolish,
            },
            MinPeptides = config.ProteinRollup.MinPeptides,
            TopN = config.ProteinRollup.Topn.N,
            TopNSelection = config.ProteinRollup.Topn.Selection,
            SharedPeptideHandling = config.Parsimony.SharedPeptideHandling,
        };

        // iBAQ needs a theoretical peptide count per leading protein (from an in-silico FASTA digest).
        IReadOnlyDictionary<string, int>? theoreticalCounts = null;
        if (proteinCfg.Method == ProteinRollupMethod.Ibaq)
        {
            var ibaqFasta = config.ProteinRollup.Ibaq.FastaPath ?? config.Parsimony.FastaPath;
            if (string.IsNullOrWhiteSpace(ibaqFasta))
            {
                report("  iBAQ: no FASTA (protein_rollup.ibaq.fasta_path / parsimony.fasta_path) - "
                    + "falling back to observed peptide counts.");
            }
            else
            {
                var leading = new HashSet<string>(groups.Select(g => g.LeadingProtein), StringComparer.Ordinal);
                theoreticalCounts = FastaParser.GetTheoreticalPeptideCounts(
                    ibaqFasta, leading, config.ProteinRollup.Ibaq.Enzyme, config.ProteinRollup.Ibaq.MissedCleavages,
                    config.ProteinRollup.Ibaq.MinPeptideLength, config.ProteinRollup.Ibaq.MaxPeptideLength);
                report($"  iBAQ: theoretical peptide counts for {theoreticalCounts.Count:N0} proteins "
                    + $"(enzyme={config.ProteinRollup.Ibaq.Enzyme}, missed_cleavages={config.ProteinRollup.Ibaq.MissedCleavages}, "
                    + $"length {config.ProteinRollup.Ibaq.MinPeptideLength}-{config.ProteinRollup.Ibaq.MaxPeptideLength}).");
            }
        }
        var protResult = ProteinRollup.Run(
            internalPath, groups, proteinCfg, cols.Peptide, proteinsRawPath, samples, theoreticalCounts,
            maxDegreeOfParallelism: config.Processing.NWorkers);
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
            config.ProteinNormalization.Method, internalLog2Path: null, correctedLinearPath: correctedProtPath,
            report: report, refIdx: refIdx, qcIdx: qcIdx,
            referenceAnchored: referenceAnchored, referenceMask: refMask,
            autoRevert: config.BatchCorrection.AutoRevert);

        report("============================================================");
        report("Stage 5: Output generation");
        report("============================================================");
        WriteSampleMetadata(Path.Combine(outputDir, "sample_metadata.csv"), samples, resolvedBatch, resolvedType);
        Provenance.Write(
            Path.Combine(outputDir, "parameters.json"), config, inputs,
            new Provenance.Stats(samples.Count, nPeptides, nProteins, groups.Count),
            DateTime.UtcNow.ToString("o"));
        var nRef = resolvedType.Values.Count(t => t == "reference");
        var nQc = resolvedType.Values.Count(t => t == "qc");
        var nExp = resolvedType.Values.Count(t => t == "experimental");
        var nBlank = resolvedType.Count - nRef - nQc - nExp;
        report($"  Sample types: {nRef} reference, {nQc} qc, {nExp} experimental"
            + (nBlank > 0 ? $", {nBlank} blank/solvent (excluded from groupings)" : "") + ".");
        report($"  Wrote corrected_peptides / corrected_proteins ({config.Output.Format}, linear), "
            + "sample_metadata.csv, and parameters.json (provenance).");

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
        Action<string> report,
        IReadOnlyList<int> refIdx,
        IReadOnlyList<int> qcIdx,
        bool referenceAnchored = false,
        IReadOnlyList<bool>? referenceMask = null,
        string? rtColumn = null,
        double rtLowessFrac = 0.3,
        int rtLowessGridPoints = 100,
        bool autoRevert = false)
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

        // Reuse matrixAll when nothing was dropped (the dense case) instead of copying it.
        var n = keep.Count;
        double[,] matrix;
        if (n == nAll)
        {
            matrix = matrixAll;
        }
        else
        {
            matrix = new double[n, samples.Count];
            for (var r = 0; r < n; r++)
                for (var j = 0; j < samples.Count; j++)
                    matrix[r, j] = matrixAll[keep[r], j];
        }
        matrixAll = null!; // free the [nAll] copy (or clear the alias) - dead from here

        // Control-sample median CV (linear scale) BEFORE normalization/correction (matrix is freed below).
        var beforeRefCv = refIdx.Count >= 2 ? CvMetrics.MedianCv(matrix, refIdx) : double.NaN;
        var beforeQcCv = qcIdx.Count >= 2 ? CvMetrics.MedianCv(matrix, qcIdx) : double.NaN;

        double[]? rtKept = null;
        if (normMethod is "rt_lowess" && rtColumn is not null && table.HasColumn(rtColumn))
        {
            var rtAll = table.GetDouble(rtColumn);
            rtKept = new double[n];
            for (var r = 0; r < n; r++)
                rtKept[r] = rtAll[keep[r]] ?? double.NaN;
        }

        var normalized = rtKept is not null
            ? Normalizer.RtLowessNormalize(matrix, rtKept, rtLowessFrac, rtLowessGridPoints)
            : normMethod switch
            {
                "quantile" => Normalizer.QuantileNormalize(matrix),
                "vsn" => Normalizer.VsnNormalize(matrix),
                "none" => matrix,
                _ => Normalizer.MedianNormalize(matrix),
            };
        if (!ReferenceEquals(normalized, matrix))
            matrix = null!; // dead once a distinct normalized matrix exists

        double[,] corrected;
        if (!combatEnabled)
        {
            corrected = normalized;
        }
        else
        {
            var combatOut = referenceAnchored && referenceMask is not null && referenceMask.Any(m => m)
                ? ReferenceAnchoredComBat.Run(normalized, batchLabels, referenceMask)
                : ComBat.Run(normalized, batchLabels);

            // Safety net (opt-in): if ComBat worsened the control CV by >10%, revert to the uncorrected
            // (post-normalization) data; separately warn on reference/QC overfitting.
            if (autoRevert)
            {
                var eval = BatchCorrectionEvaluator.Evaluate(normalized, combatOut, qcIdx, refIdx);
                if (eval.OverfittingWarning is not null)
                    report($"  WARNING: ComBat {eval.OverfittingWarning}");
                if (eval.Revert)
                {
                    report($"  ComBat REVERTED: {eval.ControlName} CV worsened "
                        + $"{eval.ControlCvBefore:F1}% -> {eval.ControlCvAfter:F1}% (>10%); keeping uncorrected data.");
                    corrected = normalized;
                }
                else
                {
                    corrected = combatOut;
                }
            }
            else
            {
                corrected = combatOut;
            }
        }
        if (!ReferenceEquals(corrected, normalized))
            normalized = null!; // dead after correction

        // Median control-sample CV before vs after normalization + batch correction (linear scale).
        // Only a type with >= 2 samples is meaningful; skip the other (or both if no controls).
        if (refIdx.Count >= 2)
            report($"  Reference CV (median): {beforeRefCv:F1}% -> {CvMetrics.MedianCv(corrected, refIdx):F1}% (before -> after)");
        if (qcIdx.Count >= 2)
            report($"  QC CV (median): {beforeQcCv:F1}% -> {CvMetrics.MedianCv(corrected, qcIdx):F1}% (before -> after)");

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

        // LOG2 "internal" output only when requested (peptide stage). Scoped so the transpose is
        // freed before the linear transpose is allocated (peak = corrected + one column set).
        if (internalLog2Path is not null)
        {
            var log2Cols = new double[samples.Count][];
            for (var j = 0; j < samples.Count; j++)
            {
                log2Cols[j] = new double[n];
                for (var r = 0; r < n; r++)
                    log2Cols[j][r] = corrected[r, j];
            }
            ParquetWideWriter.Write(internalLog2Path, metaCols, samples, log2Cols, n);
        }

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
